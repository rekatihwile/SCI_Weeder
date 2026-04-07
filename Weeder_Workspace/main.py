from config import (
    DETECTOR_MODE,
    GRBL_PORT,
    SURVEY_POS_X,
    SURVEY_POS_Y,
    TRIANGULATION_ONLY_MODE,
    SHOW_TRIANGULATION_PLOT,
    SHOW_MATCH_DEBUG_WINDOW,
    SAVE_MATCH_DEBUG_IMAGE,
    HAS_DISPLAY,
    MANUAL_DISPLAY_SCALE,
    AI_DISPLAY_SCALE,
    AI_CONFIDENCE,
    AI_MIN_STABLE_VIEWS,
    SURVEY_BURST_COUNT,
    SURVEY_MIN_HITS,
    SURVEY_CLUSTER_RADIUS_PX,
    FINE_ALIGN_SETTLE_FRAMES,
    SURVEY_TARGET_CLASSES,
)

from control.coarse_move import TriangulationCoarseMover, is_in_workspace, is_in_workspace
from control.fine_align import fine_align_target, close_fine_align_window, close_fine_align_window
from control.strike.strike_patterns import fire_target
from hardware.cameras import StereoCameras
from hardware.gantry import Gantry
from planning.target_planner import plan_targets
from ui.terminal import (
    clear_current_target_line,
    print_workspace_plan,
    show_current_target,
    print_target_result,
    print_skip_target,
    print_global_survey_ready,
    print_global_survey_results,
)
from ui.workspace_plot import show_workspace_triangulation_map
from ui.triangulation_debug import show_match_debug_view
from vision.detectors.ai_detector import AIDetector
from vision.detectors.manual_detector_local import ManualDetectorLocal
from vision.matching import match_points
from run_logger import RunSession  # Adjust import based on where you put it
import time

def build_detector():
    if DETECTOR_MODE == "manual":
        return ManualDetectorLocal(display_scale=MANUAL_DISPLAY_SCALE)
    if DETECTOR_MODE == "ai":
        return AIDetector(
            display_scale=AI_DISPLAY_SCALE,
            conf=AI_CONFIDENCE,
            min_stable_views=AI_MIN_STABLE_VIEWS,
        )
    raise ValueError(f"Unknown DETECTOR_MODE: {DETECTOR_MODE}")



def main():
    gantry = None
    cameras = None
    session = None  # Add session to the initial variables

    state = "INIT"
    detector     = None
    coarse_mover = None
    # _detections  → full dicts {"point","box","views"} (AI) or plain tuples (manual)
    # _points      → plain (x,y) tuples only, used by UI / display functions
    left_detections  = []
    right_detections = []
    left_points      = []
    right_points     = []
    matched_targets = []
    target_queue    = []
    actual_hits     = []

    try:
        # --- 1. INITIALIZE THE RUN SESSION LOGGER ---
        # This instantly starts the dual-terminal logging and prepares the video thread
        session = RunSession(base_folder="run_data")

        while state != "DONE":
            if state == "INIT":
                gantry = Gantry(GRBL_PORT)
                cameras = StereoCameras()
                detector = build_detector()
                coarse_mover = TriangulationCoarseMover()
                coarse_mover.clear_actual_targets_log()
                state = "HOME"

            elif state == "HOME":
                
                cameras.open()
                #session.start_recording()
                # --- 2. ATTACH THE BACKGROUND VIDEO RECORDER ---
                #cameras.attach_recorder(session.recorder)
                # gantry.home()
                state = "SURVEY"

            elif state == "SURVEY":
                gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
                print_global_survey_ready(SURVEY_POS_X, SURVEY_POS_Y)
                state = "SURVEY_CONFIRM"

            elif state == "SURVEY_CONFIRM":
                user = input("Enter = survey | q = quit: ").strip().lower()
                state = "DONE" if user == "q" else "DETECT"

            elif state == "DETECT":
                left_detections, right_detections = coarse_mover.detect_stable_points(
                    cameras,
                    detector,
                    detector_mode=DETECTOR_MODE,
                    burst_count=SURVEY_BURST_COUNT,
                    min_hits=SURVEY_MIN_HITS,
                    cluster_radius_px=SURVEY_CLUSTER_RADIUS_PX,
                )
                # Plain tuples for all UI / display calls
                def _to_pt(d): return d["point"] if isinstance(d, dict) else d
                left_points  = [_to_pt(d) for d in left_detections]
                right_points = [_to_pt(d) for d in right_detections]
                state = "MATCH"

            elif state == "MATCH":
                matched_targets, unmatched_left, unmatched_right = match_points(
                    left_detections,
                    right_detections,
                    verbose=True,
                )

                # --- 3. SAVE THE HIGH-RES SURVEY IMAGES ---
                if coarse_mover.last_survey_frameL is not None and coarse_mover.last_survey_frameR is not None:
                    session.save_survey_images(coarse_mover.last_survey_frameL, coarse_mover.last_survey_frameR)

                print_global_survey_results(len(left_points), len(right_points), len(matched_targets))
                user = input("Enter = accept global survey | r = rescan | q = quit: ").strip().lower()

                if user == "r":
                    state = "DETECT"
                elif user == "q":
                    state = "DONE"
                else:
                    state = "PLAN"

            elif state == "PLAN":
                absolute_targets = coarse_mover.solve_all_from_pose(
                    matched_targets,
                    ref_x=SURVEY_POS_X,
                    ref_y=SURVEY_POS_Y,
                )

                target_queue = plan_targets(
                    absolute_targets,
                    start_xy=gantry.get_estimated_xy(),
                )

                coarse_mover.save_workspace_targets(
                    target_queue,
                    filename="predicted_workspace_targets.json",
                )

                if SHOW_TRIANGULATION_PLOT:
                    show_workspace_triangulation_map(
                        target_queue,
                        survey_xy=(SURVEY_POS_X, SURVEY_POS_Y),
                        save_path=coarse_mover.planning_dir / "triangulation_overview.png",
                        show_window=HAS_DISPLAY,
                    )

                if coarse_mover.last_survey_frameL is not None and coarse_mover.last_survey_frameR is not None:
                    show_match_debug_view(
                        coarse_mover.last_survey_frameL,
                        coarse_mover.last_survey_frameR,
                        left_points,
                        right_points,
                        matched_targets,
                        absolute_targets,
                        save_path=(coarse_mover.planning_dir / "triangulation_debug_view.png") if SAVE_MATCH_DEBUG_IMAGE else None,
                        show_window=SHOW_MATCH_DEBUG_WINDOW,
                    )

                print_workspace_plan(target_queue)
                state = "EXECUTE"

            elif state == "EXECUTE":
                total = len(target_queue)

                for i, solved in enumerate(target_queue, start=1):
                    # --- 4. START THE PLANT TIMER ---
                    session.start_plant_timer(plant_id=i)

                    # ── bounds check before doing anything ─────────────────
                    tx, ty = solved["target_xy_mm"]
                    if not is_in_workspace(tx, ty):
                        print_skip_target(i, total, solved,
                            f"Outside workspace bounds ({tx:.1f}, {ty:.1f}) mm.")
                        session.end_plant_timer(plant_id=i, status="Skipped (Out of Bounds)")
                        continue

                    if coarse_mover.is_duplicate_of_actual(
                        solved["target_xy_mm"],
                        actual_hits,
                        tol_mm=8.0,
                    ):
                        print_skip_target(i, total, solved, "Already covered by a previous PD lock.")
                        # --- 5A. END TIMER (SKIPPED DUPLICATE) ---
                        session.end_plant_timer(plant_id=i, status="Skipped (Duplicate)")
                        continue

                    show_current_target(i, total, solved)
                    moved = coarse_mover.move_to_absolute_target(gantry, solved)
                    if not moved:
                        # move_to_absolute_target already printed a warning
                        session.end_plant_timer(plant_id=i, status="Skipped (Out of Bounds)")
                        continue

                    if TRIANGULATION_ONLY_MODE:
                        gantry.sync_estimate_to_machine()
                        final_xy = gantry.get_estimated_xy()
                        actual_entry = {
                            "planned_xy_mm": [float(solved["target_xy_mm"][0]), float(solved["target_xy_mm"][1])],
                            "selected_local_xy_mm": [float(solved["target_xy_mm"][0]), float(solved["target_xy_mm"][1])],
                            "final_xy_mm": [float(final_xy[0]), float(final_xy[1])],
                            "left_px": [float(solved["source_target"]["left_px"][0]), float(solved["source_target"]["left_px"][1])],
                            "right_px": [float(solved["source_target"]["right_px"][0]), float(solved["source_target"]["right_px"][1])],
                            "score": float(solved["source_target"].get("score", 1.0)),
                        }
                        actual_hits.append(actual_entry)
                        coarse_mover.append_actual_target(
                            solved, solved, final_xy,
                            filename="actual_pd_targets.json",
                        )
                        print_target_result(i, total, solved, actual_entry)
                        session.end_plant_timer(plant_id=i, status="Triangulation Only")
                        user = input("Triangulation only: Enter = next | q = quit: ").strip().lower()
                        if user == "q":
                            state = "DONE"
                            break
                        continue

                    aligned, actual_entry = fine_align_target(
                        gantry,
                        cameras,
                        detector,
                        coarse_mover,
                        solved,
                        actual_hits,
                        settle_frames=FINE_ALIGN_SETTLE_FRAMES,
                        show_debug=HAS_DISPLAY,
                        survey_targets=target_queue,   # enables constellation re-ID
                        target_idx=i,
                        total_targets=total,
                    )
                    print(f"[DEBUG] fine_align returned aligned={aligned}")

                    if aligned:
                        actual_hits.append(actual_entry)
                        print_target_result(i, total, solved, actual_entry)
                        print("[DEBUG] Entering strike...")
                        fire_target(gantry, solved)
                        session.end_plant_timer(plant_id=i, status="Locked and Fired")
                    else:
                        print_skip_target(i, total, solved, "Fine align failed")
                        session.end_plant_timer(plant_id=i, status="Failed (Fine Align)")

                # Close the shared fine-align window once all targets are done
                close_fine_align_window()
                close_fine_align_window()
                clear_current_target_line()
                print("\n  All targets complete.")
                state = "DONE"

        print("\n=== DONE ===")

    except KeyboardInterrupt:
        clear_current_target_line()
        print("\nInterrupted by user.")

    except Exception as e:
        clear_current_target_line()
        print(f"\nERROR: {e}")

    finally:
        # --- 6. CRITICAL SHUTDOWN SEQUENCE ---
        if cameras is not None:
            cameras.close()
        if gantry is not None:
            gantry.close()
        
        # Ensures the video thread safely stops encoding and writes the JSON
        
        if session is not None:
            session.end_session()


if __name__ == "__main__":
    main()