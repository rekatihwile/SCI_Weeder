import json
import math
from datetime import datetime

from config import (
    HOMING,
    DETECTOR_MODE,
    GRBL_PORT,
    SURVEY_POS_X,
    SURVEY_POS_Y,
    TRIANGULATION_ONLY_MODE,
    FULL_AUTO,
    SHOW_TRIANGULATION_PLOT,
    SHOW_MATCH_DEBUG_WINDOW,
    SAVE_MATCH_DEBUG_IMAGE,
    HAS_DISPLAY,
    MANUAL_DISPLAY_SCALE,
    AI_DISPLAY_SCALE,
    AI_CONFIDENCE,
    SURVEY_BURST_COUNT,
    SURVEY_MIN_HITS,
    SURVEY_CLUSTER_RADIUS_PX,
    SURVEY_TARGET_CLASSES,
    FINE_ALIGN_SETTLE_FRAMES,
    RECORD_TRIAL,
    TRIAL_RECORDINGS_DIR,
    WORKSPACE_X_MIN,
    WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Y_MAX,
    ENABLE_EXPERIMENT_LOGGING,
    EXPERIMENT_TRIAL_ID,
    EXPERIMENT_TRIAL_TYPE,
    EXPERIMENT_LAYOUT_TYPE,
    EXPECTED_WEED_COUNT,
    EXPECTED_KALE_COUNT,
    EXPERIMENT_NOTES,
)

from control.coarse_move import TriangulationCoarseMover, is_in_workspace
from control.fine_align import (
    fine_align_target,
    close_fine_align_window,
)
from control.strike import fire_target
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


def build_detector():
    if DETECTOR_MODE == "manual":
        return ManualDetectorLocal(display_scale=MANUAL_DISPLAY_SCALE)
    if DETECTOR_MODE == "ai":
        return AIDetector(
            display_scale=AI_DISPLAY_SCALE,
            conf=AI_CONFIDENCE,
        )
    raise ValueError(f"Unknown DETECTOR_MODE: {DETECTOR_MODE}")


def _flush_camera_buffer(cameras, n=8):
    for _ in range(n):
        cameras.read_pair()


def _save_manifest(manifest, timestamp):
    TRIAL_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRIAL_RECORDINGS_DIR / f"manifest_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Saved → {path}")


def _print_final_targets(actual_hits):
    if not actual_hits:
        return
    print("\n=== FINAL ALIGNED TARGETS (PD Locked) ===")
    print(f"  {'#':>3}  {'Planned (mm)':^26}  {'Final (mm)':^26}")
    for i, hit in enumerate(actual_hits, start=1):
        cx, cy = hit.get("planned_xy_mm", [0.0, 0.0])
        fx, fy = hit.get("final_xy_mm", [0.0, 0.0])
        print(f"  {i:>3}  ({cx:>10.2f}, {cy:>10.2f})  →  ({fx:>10.2f}, {fy:>10.2f})")
    print(f"  {len(actual_hits)} target(s) locked.\n")


def _save_metrics(logger, status):
    if logger is None:
        return
    try:
        logger.end_run(status=status)
        logger.save_csvs()
        logger.save_json()
        logger.print_summary()
    except Exception as e:
        print(f"[METRICS] Warning: could not save metrics: {e}")


def main():
    gantry = None
    cameras = None
    state = "INIT"
    detector = None
    coarse_mover = None
    logger = None

    left_detections = []
    right_detections = []
    left_points = []
    right_points = []
    matched_targets = []
    target_queue = []
    actual_hits = []

    trial_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "trial_timestamp": trial_timestamp,
        "survey_position_mm": [SURVEY_POS_X, SURVEY_POS_Y],
        "targets": [],
    }

    try:
        while state != "DONE":
            if state == "INIT":
                gantry = Gantry(GRBL_PORT)
                cameras = StereoCameras()
                detector = build_detector()
                coarse_mover = TriangulationCoarseMover()
                coarse_mover.clear_actual_targets_log()
                if ENABLE_EXPERIMENT_LOGGING:
                    from metrics.experiment_logger import ExperimentLogger
                    logger = ExperimentLogger(config={
                        "trial_id": EXPERIMENT_TRIAL_ID,
                        "trial_type": EXPERIMENT_TRIAL_TYPE,
                        "layout_type": EXPERIMENT_LAYOUT_TYPE,
                        "expected_weed_count": EXPECTED_WEED_COUNT,
                        "expected_kale_count": EXPECTED_KALE_COUNT,
                        "workspace_width_mm": WORKSPACE_X_MAX - WORKSPACE_X_MIN,
                        "workspace_height_mm": WORKSPACE_Y_MAX - WORKSPACE_Y_MIN,
                        "notes": EXPERIMENT_NOTES,
                    })
                state = "HOME"

            elif state == "HOME":
                cameras.open()
                if HOMING:
                    gantry.home()
                if logger is not None:
                    logger.start_run()
                state = "SURVEY"

            elif state == "SURVEY":
                gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
                print_global_survey_ready(SURVEY_POS_X, SURVEY_POS_Y)
                state = "SURVEY_CONFIRM"

            elif state == "SURVEY_CONFIRM":
                state = "DETECT"

            elif state == "DETECT":
                _flush_camera_buffer(cameras, n=8)

                if logger is not None:
                    logger.start_section("survey")
                left_detections, right_detections = coarse_mover.detect_stable_points(
                    cameras,
                    detector,
                    detector_mode=DETECTOR_MODE,
                    burst_count=SURVEY_BURST_COUNT,
                    min_hits=SURVEY_MIN_HITS,
                    cluster_radius_px=SURVEY_CLUSTER_RADIUS_PX,
                    survey_classes=SURVEY_TARGET_CLASSES,
                )
                if logger is not None:
                    logger.end_section("survey")
                    logger.run["num_detections_left"] = len(left_detections)
                    logger.run["num_detections_right"] = len(right_detections)

                def _to_pt(d):
                    return d["point"] if isinstance(d, dict) else d

                left_points = [_to_pt(d) for d in left_detections]
                right_points = [_to_pt(d) for d in right_detections]
                state = "MATCH"

            elif state == "MATCH":
                if logger is not None:
                    logger.start_section("stereo_matching")
                matched_targets, _, _ = match_points(
                    left_detections,
                    right_detections,
                    verbose=True,
                )
                if logger is not None:
                    logger.end_section("stereo_matching")
                    logger.run["num_stereo_matches"] = len(matched_targets)

                print_global_survey_results(len(left_points), len(right_points), len(matched_targets))
                if FULL_AUTO:
                    print("[AUTO] Accepting survey result automatically.")
                    state = "PLAN"
                else:
                    user = input("Enter = accept global survey | r = rescan | q = quit: ").strip().lower()
                    if user == "r":
                        state = "DETECT"
                    elif user == "q":
                        state = "DONE"
                    else:
                        state = "PLAN"

            elif state == "PLAN":
                if logger is not None:
                    logger.start_section("triangulation")
                coarse_mover.fit_epipolar(matched_targets)
                absolute_targets = coarse_mover.solve_all_from_pose(
                    matched_targets,
                    ref_x=SURVEY_POS_X,
                    ref_y=SURVEY_POS_Y,
                )
                if logger is not None:
                    logger.end_section("triangulation")

                if logger is not None:
                    logger.start_section("planning")
                target_queue = plan_targets(
                    absolute_targets,
                    start_xy=gantry.get_estimated_xy(),
                )
                if logger is not None:
                    logger.end_section("planning")
                    logger.run["num_targets_planned"] = len(target_queue)
                    logger.compute_path_metrics(target_queue, start_xy=gantry.get_estimated_xy())

                coarse_mover.all_planned_targets = target_queue

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

                if (SAVE_MATCH_DEBUG_IMAGE or SHOW_MATCH_DEBUG_WINDOW) and \
                        coarse_mover.last_survey_frameL is not None and \
                        coarse_mover.last_survey_frameR is not None:
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

                state = "EXECUTE"

            elif state == "EXECUTE":
                total = len(target_queue)
                prev_xy = gantry.get_estimated_xy()

                for i, solved in enumerate(target_queue, start=1):
                    tx, ty = solved["target_xy_mm"]
                    src = solved.get("source_target", {})
                    travel_dist = round(math.hypot(tx - prev_xy[0], ty - prev_xy[1]), 2)

                    if logger is not None:
                        logger.start_target(i, {
                            "x_target_mm": tx,
                            "y_target_mm": ty,
                            "x_commanded_mm": tx,
                            "y_commanded_mm": ty,
                            "travel_distance_mm": travel_dist,
                            "class_id": src.get("left_cls"),
                            "confidence": src.get("conf"),
                        })

                    if not is_in_workspace(tx, ty):
                        print_skip_target(i, total, solved, f"Outside workspace bounds ({tx:.1f}, {ty:.1f}) mm.")
                        if logger is not None:
                            logger.end_target(i, {"status": "skipped_out_of_bounds"})
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": None,
                            "reid_protocol": None,
                            "status": "skipped_out_of_bounds",
                        })
                        continue

                    if coarse_mover.is_duplicate_of_actual(
                        solved["target_xy_mm"],
                        actual_hits,
                        tol_mm=8.0,
                    ):
                        print_skip_target(i, total, solved, "Already covered by a previous PD lock.")
                        if logger is not None:
                            logger.end_target(i, {"status": "skipped_duplicate"})
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": None,
                            "reid_protocol": None,
                            "status": "skipped_duplicate",
                        })
                        continue

                    show_current_target(i, total, solved)
                    cameras.clear_overlay()
                    cameras.set_recording_status([
                        f"Target {i}/{total}",
                        f"Coarse: ({tx:.1f}, {ty:.1f}) mm",
                        "Moving to target...",
                    ])

                    if logger is not None:
                        logger.start_target_section(i, "travel")
                    moved = coarse_mover.move_to_absolute_target(gantry, solved)
                    if logger is not None:
                        logger.end_target_section(i, "travel")

                    if not moved:
                        if logger is not None:
                            logger.end_target(i, {"status": "skipped_move_failed"})
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": None,
                            "reid_protocol": None,
                            "status": "skipped_move_failed",
                        })
                        continue

                    prev_xy = (tx, ty)

                    if TRIANGULATION_ONLY_MODE:
                        gantry.sync_estimate_to_machine()
                        final_xy = gantry.get_estimated_xy()
                        actual_entry = {
                            "planned_xy_mm": [float(tx), float(ty)],
                            "selected_local_xy_mm": [float(tx), float(ty)],
                            "final_xy_mm": [float(final_xy[0]), float(final_xy[1])],
                            "left_px": src.get("left_px"),
                            "right_px": src.get("right_px"),
                            "score": float(src.get("score", 1.0)),
                        }
                        actual_hits.append(actual_entry)
                        coarse_mover.append_actual_target(
                            solved, solved, final_xy,
                            filename="actual_pd_targets.json",
                        )
                        print_target_result(i, total, solved, actual_entry)
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": list(final_xy),
                            "reid_protocol": None,
                            "status": "triangulation_only",
                        })
                        if logger is not None:
                            logger.end_target(i, {
                                "x_final_mm": float(final_xy[0]),
                                "y_final_mm": float(final_xy[1]),
                                "fired": False,
                                "status": "triangulation_only",
                            })
                        if not FULL_AUTO:
                            user = input("Triangulation only: Enter = next | q = quit: ").strip().lower()
                            if user == "q":
                                state = "DONE"
                                break
                        continue

                    cameras.set_recording_status([
                        f"Target {i}/{total}",
                        f"Coarse: ({tx:.1f}, {ty:.1f}) mm",
                        "Fine aligning...",
                    ])

                    if logger is not None:
                        logger.start_target_section(i, "pd")
                    aligned, actual_entry = fine_align_target(
                        gantry,
                        cameras,
                        detector,
                        coarse_mover,
                        solved,
                        actual_hits,
                        settle_frames=FINE_ALIGN_SETTLE_FRAMES,
                        show_debug=HAS_DISPLAY,
                        target_idx=i,
                        total_targets=total,
                    )
                    if logger is not None:
                        logger.end_target_section(i, "pd")

                    if aligned:
                        actual_hits.append(actual_entry)
                        print_target_result(i, total, solved, actual_entry)

                        if logger is not None:
                            logger.start_target_section(i, "fire")
                        fire_target(gantry, solved)
                        if logger is not None:
                            logger.end_target_section(i, "fire")

                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": actual_entry.get("final_xy_mm"),
                            "reid_protocol": actual_entry.get("reid_protocol"),
                            "status": "locked_fired",
                        })
                        if logger is not None:
                            final_xy = actual_entry.get("final_xy_mm") or [tx, ty]
                            logger.end_target(i, {
                                "x_final_mm": float(final_xy[0]),
                                "y_final_mm": float(final_xy[1]),
                                "fired": True,
                                "pd_converged": True,
                                "status": "locked_fired",
                            })
                    else:
                        cameras.set_recording_status([
                            f"Target {i}/{total}",
                            f"Coarse: ({tx:.1f}, {ty:.1f}) mm",
                            "FAILED: fine align",
                        ])
                        print_skip_target(i, total, solved, "Fine align failed")
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": None,
                            "reid_protocol": None,
                            "status": "failed_fine_align",
                        })
                        if logger is not None:
                            logger.end_target(i, {
                                "pd_converged": False,
                                "fired": False,
                                "status": "failed_fine_align",
                            })

                close_fine_align_window()
                clear_current_target_line()
                print("\n  All targets complete.")
                _print_final_targets(actual_hits)
                _save_metrics(logger, "complete")
                state = "DONE"

        print("\n=== DONE ===")

    except KeyboardInterrupt:
        clear_current_target_line()
        print("\nInterrupted by user.")
        _save_metrics(logger, "user_aborted")

    except Exception as e:
        clear_current_target_line()
        print(f"\nERROR: {e}")
        _save_metrics(logger, "failed")

    finally:
        _save_manifest(manifest, trial_timestamp)
        if cameras is not None:
            cameras.stop_recording()
            cameras.close()
        if gantry is not None:
            gantry.close()


if __name__ == "__main__":
    main()
