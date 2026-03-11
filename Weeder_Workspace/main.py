from config import DETECTOR_MODE, GRBL_PORT, SURVEY_POS_X, SURVEY_POS_Y

from control.coarse_move import TriangulationCoarseMover
from control.fine_align import fine_align_target
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
from vision.detectors.ai_detector import AIDetector
from vision.detectors.manual_detector_local import ManualDetectorLocal
from vision.matching import match_points


def build_detector():
    if DETECTOR_MODE == "manual":
        return ManualDetectorLocal(display_scale=2.5)
    if DETECTOR_MODE == "ai":
        return AIDetector(display_scale=2.0)
    raise ValueError(f"Unknown DETECTOR_MODE: {DETECTOR_MODE}")


def main():
    gantry = None
    cameras = None

    state = "INIT"
    detector = None
    coarse_mover = None
    left_points = []
    right_points = []
    matched_targets = []
    target_queue = []
    absolute_targets = []
    actual_hits = []

    try:
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
                gantry.home()
                state = "SURVEY"

            elif state == "SURVEY":
                gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
                print_global_survey_ready(SURVEY_POS_X, SURVEY_POS_Y)
                state = "SURVEY_CONFIRM"

            elif state == "SURVEY_CONFIRM":
                user = input("Enter = survey | q = quit: ").strip().lower()

                if user == "q":
                    state = "DONE"
                else:
                    state = "DETECT"

            elif state == "DETECT":
                left_points, right_points = coarse_mover.detect_stable_points(
                    cameras,
                    detector,
                    detector_mode=DETECTOR_MODE,
                    burst_count=5,
                    min_hits=3,
                    cluster_radius_px=12.0,
                )
                state = "MATCH"

            elif state == "MATCH":
                matched_targets, unmatched_left, unmatched_right = match_points(
                    left_points,
                    right_points,
                    verbose=True,
                )
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
                print_workspace_plan(target_queue)
                state = "EXECUTE"

            elif state == "EXECUTE":
                total = len(target_queue)

                for i, solved in enumerate(target_queue, start=1):
                    if coarse_mover.is_duplicate_of_actual(
                        solved["target_xy_mm"],
                        actual_hits,
                        tol_mm=8.0,
                    ):
                        print_skip_target(i, total, solved, "Already covered by a previous PD lock.")
                        continue

                    show_current_target(i, total, solved)
                    coarse_mover.move_to_absolute_target(gantry, solved)

                    aligned, actual_entry = fine_align_target(
                        gantry,
                        cameras,
                        detector,
                        coarse_mover,
                        solved,
                        actual_hits,
                    )

                    if aligned:
                        actual_hits.append(actual_entry)
                        print
                        fire_target(gantry, solved)
                    else:
                        print_skip_target(i, total, solved, "Fine align failed")

                clear_current_target_line()
                print("Finished all planned targets.")
                state = "DONE"

        print("\n=== DONE ===")

    except KeyboardInterrupt:
        clear_current_target_line()
        print("\nInterrupted by user.")

    except Exception as e:
        clear_current_target_line()
        print(f"\nERROR: {e}")

    finally:
        if cameras is not None:
            cameras.close()
        if gantry is not None:
            gantry.close()


if __name__ == "__main__":
    main()