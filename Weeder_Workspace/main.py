from config import DETECTOR_MODE, GRBL_PORT, SURVEY_POS_X, SURVEY_POS_Y

from control.coarse_move import TriangulationCoarseMover
from control.fine_align import fine_align_target
from control.strike.strike_patterns import fire_target
from hardware.cameras import StereoCameras
from hardware.gantry import Gantry
from planning.target_planner import plan_targets
from ui.terminal import clear_current_target_line, print_workspace_plan, show_current_target
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

    try:
        while state != "DONE":
            if state == "INIT":
                gantry = Gantry(GRBL_PORT)
                cameras = StereoCameras()
                detector = build_detector()
                coarse_mover = TriangulationCoarseMover()
                state = "HOME"

            elif state == "HOME":
                cameras.open()
                gantry.home()
                state = "SURVEY"

            elif state == "SURVEY":
                gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
                state = "DETECT"

            elif state == "DETECT":
                left_points, right_points = detector.detect_live(cameras)
                state = "MATCH"

            elif state == "MATCH":
                matched_targets, unmatched_left, unmatched_right = match_points(
                    left_points,
                    right_points,
                    verbose=True,
                )
                state = "PLAN"

            elif state == "PLAN":
                target_queue = plan_targets(matched_targets)
                absolute_targets = []
                for target in target_queue:
                    solved = coarse_mover.solve_target_from_survey(
                        target,
                        survey_x=SURVEY_POS_X,
                        survey_y=SURVEY_POS_Y,
                    )
                    absolute_targets.append(solved)
                print_workspace_plan(absolute_targets)
                state = "EXECUTE"

            elif state == "EXECUTE":
                total = len(absolute_targets)
                for i, solved in enumerate(absolute_targets, start=1):
                    show_current_target(i, total, solved)
                    coarse_mover.move_to_absolute_target(gantry, solved)

                    aligned = fine_align_target(gantry, cameras, detector)

                    if aligned:
                        fire_target(gantry, solved)
                    else:
                        print("Skipped firing: fine align failed.")

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
