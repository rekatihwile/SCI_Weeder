import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GRBL_PORT, MANUAL_DISPLAY_SCALE, SURVEY_POS_X, SURVEY_POS_Y
from control.coarse_move import TriangulationCoarseMover
from control.fine_align_motion import fine_align_target
import control.fine_align_motion as fine_align_mod
from hardware.cameras import StereoCameras
from hardware.gantry import Gantry
from vision.detectors.manual_detector_local import ManualDetectorLocal
import build_stereo_pixel_error_model


OUT_DIR = ROOT / "planning" / "triangulation_calibration"
OUT_PATH = OUT_DIR / "manual_triangulation_samples.json"


def load_samples(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_samples(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)


def to_xy_list(xy):
    return [float(xy[0]), float(xy[1])]


def main():
    gantry = None
    cameras = None

    samples = load_samples(OUT_PATH)
    next_id = len(samples) + 1

    try:
        gantry = Gantry(GRBL_PORT)
        cameras = StereoCameras()
        try:
            cameras.open()
        except RuntimeError:
            cameras.recover()
            cameras.start_recording()

        survey_detector = ManualDetectorLocal(display_scale=MANUAL_DISPLAY_SCALE)
        survey_detector.window_name = "Survey - Stereo Pair"

        fine_detector = ManualDetectorLocal(display_scale=MANUAL_DISPLAY_SCALE)
        fine_detector.window_name = "Fine Align - Stereo Pair"
        
        coarse_mover = TriangulationCoarseMover()

        print("\n=== MANUAL TRIANGULATION CALIBRATION ===")
        print(f"Samples file: {OUT_PATH}")
        print(f"Existing samples: {len(samples)}")
        print(f"Pixel Error Correction Active: {bool(coarse_mover.pixel_err_model)}")

        gantry.home()

        while True:
            user = input("\nEnter = capture next sample | q = quit: ").strip().lower()
            if user == "q":
                break
            build_stereo_pixel_error_model.main()
            coarse_mover = TriangulationCoarseMover()

            print("\n--- Returning to survey pose ---")
            gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
            gantry.sync_estimate_to_machine()
            time.sleep(0.25)

            print("\n[STEP 1] Click survey stereo point")
            survey_left, survey_right = survey_detector.refine_live(cameras)
            if survey_left is None or survey_right is None:
                print("Survey click cancelled. Skipping sample.")
                continue

            matched_target = {
                "id": next_id,
                "left_px": [float(survey_left[0]), float(survey_left[1])],
                "right_px": [float(survey_right[0]), float(survey_right[1])],
                "score": 1.0,
            }

            solved = coarse_mover.solve_target_from_pose(
                matched_target,
                ref_x=SURVEY_POS_X,
                ref_y=SURVEY_POS_Y,
            )

            # tri_xy is the 'Corrected' move, raw_tri_xy is the baseline geometry
            tri_xy = solved["target_xy_mm"]
            raw_tri_xy = solved.get("raw_triangulated_xy_mm", tri_xy)
            
            print(f"Raw Triangulated = ({raw_tri_xy[0]:.3f}, {raw_tri_xy[1]:.3f}) mm")
            if "pixel_correction_applied_mm" in solved:
                cx, cy = solved["pixel_correction_applied_mm"]
                print(f"Model Correction = ({cx:+.3f}, {cy:+.3f}) mm")

            print("\n[STEP 2] Moving to target XY")
            coarse_mover.move_to_absolute_target(gantry, solved)
            gantry.sync_estimate_to_machine()
            time.sleep(0.25)

            print("\n[STEP 3] Re-click target for fine align")
            fine_align_mod.DETECTOR_MODE = "manual"
            success, actual_entry = fine_align_target(
                gantry=gantry,
                cameras=cameras,
                detector=fine_detector,
                coarse_mover=coarse_mover,
                planned_target=solved,
                actual_hits=[],
                show_debug=True,
            )

            if not success or actual_entry is None:
                print("Fine align failed. Sample not saved.")
                continue

            final_xy = actual_entry["final_xy_mm"]
            
            # Error relative to RAW geometry (for model training)
            err_x = float(final_xy[0] - raw_tri_xy[0])
            err_y = float(final_xy[1] - raw_tri_xy[1])

            # Performance relative to the CORRECTED guess
            residual_x = float(final_xy[0] - tri_xy[0])
            residual_y = float(final_xy[1] - tri_xy[1])

            sample = {
                "sample_id": int(next_id),
                "survey_pose_xy_mm": [float(SURVEY_POS_X), float(SURVEY_POS_Y)],
                "survey_left_px": [float(survey_left[0]), float(survey_left[1])],
                "survey_right_px": [float(survey_right[0]), float(survey_right[1])],
                "raw_triangulated_xy_mm": to_xy_list(raw_tri_xy),
                "final_xy_mm": [float(final_xy[0]), float(final_xy[1])],
                "error_xy_mm": [err_x, err_y],
                "correction_active": bool(coarse_mover.pixel_err_model),
                "timestamp_unix": time.time(),
            }

            samples.append(sample)
            save_samples(OUT_PATH, samples)

            print("\n" + "="*40)
            print("         SAMPLE SAVED")
            print("="*40)
            print(f"Sample ID        : {sample['sample_id']}")
            print(f"Initial Guess    : ({tri_xy[0]:.3f}, {tri_xy[1]:.3f})")
            print(f"PD Ground Truth  : ({final_xy[0]:.3f}, {final_xy[1]:.3f})")
            print("-" * 40)
            print(f"Residual Error X : {residual_x:+.4f} mm")
            print(f"Residual Error Y : {residual_y:+.4f} mm")
            print(f"Total Distance   : {float(np.hypot(residual_x, residual_y)):.4f} mm")
            print("="*40)

            next_id += 1

    finally:
        if cameras: cameras.close()
        if gantry: gantry.close()


if __name__ == "__main__":
    import numpy as np # Ensure numpy is available for hypot
    main()
