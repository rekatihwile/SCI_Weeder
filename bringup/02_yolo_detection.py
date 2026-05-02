"""
bringup/02_yolo_detection.py
-----------------------------
Open cameras and YOLO, run actual detection on one stereo pair. No gantry.

Run with:
    ./run_with_eli_venv.sh bringup/02_yolo_detection.py | tee bringup/logs/02_yolo_detection.log
"""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from pipeline.steps.camera_setup import close_cameras, open_cameras
    from pipeline.steps.detector_setup import build_and_warm_detector

    print("=" * 60)
    print("BRINGUP 02 — YOLO Detection")
    print("=" * 60)

    print("\n--- Building + warming AIDetector ---")
    detector, warmup_info, _model_load_time_s = build_and_warm_detector()
    print(f"  warmup result: {warmup_info}")

    # Open cameras after warmup
    print("\n--- Opening cameras (start_recorder=False) ---")
    cameras = open_cameras(start_recorder=False)

    # Read one valid stereo pair
    print("\n--- Reading stereo pair ---")
    fL, fR = None, None
    for attempt in range(10):
        fL, fR = cameras.read_pair()
        if fL is not None and fR is not None:
            print(f"  Got valid pair on attempt {attempt+1}")
            break
    else:
        print("  ERROR: could not get a valid stereo pair after 10 attempts")
        close_cameras(cameras)
        sys.exit(1)

    # Run detection
    print("\n--- Running detection ---")
    left_dets  = detector.cv_left.detect_rich_points(fL)
    right_dets = detector.cv_right.detect_rich_points(fR)

    print(f"\n  Left  detections: {len(left_dets)}")
    for i, d in enumerate(left_dets):
        pt   = d.get("point", "?")
        cls  = d.get("cls", "?")
        conf = d.get("conf", "?")
        print(f"    [{i}] class={cls}  conf={conf:.3f}  point={pt}")

    print(f"\n  Right detections: {len(right_dets)}")
    for i, d in enumerate(right_dets):
        pt   = d.get("point", "?")
        cls  = d.get("cls", "?")
        conf = d.get("conf", "?")
        print(f"    [{i}] class={cls}  conf={conf:.3f}  point={pt}")

    # Save frames — draw detections if possible
    left_out  = fL.copy()
    right_out = fR.copy()

    for d in left_dets:
        pt = d.get("point")
        if pt:
            cv2.circle(left_out, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), 2)
    for d in right_dets:
        pt = d.get("point")
        if pt:
            cv2.circle(right_out, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), 2)

    left_path  = str(LOGS_DIR / "02_left.jpg")
    right_path = str(LOGS_DIR / "02_right.jpg")
    cv2.imwrite(left_path,  left_out)
    cv2.imwrite(right_path, right_out)
    print(f"\n  Saved: {left_path}")
    print(f"  Saved: {right_path}")

    close_cameras(cameras)
    print("  Cameras closed.")

    # PASS = script completed and printed detection counts (0 is okay)
    print("\n" + "=" * 60)
    print(f"RESULT: PASS  (left={len(left_dets)} right={len(right_dets)} detections)")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
