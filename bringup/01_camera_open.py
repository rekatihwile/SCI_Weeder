"""
bringup/01_camera_open.py
--------------------------
Open cameras and read frames. No YOLO. No gantry.

Run with:
    ./run_with_eli_venv.sh bringup/01_camera_open.py | tee bringup/logs/01_camera_open.log
"""

import sys
import time
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_PAIRS = 30


def main():
    from hardware.cameras import StereoCameras

    print("=" * 60)
    print("BRINGUP 01 — Camera Open")
    print("=" * 60)

    cameras = StereoCameras()
    print("\n--- Opening cameras (start_recorder=False) ---")
    cameras.open(start_recorder=False)

    print("\n--- Camera Properties ---")
    for name, cap in [("Left", cameras.left), ("Right", cameras.right)]:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        print(f"  {name}: {w}x{h} @ {fps:.1f} fps  fourcc={fourcc_str!r}")

    # health check if available
    if hasattr(cameras, "camera_health_check"):
        print("\n--- camera_health_check(10) ---")
        cameras.camera_health_check(10)

    print(f"\n--- Reading {TOTAL_PAIRS} stereo pairs ---")
    valid = 0
    first_valid_left = None
    first_valid_right = None

    for i in range(TOTAL_PAIRS):
        try:
            fL, fR = cameras.read_pair()
            if fL is not None and fR is not None:
                valid += 1
                if first_valid_left is None:
                    first_valid_left = fL
                    first_valid_right = fR
            else:
                print(f"  pair {i+1}: None returned")
        except Exception as e:
            print(f"  pair {i+1}: read_pair() raised {e}")

    print(f"\n  Valid pairs: {valid}/{TOTAL_PAIRS}")

    # Save first valid pair
    if first_valid_left is not None:
        left_path  = str(LOGS_DIR / "01_left.jpg")
        right_path = str(LOGS_DIR / "01_right.jpg")
        cv2.imwrite(left_path,  first_valid_left)
        cv2.imwrite(right_path, first_valid_right)
        print(f"  Saved: {left_path}")
        print(f"  Saved: {right_path}")
    else:
        print("  No valid frames — nothing saved.")

    cameras.close()
    print("  Cameras closed.")

    passed = valid == TOTAL_PAIRS
    print("\n" + "=" * 60)
    if passed:
        print(f"RESULT: PASS  ({valid}/{TOTAL_PAIRS} pairs)")
    else:
        print(f"RESULT: FAIL  ({valid}/{TOTAL_PAIRS} pairs)")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
