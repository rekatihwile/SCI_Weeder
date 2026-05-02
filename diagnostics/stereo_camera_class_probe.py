"""
Probe StereoCameras class the same way main.py does,
without touching YOLO/matching/rectification.

Run from repo root:
    python diagnostics/stereo_camera_class_probe.py
"""

import sys
import os
import cv2
from pathlib import Path

# Mirror main.py's sys.path so imports resolve identically.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hardware.cameras import StereoCameras


def _fourcc_str(val):
    try:
        return "".join([chr((int(val) >> (8 * i)) & 0xFF) for i in range(4)])
    except Exception:
        return str(val)


def _print_props(label, cap):
    w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fcc = cap.get(cv2.CAP_PROP_FOURCC)
    print(
        f"  [{label}] isOpened={cap.isOpened()} "
        f"W={w} H={h} FPS={fps} FOURCC={int(fcc)}({_fourcc_str(fcc)})"
    )


def main():
    diag_dir = REPO_ROOT / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    print("\n=== stereo_camera_class_probe ===")
    print(f"repo_root={REPO_ROOT}")

    cams = StereoCameras()
    cams.open(start_recorder=False)

    print("\n--- Camera properties after open() ---")
    _print_props("left",  cams.left)
    _print_props("right", cams.right)

    print("\n--- camera_health_check(n=10) ---")
    cams.camera_health_check(n=10)

    print("\n--- read_pair() x30 ---")
    ok = 0
    fail = 0
    first_left = None
    first_right = None

    for i in range(30):
        try:
            fl, fr = cams.read_pair()
            if fl is not None and fr is not None:
                ok += 1
                if first_left is None:
                    first_left  = fl
                    first_right = fr
            else:
                fail += 1
                print(f"  [{i+1}/30] NONE frame returned (fl={fl is None} fr={fr is None})")
        except RuntimeError as e:
            fail += 1
            print(f"  [{i+1}/30] RuntimeError: {e}")

    print(f"\nread_pair results: {ok}/30 OK, {fail}/30 failed")

    if first_left is not None:
        left_path  = diag_dir / "stereo_class_left.jpg"
        right_path = diag_dir / "stereo_class_right.jpg"
        cv2.imwrite(str(left_path),  first_left)
        cv2.imwrite(str(right_path), first_right)
        print(f"Saved first valid left  -> {left_path}")
        print(f"Saved first valid right -> {right_path}")
    else:
        print("No valid frames — no images saved.")

    cams.close()
    print("\n=== probe complete ===")


if __name__ == "__main__":
    main()
