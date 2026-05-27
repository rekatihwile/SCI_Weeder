"""
bringup/10_survey_photo_burst.py
---------------------------------
Move the gantry to the configured survey position, then save a stereo photo
burst to the repo-root SurveyPics directory.

Run with:
    ./run_with_eli_venv.sh bringup/10_survey_photo_burst.py

Optional:
    ./run_with_eli_venv.sh bringup/10_survey_photo_burst.py --count 12
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAVE_DIR = ROOT / "SurveyPics"
PASS_THRESHOLD_MM = 5.0
DEFAULT_WARMUP_FRAMES = 8


def next_start_index(save_dir: Path) -> int:
    max_idx = -1
    pattern = re.compile(r"^left_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    for path in save_dir.glob("left_*"):
        match = pattern.match(path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def parse_args(default_count: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move to survey position and save stereo photo burst."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=default_count,
        help=f"number of valid stereo pairs to save (default: {default_count})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help=f"camera pairs to discard before saving (default: {DEFAULT_WARMUP_FRAMES})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="seconds to wait between saved stereo pairs (default: 0.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAVE_DIR,
        help=f"directory for saved photos (default: {SAVE_DIR})",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip typed MOVE confirmation",
    )
    return parser.parse_args()


def main():
    from config import GRBL_PORT, SURVEY_BURST_COUNT, SURVEY_POS_X, SURVEY_POS_Y
    from config.survey_params import resolve_burst_count
    from pipeline.steps.camera_setup import close_cameras, open_cameras
    from pipeline.steps.gantry_setup import move_to_survey, open_gantry

    args = parse_args(resolve_burst_count(SURVEY_BURST_COUNT))
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be 0 or greater")
    if args.interval < 0:
        raise ValueError("--interval must be 0 or greater")

    save_dir = args.output_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SURVEY PHOTO BURST")
    print("=" * 60)
    print(f"\n  GRBL_PORT     : {GRBL_PORT}")
    print(f"  SURVEY_POS_X  : {SURVEY_POS_X}")
    print(f"  SURVEY_POS_Y  : {SURVEY_POS_Y}")
    print(f"  SAVE_DIR      : {save_dir}")
    print(f"  BURST_COUNT   : {args.count}")
    print("\n  WARNING: This script will move the gantry (physical motion).")

    if not args.yes:
        answer = input("\nType MOVE to move to survey position and capture photos: ").strip()
        if answer != "MOVE":
            print("  Confirmation not received. Aborting.")
            sys.exit(1)

    gantry = None
    cameras = None
    saved_pairs = 0
    pos_after = None

    try:
        print("\n--- Instantiating Gantry ---")
        gantry = open_gantry(use_mock=False)
        print("  Gantry opened.")

        pos_before = gantry.get_position()
        print(f"  Current position: {pos_before}")

        print(f"\n--- Moving to ({SURVEY_POS_X}, {SURVEY_POS_Y}) ---")
        pos_after = move_to_survey(gantry)
        print(f"  Final position: {pos_after}")

        print("\n--- Opening Cameras ---")
        cameras = open_cameras(start_recorder=False)
        print("  Cameras opened.")

        print(f"\n--- Warming up ({args.warmup} stereo pairs) ---")
        for i in range(args.warmup):
            left, right = cameras.read_pair()
            if left is None or right is None:
                print(f"  warmup {i + 1}: None returned")

        idx = next_start_index(save_dir)
        max_attempts = args.count * 3

        print(f"\n--- Capturing {args.count} valid stereo pairs ---")
        for attempt in range(max_attempts):
            if saved_pairs >= args.count:
                break

            left, right = cameras.read_pair()
            if left is None or right is None:
                print(f"  attempt {attempt + 1}: None returned")
                continue

            left_path = save_dir / f"left_{idx:04d}.jpg"
            right_path = save_dir / f"right_{idx:04d}.jpg"

            left_ok = cv2.imwrite(str(left_path), left)
            right_ok = cv2.imwrite(str(right_path), right)
            if not left_ok or not right_ok:
                raise RuntimeError(f"Failed to write stereo pair {idx} to {save_dir}")

            print(f"  Saved pair {idx:04d}: {left_path.name}, {right_path.name}")
            idx += 1
            saved_pairs += 1

            if args.interval > 0 and saved_pairs < args.count:
                time.sleep(args.interval)

    finally:
        if cameras is not None:
            close_cameras(cameras)
            print("  Cameras closed.")
        if gantry is not None:
            try:
                gantry.serial.close()
            except Exception:
                try:
                    gantry.close()
                except Exception:
                    pass
            print("  Gantry serial closed.")

    passed = saved_pairs == args.count
    if pos_after is not None:
        dx = pos_after["x"] - SURVEY_POS_X
        dy = pos_after["y"] - SURVEY_POS_Y
        dist = math.hypot(dx, dy)
        print(f"\n  Distance from survey target: {dist:.3f} mm")
        passed = passed and dist <= PASS_THRESHOLD_MM
    else:
        print("\n  ERROR: get_position() returned None after move")
        passed = False

    print("\n" + "=" * 60)
    if passed:
        print(f"RESULT: PASS  ({saved_pairs}/{args.count} stereo pairs saved to {save_dir})")
    else:
        print(f"RESULT: FAIL  ({saved_pairs}/{args.count} stereo pairs saved)")
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
