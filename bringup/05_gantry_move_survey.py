"""
bringup/05_gantry_move_survey.py
---------------------------------
Move to survey position after homing/status is trusted.
Requires explicit typed confirmation before any movement.

Run with:
    ./run_with_eli_venv.sh bringup/05_gantry_move_survey.py
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS_THRESHOLD_MM = 5.0


def main():
    from config import GRBL_PORT, SURVEY_POS_X, SURVEY_POS_Y
    from pipeline.steps.gantry_setup import move_to_survey, open_gantry

    print("=" * 60)
    print("BRINGUP 05 — Gantry Move to Survey Position")
    print("=" * 60)
    print(f"\n  GRBL_PORT     : {GRBL_PORT}")
    print(f"  SURVEY_POS_X  : {SURVEY_POS_X}")
    print(f"  SURVEY_POS_Y  : {SURVEY_POS_Y}")
    print("\n  WARNING: This script will move the gantry (physical motion).")

    answer = input(f"\nType MOVE to move to SURVEY_POS_X/SURVEY_POS_Y: ").strip()
    if answer != "MOVE":
        print("  Confirmation not received. Aborting.")
        sys.exit(1)

    print("\n--- Instantiating Gantry ---")
    gantry = open_gantry(use_mock=False)
    print("  Gantry opened.")

    pos_before = gantry.get_position()
    print(f"  Current position: {pos_before}")

    print(f"\n--- Moving to ({SURVEY_POS_X}, {SURVEY_POS_Y}) ---")
    pos_after = move_to_survey(gantry)
    print(f"\n  Final position: {pos_after}")

    gantry.serial.close()
    print("  Serial closed.")

    # Check if within 5mm of target
    passed = False
    if pos_after is not None:
        dx = pos_after["x"] - SURVEY_POS_X
        dy = pos_after["y"] - SURVEY_POS_Y
        dist = math.hypot(dx, dy)
        print(f"\n  Distance from target: {dist:.3f} mm  (threshold: {PASS_THRESHOLD_MM} mm)")
        passed = dist <= PASS_THRESHOLD_MM
    else:
        print("\n  ERROR: get_position() returned None")

    print("\n" + "=" * 60)
    if passed:
        print(f"RESULT: PASS  (final position within {PASS_THRESHOLD_MM} mm of survey target)")
    else:
        print(f"RESULT: FAIL  (position not within {PASS_THRESHOLD_MM} mm of target, or None returned)")
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
