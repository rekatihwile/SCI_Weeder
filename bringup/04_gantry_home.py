"""
bringup/04_gantry_home.py
--------------------------
Home gantry only. First motion script.
Requires explicit typed confirmation before any movement.

Run with:
    ./run_with_eli_venv.sh bringup/04_gantry_home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from config import GRBL_PORT
    from hardware.gantry import Gantry

    print("=" * 60)
    print("BRINGUP 04 — Gantry Home")
    print("=" * 60)
    print(f"\n  GRBL_PORT: {GRBL_PORT}")
    print("\n  WARNING: This script will home the gantry (physical motion).")

    answer = input("\nType HOME to home the gantry: ").strip()
    if answer != "HOME":
        print("  Confirmation not received. Aborting.")
        sys.exit(1)

    print("\n--- Instantiating Gantry ---")
    gantry = Gantry(GRBL_PORT)
    print("  Gantry opened.")

    pos_before = gantry.get_position()
    print(f"  Position before home: {pos_before}")

    print("\n--- Homing ---")
    gantry.home()

    pos_after = gantry.get_position()
    print(f"\n  Position after home: {pos_after}")

    gantry.serial.close()
    print("  Serial closed.")

    passed = pos_after is not None
    print("\n" + "=" * 60)
    if passed:
        print(f"RESULT: PASS  (position after home: {pos_after})")
    else:
        print("RESULT: FAIL  (get_position() returned None after homing)")
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
