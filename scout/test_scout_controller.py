#!/usr/bin/env python3
"""scout/test_scout_controller.py — Quick standalone test for ScoutController.

Run from repo root:
    ./run_with_eli_venv.sh scout/test_scout_controller.py
    ./run_with_eli_venv.sh scout/test_scout_controller.py --dry-run
    ./run_with_eli_venv.sh scout/test_scout_controller.py --move 0.10
"""

import argparse
import sys
from pathlib import Path

# Allow repo-root imports when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scout.scout_controller import ScoutController


def main():
    parser = argparse.ArgumentParser(description="ScoutController quick test")
    parser.add_argument(
        "--interface", default="can0", metavar="IFACE",
        help="CAN interface (default: can0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate without sending motion",
    )
    parser.add_argument(
        "--move", type=float, default=None, metavar="DIST_M",
        help="Also command a forward move of this distance (m)",
    )
    parser.add_argument(
        "--speed", type=float, default=0.10, metavar="SPEED_MPS",
        help="Speed for --move test in m/s (default: 0.10)",
    )
    args = parser.parse_args()

    sc = ScoutController(interface=args.interface, dry_run=args.dry_run)

    try:
        print("=== Scout connection check ===")
        r = sc.check_connection()
        print(f"  ok:        {r['ok']}")
        print(f"  interface: {r['interface']}")
        print(f"  rx_ok:     {r['rx_ok']}")
        print(f"  tx_ok:     {r['tx_ok']}")
        print(f"  message:   {r['message']}")
        if "battery_v" in r:
            print(f"  battery:   {r['battery_v']:.2f} V")

        if not r["ok"] and not args.dry_run:
            print("\nCannot proceed — Scout not connected.")
            sys.exit(1)

        print("\n=== Stop (safety) ===")
        sc.stop()
        print("  stop() sent.")

        if args.move is not None:
            print(f"\n=== Move forward {args.move:.3f} m @ {args.speed:.3f} m/s ===")
            mr = sc.move_forward(distance_m=args.move, speed_mps=args.speed)
            print(f"  ok:         {mr['ok']}")
            print(f"  distance_m: {mr['distance_m']}")
            print(f"  speed_mps:  {mr['speed_mps']}")
            print(f"  duration_s: {mr['duration_s']}")
            print(f"  message:    {mr['message']}")
        else:
            print("\n(Pass --move 0.10 to also test a forward move.)")

    finally:
        sc.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
