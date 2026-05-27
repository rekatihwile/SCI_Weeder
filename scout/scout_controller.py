#!/usr/bin/env python3
"""scout/scout_controller.py — ScoutController for automated Scout movement.

Wraps the existing Scout class (scout/scout.py) with a clean API for
automated forward motion between laser-weeder trials.

Usage as a module:
    from scout.scout_controller import ScoutController

    sc = ScoutController(interface="can0", dry_run=False)
    result = sc.check_connection()
    result = sc.move_forward(distance_m=0.5, speed_mps=0.15)
    sc.close()

Standalone CLI:
    python -m scout.scout_controller --check
    python -m scout.scout_controller --move-forward 0.25 --speed 0.10 --dry-run
    python -m scout.scout_controller --move-backwards 0.25 --speed 0.10 --dry-run
    python -m scout.scout_controller --move-forward 0.25 --speed 0.10
"""

import subprocess
import sys
import time


def _link_up(interface: str) -> bool:
    """Return True if the named SocketCAN interface is UP."""
    out = subprocess.run(
        ["ip", "-details", "link", "show", interface],
        capture_output=True,
        text=True,
    )
    return out.returncode == 0 and "state UP" in out.stdout


class ScoutController:
    """Thin controller for automated Scout movement between laser-weeder trials.

    Reuses the Scout class from scout/scout.py for all CAN communication.
    """

    def __init__(self, interface: str = "can0", dry_run: bool = False):
        self.interface = interface
        self.dry_run = dry_run
        self._scout = None  # lazily created on first use

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_scout(self):
        """Open Scout CAN connection if not already open."""
        if self._scout is None:
            # Import here to keep top-level import lightweight.
            # scout/scout.py is at repo_root/scout/scout.py.
            from scout.scout import Scout
            self._scout = Scout(channel=self.interface)
        return self._scout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_connection(self, timeout_s: float = 2.0) -> dict:
        """Check CAN link and Scout responsiveness.

        Returns:
            {
                "ok": bool,
                "interface": str,
                "rx_ok": bool,
                "tx_ok": bool,
                "message": str,
                # when ok: also "battery_v", "error_code", "ctrl_mode"
            }
        """
        result = {
            "ok": False,
            "interface": self.interface,
            "rx_ok": False,
            "tx_ok": False,
            "message": "",
        }

        if self.dry_run:
            result.update({
                "ok": True,
                "rx_ok": True,
                "tx_ok": True,
                "message": "[dry-run] Connection check skipped.",
            })
            return result

        if not _link_up(self.interface):
            result["message"] = (
                f"SocketCAN interface {self.interface!r} is not UP. "
                f"Run: sudo ip link set {self.interface} up type can bitrate 500000"
            )
            return result

        # CAN link is up — RX is plausible.
        result["rx_ok"] = True

        try:
            scout = self._ensure_scout()
            st = scout.state()
            batt = st.get("battery_v", 0.0)
            err = st.get("error", -1)
            mode = st.get("ctrl_mode", -1)
            result.update({
                "ok": True,
                "tx_ok": True,
                "battery_v": batt,
                "error_code": err,
                "ctrl_mode": mode,
                "message": (
                    f"Scout connected. battery={batt:.2f}V "
                    f"error={err} mode={mode}"
                ),
            })
        except Exception as exc:
            result["message"] = f"Scout CAN init failed: {exc}"

        return result

    def move_forward(
        self,
        distance_m: float,
        speed_mps: float,
        timeout_s: float = None,
    ) -> dict:
        """Command forward motion for distance_m / speed_mps seconds, then stop.

        Uses time-based distance estimation (no wheel odometry required).
        Always calls stop() in a finally block.

        Returns:
            {
                "ok": bool,
                "distance_m": float,
                "speed_mps": float,
                "duration_s": float,
                "message": str,
            }
        """
        result = {
            "ok": False,
            "distance_m": distance_m,
            "speed_mps": speed_mps,
            "duration_s": 0.0,
            "message": "",
        }

        if speed_mps <= 0:
            result["message"] = "speed_mps must be > 0"
            return result

        linear_cmd = speed_mps if distance_m >= 0 else -speed_mps
        duration_s = abs(distance_m) / speed_mps
        if timeout_s is not None and timeout_s > 0:
            duration_s = min(duration_s, timeout_s)

        result["duration_s"] = round(duration_s, 3)

        direction = "forward" if linear_cmd >= 0 else "backward"
        distance_mag = abs(distance_m)

        if self.dry_run:
            print(
                f"[Scout dry-run] move_{direction}: {distance_mag:.3f} m "
                f"@ {speed_mps:.3f} m/s → {duration_s:.2f} s (no motion sent)"
            )
            result.update({
                "ok": True,
                "message": (
                    f"[dry-run] Would move {direction} {distance_mag:.3f} m "
                    f"@ {speed_mps:.3f} m/s."
                ),
            })
            return result

        try:
            scout = self._ensure_scout()
            print(
                f"[Scout] Moving {direction} {distance_mag:.3f} m "
                f"@ {speed_mps:.3f} m/s for {duration_s:.2f} s..."
            )

            command_hz = 20.0
            dt = 1.0 / command_hz

            t_start = time.perf_counter()
            t_end = t_start + duration_s

            while time.perf_counter() < t_end:
                scout.drive(linear=linear_cmd, angular=0.0)
                time.sleep(dt)

            elapsed = round(time.perf_counter() - t_start, 3)
            result.update({
                "ok": True,
                "duration_s": elapsed,
                "message": (
                    f"Moved {direction} {distance_mag:.3f} m in {elapsed:.2f} s."
                ),
            })

        except Exception as exc:
            result["message"] = f"Scout move_forward failed: {exc}"
        finally:
            self.stop()

        return result

    def move_backward(
        self,
        distance_m: float,
        speed_mps: float,
        timeout_s: float = None,
    ) -> dict:
        """Command backward motion for distance_m at speed_mps, then stop.

        Equivalent to move_forward with a negated distance.
        distance_m should be positive (the magnitude of reverse travel).
        """
        result = self.move_forward(-abs(distance_m), speed_mps, timeout_s)
        result["distance_m"] = abs(distance_m)
        return result

    def stop(self):
        """Send zero-velocity stop command."""
        if self.dry_run:
            print("[Scout dry-run] stop()")
            return
        if self._scout is not None:
            try:
                self._scout.stop()
            except Exception:
                pass

    def close(self):
        """Stop Scout and release resources."""
        self.stop()
        self._scout = None


# =============================================================================
# Standalone CLI
# =============================================================================

def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ScoutController standalone test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scout.scout_controller --check
  python -m scout.scout_controller --move-forward 0.25 --speed 0.10 --dry-run
  python -m scout.scout_controller --move-backwards 0.25 --speed 0.10 --dry-run
  python -m scout.scout_controller --move-forward 0.25 --speed 0.10
""",
    )
    parser.add_argument(
        "--interface", default="can0", metavar="IFACE",
        help="SocketCAN interface name (default: can0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print intended commands without sending motion",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check Scout CAN connection",
    )
    parser.add_argument(
        "--move-forward", type=float, default=None, metavar="DIST_M",
        help="Move forward by this distance in meters",
    )
    parser.add_argument(
        "--move-backward", "--move-backwards",
        dest="move_backward",
        type=float,
        default=None,
        metavar="DIST_M",
        help="Move backward by this distance in meters",
    )
    parser.add_argument(
        "--speed", type=float, default=0.15, metavar="SPEED_MPS",
        help="Linear speed in m/s (default: 0.15)",
    )
    parser.add_argument(
        "--timeout", type=float, default=None, metavar="SEC",
        help="Move timeout in seconds",
    )
    args = parser.parse_args()

    if args.move_forward is not None and args.move_backward is not None:
        parser.error("--move-forward and --move-backward cannot be used together")

    if not args.check and args.move_forward is None and args.move_backward is None:
        parser.print_help()
        sys.exit(0)

    sc = ScoutController(interface=args.interface, dry_run=args.dry_run)
    exit_code = 0
    try:
        if args.check:
            r = sc.check_connection()
            print(f"Connection: {'OK' if r['ok'] else 'FAIL'}")
            for k, v in r.items():
                print(f"  {k}: {v}")
            if not r["ok"]:
                exit_code = 1

        if args.move_forward is not None:
            r = sc.move_forward(
                distance_m=args.move_forward,
                speed_mps=args.speed,
                timeout_s=args.timeout,
            )
            print(f"Move: {'OK' if r['ok'] else 'FAIL'}")
            for k, v in r.items():
                print(f"  {k}: {v}")
            if not r["ok"]:
                exit_code = 1

        if args.move_backward is not None:
            r = sc.move_backward(
                distance_m=args.move_backward,
                speed_mps=args.speed,
                timeout_s=args.timeout,
            )
            print(f"Move: {'OK' if r['ok'] else 'FAIL'}")
            for k, v in r.items():
                print(f"  {k}: {v}")
            if not r["ok"]:
                exit_code = 1
    finally:
        sc.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    _main()
