#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time

import pyagxrobots


def _link_state(channel):
    probe = subprocess.run(
        ["ip", "-details", "link", "show", channel],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return False, f"Interface {channel!r} not found"
    details = probe.stdout.strip()
    return "state UP" in details, details


def _print_can_fix_hint(channel, bitrate, *, reason=None):
    if reason:
        print(f"\n[CAN] {reason}")
    print("Run:")
    print(f"  sudo ip link set {channel} down")
    print(f"  sudo ip link set {channel} up type can bitrate {bitrate}")
    print("  ip -details link show {0}".format(channel))


def main():
    parser = argparse.ArgumentParser(description="Scout CAN RX/TX sanity test")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=500000)
    args = parser.parse_args()

    is_up, details = _link_state(args.channel)
    if not is_up:
        _print_can_fix_hint(args.channel, args.bitrate, reason="SocketCAN link is not up.")
        print(f"\n[CAN] Current status: {details}")
        raise SystemExit(2)

    try:
        robot = pyagxrobots.pysdkugv.ScoutMiniBase()
        robot.EnableCAN()
    except Exception as exc:
        is_up_now, details_now = _link_state(args.channel)
        print(f"\n[Scout] Failed to initialize robot CAN session: {exc}")
        if not is_up_now:
            _print_can_fix_hint(args.channel, args.bitrate, reason="SocketCAN link dropped during init.")
        else:
            print("\n[CAN] Link is UP, so this is likely robot-side comms (power/CAN wiring/termination/ID).")
            print("[CAN] Verify robot is powered and connected to the same CAN bus @ 500000 bps.")
        print(f"\n[CAN] Current status: {details_now}")
        raise SystemExit(3)

    time.sleep(0.2)

    print(f"battery:    {robot.GetBatteryVoltage():.2f} V")
    print(f"error code: {robot.GetErrorCode()}")
    print(f"ctrl mode:  {robot.GetControlMode()}")

    print("\nsending 0.1 m/s forward for ~1.5s — WHEELS OFF GROUND PLEASE\n")
    for _ in range(5):
        robot.SetMotionCommand(linear_vel=0.1, angular_vel=0.0)
        lin = robot.GetLinearVelocity()
        ang = robot.GetAngularVelocity()
        lw = robot.GetLeftWheelOdom()
        rw = robot.GetRightWheelOdom()
        print(f"  lin={lin:+.3f} m/s  ang={ang:+.3f} rad/s  L={lw}  R={rw}")
        time.sleep(0.3)

    robot.SetMotionCommand(linear_vel=0.0, angular_vel=0.0)
    print("\nstopped.")


if __name__ == "__main__":
    main()