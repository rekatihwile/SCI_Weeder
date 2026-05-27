#!/usr/bin/env python3
"""Thin wrapper around pyagxrobots for the AgileX Scout."""
import subprocess
import time

import pyagxrobots


class Scout:
    # safe defaults for Scout 2.0 — bump if you want
    MAX_LIN = 1.5     # m/s
    MAX_ANG = 0.5236  # rad/s (~30 deg/s)

    def __init__(self, channel="can0", bitrate=500000, model="mini", auto_bringup=True):
        """
        model: "mini" → ScoutMiniBase (confirmed working on our 2.0)
               "scout2" → ScoutBase   (the "proper" Scout 2.0 class)
        """
        self.channel = channel
        self.bitrate = bitrate
        if auto_bringup:
            self._ensure_link_up()
        if not self._link_up():
            raise RuntimeError(
                f"SocketCAN link {channel!r} is not up. Run:\n"
                f"  sudo ip link set {channel} down\n"
                f"  sudo ip link set {channel} up type can bitrate {bitrate}"
            )

        cls = {
            "mini":   pyagxrobots.pysdkugv.ScoutMiniBase,
            "scout2": pyagxrobots.pysdkugv.ScoutBase,
        }[model]
        self._robot = cls()
        self._robot.EnableCAN()
        time.sleep(0.2)  # let bg threads populate state

    def _run_cmd(self, cmd):
        return subprocess.run(cmd, capture_output=True, text=True)

    def _try_bringup(self):
        down_cmd = ["ip", "link", "set", self.channel, "down"]
        up_cmd = ["ip", "link", "set", self.channel, "up", "type", "can", "bitrate", str(self.bitrate)]

        down = self._run_cmd(down_cmd)
        up = self._run_cmd(up_cmd)
        if up.returncode == 0:
            return True, None

        need_privilege = "Operation not permitted" in (up.stderr or "") or up.returncode != 0
        if not need_privilege:
            return False, up.stderr.strip() or up.stdout.strip()

        down_sudo = self._run_cmd(["sudo", "-n", *down_cmd])
        up_sudo = self._run_cmd(["sudo", "-n", *up_cmd])
        if up_sudo.returncode == 0:
            return True, None

        if up_sudo.returncode != 0 and "a password is required" in (up_sudo.stderr or "").lower():
            return False, "sudo requires a password"
        return False, up_sudo.stderr.strip() or up.stderr.strip() or up.stdout.strip()

    def _ensure_link_up(self):
        if self._link_up():
            return
        ok, err = self._try_bringup()
        if not ok:
            hint = (
                f"Could not auto bring-up {self.channel}. {err or ''}\n"
                f"Run once:\n"
                f"  sudo ip link set {self.channel} down\n"
                f"  sudo ip link set {self.channel} up type can bitrate {self.bitrate}\n"
                f"Or allow passwordless sudo for /sbin/ip to avoid prompts in teleop."
            )
            raise RuntimeError(hint)
        time.sleep(0.05)

    def _link_up(self):
        out = subprocess.run(
            ["ip", "-details", "link", "show", self.channel],
            capture_output=True, text=True,
        )
        return out.returncode == 0 and "state UP" in out.stdout

    # ---- control ----
    def drive(self, linear=0.0, angular=0.0):
        """Send one motion command. Values clamped to MAX_LIN / MAX_ANG."""
        lin = max(-self.MAX_LIN, min(self.MAX_LIN, linear))
        ang = max(-self.MAX_ANG, min(self.MAX_ANG, angular))
        self._robot.SetMotionCommand(linear_vel=lin, angular_vel=ang)

    def stop(self):
        self._robot.SetMotionCommand(linear_vel=0.0, angular_vel=0.0)

    # ---- telemetry ----
    def state(self):
        r = self._robot

        def _safe_call(name, default=None):
            fn = getattr(r, name, None)
            if fn is None:
                return default
            try:
                return fn()
            except Exception:
                return default

        left_odom = _safe_call("GetLeftWheelOdom", None)
        right_odom = _safe_call("GetRightWheelOdom", None)
        if left_odom is None:
            left_odom = _safe_call("GetLeftWheel", None)
        if right_odom is None:
            right_odom = _safe_call("GetRightWheel", None)

        return {
            "battery_v":  _safe_call("GetBatteryVoltage", 0.0),
            "error":      _safe_call("GetErrorCode", 0),
            "ctrl_mode":  _safe_call("GetControlMode", 0),
            "linear":     _safe_call("GetLinearVelocity", 0.0),
            "angular":    _safe_call("GetAngularVelocity", 0.0),
            "left_odom":  left_odom,
            "right_odom": right_odom,
        }

    # ---- context manager: auto-stop on exit ----
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.stop()
        except Exception:
            pass