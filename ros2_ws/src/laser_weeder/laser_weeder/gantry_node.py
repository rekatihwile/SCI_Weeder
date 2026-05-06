"""
gantry_node.py — wraps the Gantry (or MockGantry) hardware class.

Published:
  /lw/gantry/status   GantryStatus  (10 Hz)

Subscriptions:
  /lw/gantry/command  GantryCommand

Command dispatch:
  "move_abs"  — blocking absolute move (queued, executed in background thread)
  "jog"       — non-blocking GRBL jog (latest-wins, replaces pending jog)
  "home"      — blocking home sequence (queued)
  "fire"      — laser pulse (queued)
  "stop"      — immediate cancel-jog + stop
  "unlock"    — GRBL alarm unlock ($X)

Parameters:
  mock        bool   Use MockGantry (default False; overridden by config MOCK_GANTRY)
  port        str    Serial port (default from config GRBL_PORT)
  status_hz   float  Status publish rate (default 10.0)
"""

# ── repo path bootstrap ───────────────────────────────────────────────────────
import laser_weeder._repo_path  # noqa: F401

import queue
import threading
import time

import rclpy
from rclpy.node import Node

from config import (
    MOCK_GANTRY,
    GRBL_PORT,
    LASER_FIRE_POWER,
    LASER_FIRE_DURATION_SEC,
    FINE_ALIGN_FEED,
)
from hardware.gantry      import Gantry
from hardware.mock_gantry import MockGantry

from laser_weeder_msgs.msg import GantryCommand, GantryStatus


def _parse_idle(status_raw: str) -> bool:
    return "<Idle" in status_raw or "Idle" in status_raw


def _parse_alarmed(status_raw: str) -> bool:
    return "Alarm" in status_raw.lower()


class GantryNode(Node):
    def __init__(self):
        super().__init__("gantry_node")

        self.declare_parameter("mock",       MOCK_GANTRY)
        self.declare_parameter("port",       GRBL_PORT or "/dev/ttyUSB0")
        self.declare_parameter("status_hz",  10.0)

        use_mock  = bool(self.get_parameter("mock").value)
        port      = self.get_parameter("port").value
        status_hz = float(self.get_parameter("status_hz").value)

        if use_mock:
            self._gantry = MockGantry()
            self.get_logger().info("GantryNode: using MockGantry")
        else:
            self._gantry = Gantry(port=port)
            self.get_logger().info(f"GantryNode: connected to {port}")

        self._homed  = False

        # Command queue for blocking ops; separate slot for latest jog
        self._cmd_q     = queue.Queue()
        self._latest_jog: GantryCommand | None = None
        self._jog_lock  = threading.Lock()

        # Background command execution thread
        self._cmd_thread = threading.Thread(
            target=self._command_loop, daemon=True, name="gantry_cmd"
        )
        self._cmd_thread.start()

        # Publishers / subscribers
        self._pub_status = self.create_publisher(GantryStatus, "/lw/gantry/status", 10)
        self._sub_cmd    = self.create_subscription(
            GantryCommand, "/lw/gantry/command", self._on_command, 10
        )

        self._status_timer = self.create_timer(1.0 / max(status_hz, 0.1), self._publish_status)

    # ── status ────────────────────────────────────────────────────────────────

    def _publish_status(self):
        try:
            raw = self._gantry.get_status_line()
        except Exception as exc:
            self.get_logger().warning(f"get_status_line failed: {exc}", throttle_duration_sec=5.0)
            return

        try:
            pos = self._gantry.get_position() or {}
        except Exception:
            pos = {}

        est_x, est_y = self._gantry.get_estimated_xy()

        msg = GantryStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.x_mm    = float(pos.get("x", est_x))
        msg.y_mm    = float(pos.get("y", est_y))
        msg.idle    = _parse_idle(raw)
        msg.homed   = self._homed
        msg.alarmed = _parse_alarmed(raw)
        msg.raw     = raw
        self._pub_status.publish(msg)

    # ── command routing ───────────────────────────────────────────────────────

    def _on_command(self, msg: GantryCommand):
        cmd = msg.cmd.strip().lower()

        if cmd == "stop":
            # Immediate — cancel jog and stop; clear any queued ops
            with self._jog_lock:
                self._latest_jog = None
            while not self._cmd_q.empty():
                try:
                    self._cmd_q.get_nowait()
                except queue.Empty:
                    break
            try:
                self._gantry.stop()
            except Exception as exc:
                self.get_logger().error(f"stop failed: {exc}")

        elif cmd == "jog":
            # Latest-wins: replace pending jog
            with self._jog_lock:
                self._latest_jog = msg

        else:
            # All other commands are queued for sequential execution
            self._cmd_q.put(msg)

    # ── background command thread ─────────────────────────────────────────────

    def _command_loop(self):
        while True:
            # Drain queued blocking commands first
            try:
                cmd = self._cmd_q.get(timeout=0.01)
                self._execute_blocking(cmd)
                continue
            except queue.Empty:
                pass

            # Apply latest jog if any
            with self._jog_lock:
                jog = self._latest_jog
                self._latest_jog = None

            if jog is not None:
                try:
                    feed = int(jog.feed_rate) if jog.feed_rate > 0 else FINE_ALIGN_FEED
                    self._gantry.jog(jog.x_mm, jog.y_mm, feed)
                except Exception as exc:
                    self.get_logger().error(f"jog failed: {exc}", throttle_duration_sec=2.0)
            else:
                time.sleep(0.005)  # 5 ms idle wait

    def _execute_blocking(self, msg: GantryCommand):
        cmd = msg.cmd.strip().lower()
        try:
            if cmd == "move_abs":
                feed = int(msg.feed_rate) if msg.feed_rate > 0 else 12000
                self.get_logger().info(
                    f"move_abs ({msg.x_mm:.1f}, {msg.y_mm:.1f}) @ {feed} mm/min"
                )
                self._gantry.move_absolute(msg.x_mm, msg.y_mm, feed=feed)

            elif cmd == "home":
                self.get_logger().info("Homing gantry...")
                self._gantry.home()
                self._homed = True
                self.get_logger().info("Homing complete")

            elif cmd == "fire":
                duration = msg.duration_sec if msg.duration_sec > 0 else LASER_FIRE_DURATION_SEC
                self.get_logger().info(f"Firing laser {duration:.3f}s @ power {LASER_FIRE_POWER}")
                self._gantry.fire_pulse(LASER_FIRE_POWER, duration)

            elif cmd == "unlock":
                self._gantry.send_raw("$X")
                self.get_logger().info("Alarm unlocked ($X sent)")

            else:
                self.get_logger().warning(f"Unknown gantry command: '{cmd}'")

        except Exception as exc:
            self.get_logger().error(f"Command '{cmd}' failed: {exc}")

    def destroy_node(self):
        try:
            self._gantry.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GantryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
