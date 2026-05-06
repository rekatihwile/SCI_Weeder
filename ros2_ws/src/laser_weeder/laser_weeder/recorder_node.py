"""
recorder_node.py — saves stereo frame pairs to disk with a manifest.

Subscriptions:
  /lw/left/image_raw   sensor_msgs/Image
  /lw/right/image_raw  sensor_msgs/Image
  /lw/recorder/control std_msgs/String
    "start"            begin recording (auto-generates trial ID)
    "start:<trial_id>" begin recording with given trial ID
    "stop"             stop recording and flush manifest

Recordings land in:  <TRIAL_RECORDINGS_DIR>/trial_<id>_<timestamp>/
Manifest:            trial_recordings/manifest_<timestamp>.json
"""

# ── repo path bootstrap ───────────────────────────────────────────────────────
import laser_weeder._repo_path  # noqa: F401

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from message_filters import ApproximateTimeSynchronizer, Subscriber

from config import TRIAL_RECORDINGS_DIR


def _img_msg_to_array(msg: Image) -> np.ndarray:
    return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
        (msg.height, msg.width, 3)
    )


class RecorderNode(Node):
    def __init__(self):
        super().__init__("recorder_node")

        self.declare_parameter("jpeg_quality", 90)
        self._quality = self.get_parameter("jpeg_quality").value

        self._lock = threading.Lock()
        self._recording = False
        self._trial_dir: Path | None = None
        self._manifest: list = []
        self._frame_idx = 0

        # Synchronised stereo pair subscriber
        self._sub_left  = Subscriber(self, Image, "/lw/left/image_raw")
        self._sub_right = Subscriber(self, Image, "/lw/right/image_raw")
        self._sync = ApproximateTimeSynchronizer(
            [self._sub_left, self._sub_right],
            queue_size=10,
            slop=0.05,
        )
        self._sync.registerCallback(self._on_pair)

        self._sub_ctrl = self.create_subscription(
            String, "/lw/recorder/control", self._on_control, 10
        )

        self.get_logger().info("RecorderNode ready — send 'start' to /lw/recorder/control")

    # ── control ───────────────────────────────────────────────────────────────

    def _on_control(self, msg: String):
        cmd = msg.data.strip()

        if cmd.startswith("start"):
            parts = cmd.split(":", 1)
            trial_id = parts[1] if len(parts) == 2 else None
            self._start_recording(trial_id)

        elif cmd == "stop":
            self._stop_recording()

        else:
            self.get_logger().warning(f"Unknown recorder command: '{cmd}'")

    def _start_recording(self, trial_id: str | None = None):
        with self._lock:
            if self._recording:
                self.get_logger().info("Already recording — ignoring start")
                return

            ts = time.strftime("%Y%m%d_%H%M%S")
            tid = trial_id or ts
            self._trial_dir = Path(TRIAL_RECORDINGS_DIR) / f"trial_{tid}_{ts}"
            self._trial_dir.mkdir(parents=True, exist_ok=True)
            self._manifest = []
            self._frame_idx = 0
            self._recording = True

        self.get_logger().info(f"Recording started → {self._trial_dir}")

    def _stop_recording(self):
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            trial_dir = self._trial_dir
            manifest  = list(self._manifest)

        # Write manifest outside the lock
        if trial_dir is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            manifest_path = Path(TRIAL_RECORDINGS_DIR) / f"manifest_{ts}.json"
            try:
                manifest_path.write_text(json.dumps(manifest, indent=2))
                self.get_logger().info(
                    f"Recording stopped — {len(manifest)} frames, manifest → {manifest_path}"
                )
            except Exception as exc:
                self.get_logger().error(f"Failed to write manifest: {exc}")

    # ── frame callback ────────────────────────────────────────────────────────

    def _on_pair(self, left_msg: Image, right_msg: Image):
        with self._lock:
            if not self._recording or self._trial_dir is None:
                return
            idx   = self._frame_idx
            tdir  = self._trial_dir
            self._frame_idx += 1

        left_arr  = _img_msg_to_array(left_msg)
        right_arr = _img_msg_to_array(right_msg)

        ts_ns = left_msg.header.stamp.sec * 10**9 + left_msg.header.stamp.nanosec
        left_name  = f"frame_{idx:06d}_left.jpg"
        right_name = f"frame_{idx:06d}_right.jpg"

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        cv2.imwrite(str(tdir / left_name),  left_arr,  encode_params)
        cv2.imwrite(str(tdir / right_name), right_arr, encode_params)

        entry = {
            "frame": idx,
            "timestamp_ns": ts_ns,
            "left":  str(tdir / left_name),
            "right": str(tdir / right_name),
        }
        with self._lock:
            self._manifest.append(entry)


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_recording()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
