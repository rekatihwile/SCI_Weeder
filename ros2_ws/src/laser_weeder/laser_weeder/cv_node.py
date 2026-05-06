"""
cv_node.py — runs AI detection (YOLO via AIDetector) on incoming camera frames.

The NMS C++ patch is applied before any ultralytics import.

Subscriptions:
  /lw/left/image_raw    sensor_msgs/Image
  /lw/right/image_raw   sensor_msgs/Image
  /lw/cv/trigger_burst  std_msgs/Int32   (value = burst frame count)

Published:
  /lw/detections/left         DetectionList  (single-frame live, ~5Hz)
  /lw/detections/right        DetectionList  (single-frame live, ~5Hz)
  /lw/detections/burst_left   DetectionList  (burst-stable, on burst trigger)
  /lw/detections/burst_right  DetectionList  (burst-stable, on burst trigger)

Parameters:
  model_path      str   Path to YOLO weights (default from config DEFAULT_MODEL)
  live_hz         float Single-frame publish rate in Hz (default 5.0)
  conf            float Detection confidence threshold (default from config AI_CONFIDENCE)
  target_class    int   YOLO class to detect; -1 = all classes (default AI_TARGET_CLASS)
"""

# ── NMS patch — MUST happen before any ultralytics / torch import ─────────────
import importlib.util as _ilu
import laser_weeder._repo_path as _rp  # noqa: F401 — adds repo root to sys.path
from pathlib import Path as _Path

_nms_spec = _ilu.spec_from_file_location(
    "_nms_patch", _rp.REPO_ROOT / "bringup" / "_nms_patch.py"
)
_nms_mod = _ilu.module_from_spec(_nms_spec)
_nms_spec.loader.exec_module(_nms_mod)

# ── now safe to import repo modules ──────────────────────────────────────────
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from message_filters import ApproximateTimeSynchronizer, Subscriber

from config import DEFAULT_MODEL, AI_CONFIDENCE, AI_TARGET_CLASS
from vision.detectors.ai_detector import AIDetector
from laser_weeder_msgs.msg import Detection, DetectionList


def _img_msg_to_array(msg: Image) -> np.ndarray:
    return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
        (msg.height, msg.width, 3)
    )


def _detections_to_msg(detections: list, stamp, side: str) -> DetectionList:
    out = DetectionList()
    out.header.stamp = stamp
    out.camera_side = side
    for d in detections:
        det = Detection()
        x1, y1, x2, y2 = d.get("box", (0, 0, 0, 0))
        det.box   = [float(x1), float(y1), float(x2), float(y2)]
        cx, cy    = d.get("point", ((x1 + x2) / 2, (y1 + y2) / 2))
        det.point = [float(cx), float(cy)]
        det.cls   = int(d.get("cls", 0))
        det.conf  = float(d.get("conf", 0.0))
        det.views = int(d.get("views", 1))
        out.detections.append(det)
    return out


class CVNode(Node):
    def __init__(self):
        super().__init__("cv_node")

        self.declare_parameter("model_path",   str(DEFAULT_MODEL))
        self.declare_parameter("live_hz",      5.0)
        self.declare_parameter("conf",         AI_CONFIDENCE)
        self.declare_parameter("target_class", AI_TARGET_CLASS)

        model_path   = self.get_parameter("model_path").value
        live_hz      = float(self.get_parameter("live_hz").value)
        self._conf   = float(self.get_parameter("conf").value)
        self._cls    = int(self.get_parameter("target_class").value)

        self._detector = AIDetector(model_path=model_path)

        # Frame buffers for live and burst detection
        self._latest_left:  np.ndarray | None = None
        self._latest_right: np.ndarray | None = None
        self._burst_frames_left:  list = []
        self._burst_frames_right: list = []
        self._burst_count = 0
        self._burst_lock  = threading.Lock()

        # Publishers
        self._pub_live_left   = self.create_publisher(DetectionList, "/lw/detections/left",         10)
        self._pub_live_right  = self.create_publisher(DetectionList, "/lw/detections/right",        10)
        self._pub_burst_left  = self.create_publisher(DetectionList, "/lw/detections/burst_left",   10)
        self._pub_burst_right = self.create_publisher(DetectionList, "/lw/detections/burst_right",  10)

        # Synchronised subscriber for live detection
        self._sub_left  = Subscriber(self, Image, "/lw/left/image_raw")
        self._sub_right = Subscriber(self, Image, "/lw/right/image_raw")
        self._sync = ApproximateTimeSynchronizer(
            [self._sub_left, self._sub_right], queue_size=10, slop=0.05
        )
        self._sync.registerCallback(self._on_pair)

        # Burst trigger
        self._sub_burst = self.create_subscription(
            Int32, "/lw/cv/trigger_burst", self._on_burst_trigger, 10
        )

        # Live detection timer
        self._live_timer = self.create_timer(1.0 / max(live_hz, 0.1), self._publish_live)

        self.get_logger().info(
            f"CVNode ready: model={model_path} conf={self._conf} class={self._cls}"
        )

    # ── frame ingestion ───────────────────────────────────────────────────────

    def _on_pair(self, left_msg: Image, right_msg: Image):
        left  = _img_msg_to_array(left_msg)
        right = _img_msg_to_array(right_msg)
        with self._burst_lock:
            self._latest_left  = left
            self._latest_right = right
            if self._burst_count > 0:
                self._burst_frames_left.append(left.copy())
                self._burst_frames_right.append(right.copy())
                self._burst_count -= 1
                if self._burst_count == 0:
                    # Detach for processing in background thread
                    frames_l = self._burst_frames_left
                    frames_r = self._burst_frames_right
                    self._burst_frames_left  = []
                    self._burst_frames_right = []
                    threading.Thread(
                        target=self._run_burst, args=(frames_l, frames_r), daemon=True
                    ).start()

    # ── live detection ────────────────────────────────────────────────────────

    def _publish_live(self):
        with self._burst_lock:
            left  = self._latest_left
            right = self._latest_right
        if left is None or right is None:
            return

        stamp = self.get_clock().now().to_msg()
        classes_arg = [self._cls] if self._cls >= 0 else None

        dets_l = self._detector.cv_left.detect(left,   classes=classes_arg, conf=self._conf)
        dets_r = self._detector.cv_right.detect(right, classes=classes_arg, conf=self._conf)

        self._pub_live_left.publish(_detections_to_msg(dets_l, stamp, "left"))
        self._pub_live_right.publish(_detections_to_msg(dets_r, stamp, "right"))

    # ── burst detection ───────────────────────────────────────────────────────

    def _on_burst_trigger(self, msg: Int32):
        count = max(1, msg.data)
        with self._burst_lock:
            if self._burst_count > 0:
                self.get_logger().info("Burst already in progress — ignoring trigger")
                return
            self._burst_frames_left  = []
            self._burst_frames_right = []
            self._burst_count = count
        self.get_logger().info(f"Burst triggered: collecting {count} frames")

    def _run_burst(self, frames_l: list, frames_r: list):
        t0 = time.perf_counter()
        classes_arg = [self._cls] if self._cls >= 0 else None

        stable_l = self._detector.cv_left.return_burst_stable(
            frames_l,
            min_stable_views=max(1, len(frames_l) // 2),
            group_radius_px=8.0,
            classes_override=classes_arg,
        )
        stable_r = self._detector.cv_right.return_burst_stable(
            frames_r,
            min_stable_views=max(1, len(frames_r) // 2),
            group_radius_px=8.0,
            classes_override=classes_arg,
        )

        stamp = self.get_clock().now().to_msg()
        self._pub_burst_left.publish( _detections_to_msg(stable_l, stamp, "left"))
        self._pub_burst_right.publish(_detections_to_msg(stable_r, stamp, "right"))

        dt = time.perf_counter() - t0
        self.get_logger().info(
            f"Burst complete in {dt:.2f}s: left={len(stable_l)} right={len(stable_r)} "
            f"stable detections"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
