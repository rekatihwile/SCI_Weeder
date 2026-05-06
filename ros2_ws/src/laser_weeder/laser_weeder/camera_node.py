"""
camera_node.py — publishes raw frames from left and right USB cameras.

Topics published:
  /lw/left/image_raw   sensor_msgs/Image   (bgr8)
  /lw/right/image_raw  sensor_msgs/Image   (bgr8)

Parameters:
  left_index   int   OpenCV capture index for left  camera  (default 0)
  right_index  int   OpenCV capture index for right camera  (default 2)
  fps          int   Target publish rate (default 30)
  width        int   Capture width  (default from config FRAME_WIDTH)
  height       int   Capture height (default from config FRAME_HEIGHT)
"""

# ── repo path bootstrap (must come before any repo import) ────────────────────
import laser_weeder._repo_path  # noqa: F401  — adds repo root to sys.path

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from config import FRAME_WIDTH, FRAME_HEIGHT


def _build_image_msg(frame: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = frame.shape[0]
    msg.width = frame.shape[1]
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = frame.tobytes()
    return msg


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("left_index",  0)
        self.declare_parameter("right_index", 2)
        self.declare_parameter("fps",         30)
        self.declare_parameter("width",       FRAME_WIDTH)
        self.declare_parameter("height",      FRAME_HEIGHT)

        left_idx  = self.get_parameter("left_index").value
        right_idx = self.get_parameter("right_index").value
        fps       = self.get_parameter("fps").value
        width     = self.get_parameter("width").value
        height    = self.get_parameter("height").value

        self._pub_left  = self.create_publisher(Image, "/lw/left/image_raw",  10)
        self._pub_right = self.create_publisher(Image, "/lw/right/image_raw", 10)

        self._cap_left  = self._open_cap(left_idx,  width, height)
        self._cap_right = self._open_cap(right_idx, width, height)

        period = 1.0 / max(1, fps)
        self._timer = self.create_timer(period, self._publish_pair)
        self.get_logger().info(
            f"CameraNode ready: left={left_idx} right={right_idx} "
            f"{width}x{height} @ {fps}fps"
        )

    def _open_cap(self, index: int, width: int, height: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            # Fallback: try without backend hint
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _publish_pair(self):
        stamp = self.get_clock().now().to_msg()

        ret_l, frame_l = self._cap_left.read()
        ret_r, frame_r = self._cap_right.read()

        if ret_l and frame_l is not None:
            self._pub_left.publish(_build_image_msg(frame_l, stamp, "left_camera"))
        else:
            self.get_logger().warning("Left camera read failed", throttle_duration_sec=5.0)

        if ret_r and frame_r is not None:
            self._pub_right.publish(_build_image_msg(frame_r, stamp, "right_camera"))
        else:
            self.get_logger().warning("Right camera read failed", throttle_duration_sec=5.0)

    def destroy_node(self):
        self._cap_left.release()
        self._cap_right.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
