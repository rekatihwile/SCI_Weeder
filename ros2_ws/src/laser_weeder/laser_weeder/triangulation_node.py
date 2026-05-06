"""
triangulation_node.py — stereo match + triangulate burst detections into gantry
workspace coordinates, then plan a visitation tour.

Subscriptions:
  /lw/detections/burst_left   DetectionList  (burst-stable left)
  /lw/detections/burst_right  DetectionList  (burst-stable right)
  /lw/gantry/status           GantryStatus   (current gantry position)

Published:
  /lw/targets   TargetList   (tour-ordered stereo targets in workspace mm)

The node holds the latest burst from each side and triangulates when it has a
fresh pair with matching headers (within 1 s of each other).

Parameters:
  survey_pos_x  float  Reference gantry X when survey was taken (default 0.0)
  survey_pos_y  float  Reference gantry Y when survey was taken (default 0.0)
  output_json   str    Optional path to write planned targets as JSON ("" = skip)
"""

# ── repo path bootstrap ───────────────────────────────────────────────────────
import laser_weeder._repo_path  # noqa: F401

import json
import time
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber

from config import SURVEY_POS_X, SURVEY_POS_Y
from control.coarse_move import TriangulationCoarseMover
from pipeline.steps.match_plan import run_match_and_plan

from laser_weeder_msgs.msg import DetectionList, StereoTarget, TargetList, GantryStatus


def _det_msg_to_dicts(msg: DetectionList) -> list:
    """Convert DetectionList ROS msg to list of detection dicts the pipeline expects."""
    out = []
    for d in msg.detections:
        out.append({
            "box":   (d.box[0], d.box[1], d.box[2], d.box[3]),
            "point": (d.point[0], d.point[1]),
            "cls":   int(d.cls),
            "conf":  float(d.conf),
            "views": int(d.views),
        })
    return out


class TriangulationNode(Node):
    def __init__(self):
        super().__init__("triangulation_node")

        self.declare_parameter("survey_pos_x", SURVEY_POS_X)
        self.declare_parameter("survey_pos_y", SURVEY_POS_Y)
        self.declare_parameter("output_json",  "")

        self._ref_x = float(self.get_parameter("survey_pos_x").value)
        self._ref_y = float(self.get_parameter("survey_pos_y").value)
        self._output_json = self.get_parameter("output_json").value or ""

        # Gantry current position — updated from /lw/gantry/status
        self._gantry_x = self._ref_x
        self._gantry_y = self._ref_y
        self._status_lock = threading.Lock()

        # Mover (loads calibration from config, no constructor args)
        self._mover = TriangulationCoarseMover()

        # Publisher
        self._pub_targets = self.create_publisher(TargetList, "/lw/targets", 10)

        # Synchronised burst detection subscribers
        self._sub_left  = Subscriber(self, DetectionList, "/lw/detections/burst_left")
        self._sub_right = Subscriber(self, DetectionList, "/lw/detections/burst_right")
        self._sync = ApproximateTimeSynchronizer(
            [self._sub_left, self._sub_right], queue_size=5, slop=1.0
        )
        self._sync.registerCallback(self._on_burst_pair)

        # Gantry status subscriber (use position at time of survey)
        self._sub_status = self.create_subscription(
            GantryStatus, "/lw/gantry/status", self._on_gantry_status, 10
        )

        self.get_logger().info(
            f"TriangulationNode ready: ref=({self._ref_x:.1f}, {self._ref_y:.1f}) mm"
        )

    def _on_gantry_status(self, msg: GantryStatus):
        with self._status_lock:
            self._gantry_x = msg.x_mm
            self._gantry_y = msg.y_mm

    def _on_burst_pair(self, left_msg: DetectionList, right_msg: DetectionList):
        dets_l = _det_msg_to_dicts(left_msg)
        dets_r = _det_msg_to_dicts(right_msg)

        if not dets_l or not dets_r:
            self.get_logger().info(
                f"Burst pair received but empty: left={len(dets_l)} right={len(dets_r)}"
            )
            return

        with self._status_lock:
            ref_x = self._gantry_x
            ref_y = self._gantry_y

        self.get_logger().info(
            f"Triangulating: left={len(dets_l)} right={len(dets_r)} "
            f"ref=({ref_x:.1f},{ref_y:.1f})"
        )

        threading.Thread(
            target=self._triangulate,
            args=(dets_l, dets_r, ref_x, ref_y),
            daemon=True,
        ).start()

    def _triangulate(self, dets_l: list, dets_r: list, ref_x: float, ref_y: float):
        t0 = time.perf_counter()
        try:
            output_path = Path(self._output_json) if self._output_json else None
            matched, absolute, planned = run_match_and_plan(
                dets_l, dets_r, self._mover,
                start_xy=(ref_x, ref_y),
                output_path=output_path,
            )
        except Exception as exc:
            self.get_logger().error(f"Triangulation failed: {exc}")
            return

        dt = time.perf_counter() - t0
        self.get_logger().info(
            f"Triangulation done in {dt:.2f}s: {len(planned)} planned targets"
        )

        self._publish_targets(planned)

    def _publish_targets(self, planned: list):
        msg = TargetList()
        msg.header.stamp = self.get_clock().now().to_msg()

        for p in planned:
            t = StereoTarget()
            src = p.get("source_target", {})

            lx, ly = src.get("left_px",  (0.0, 0.0))
            rx, ry = src.get("right_px", (0.0, 0.0))
            t.left_px  = [float(lx), float(ly)]
            t.right_px = [float(rx), float(ry)]

            xy = p.get("target_xy_mm", (0.0, 0.0))
            t.x_mm = float(xy[0])
            t.y_mm = float(xy[1])
            t.score = float(src.get("score", 0.0))
            t.cls   = int(src.get("left_cls",  src.get("cls", 0)) or 0)
            t.conf  = float(src.get("left_conf", src.get("conf", 0.0)) or 0.0)

            msg.targets.append(t)

        self._pub_targets.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TriangulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
