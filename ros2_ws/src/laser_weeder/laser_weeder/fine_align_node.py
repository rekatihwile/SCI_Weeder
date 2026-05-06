"""
fine_align_node.py — Lucas-Kanade optical-flow PD servo loop for laser alignment.

Subscriptions:
  /lw/left/image_raw     sensor_msgs/Image    (synchronized stereo frames)
  /lw/right/image_raw    sensor_msgs/Image
  /lw/fine_align/goal    FineAlignGoal        (start or cancel a session)

Published:
  /lw/fine_align/pixel_error  PixelError      (every frame while active)
  /lw/gantry/command          GantryCommand   (jog commands to servo the gantry)

Algorithm (mirrors control/fine_align_motion.py):
  1. FineAlignGoal gives initial left+right detection pixels (full-frame).
  2. Both points are converted to crop-space (centre crop of the frame).
  3. LK optical flow tracks the crop-space point in subsequent frames.
  4. Error: dx = (xl + xr) - 2*TARGET_X,  dy = TARGET_Y - (yl+yr)/2
     (TARGET_X/Y = crop centre = laser aim point)
  5. PD correction:
       jog_x = -(Kp_x*dx + Kd_x*(dx-prev_dx)) * STEP_MM
       jog_y =  (Kp_y*dy + Kd_y*(dy-prev_dy)) * STEP_MM
     Clamped to [-MAX_JOG, +MAX_JOG].
  6. Publish GantryCommand(cmd="jog", x_mm=jog_x, y_mm=jog_y, feed_rate=FINE_FEED).
  7. Locked when |dx|<=DEADZONE and |dy|<=DEADZONE for settle_frames consecutive frames.
  8. Timed out when elapsed > max_time_sec.

Parameters:
  None — all tuning comes from config alignment_params.
"""

# ── NMS patch — MUST come before ultralytics ──────────────────────────────────
import importlib.util as _ilu
import laser_weeder._repo_path as _rp  # noqa: F401
from pathlib import Path as _Path

_nms_spec = _ilu.spec_from_file_location(
    "_nms_patch", _rp.REPO_ROOT / "bringup" / "_nms_patch.py"
)
_nms_mod = _ilu.module_from_spec(_nms_spec)
_nms_spec.loader.exec_module(_nms_mod)

# ── repo imports ──────────────────────────────────────────────────────────────
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

from config import (
    FRAME_WIDTH, FRAME_HEIGHT,
    FINE_ALIGN_CROP_SCALE,
    FINE_ALIGN_DEADZONE_PX,
    FINE_ALIGN_MAX_TIME_SEC,
    FINE_ALIGN_SETTLE_FRAMES,
    FINE_ALIGN_KP_X, FINE_ALIGN_KD_X,
    FINE_ALIGN_KP_Y, FINE_ALIGN_KD_Y,
    FINE_ALIGN_STEP_MM,
    FINE_ALIGN_MAX_JOG_MM,
    FINE_ALIGN_FEED,
    FINE_ALIGN_LK_WIN_SIZE,
    FINE_ALIGN_LK_MAX_LEVEL,
)
from laser_weeder_msgs.msg import FineAlignGoal, PixelError, GantryCommand

# ── crop constants (mirrors fine_align_motion.py) ─────────────────────────────
FULL_W = FRAME_WIDTH
FULL_H = FRAME_HEIGHT
FINE_W = int(FRAME_WIDTH  * FINE_ALIGN_CROP_SCALE)
FINE_H = int(FRAME_HEIGHT * FINE_ALIGN_CROP_SCALE)
CROP_X0 = (FULL_W - FINE_W) // 2
CROP_Y0 = (FULL_H - FINE_H) // 2
CROP_X1 = CROP_X0 + FINE_W
CROP_Y1 = CROP_Y0 + FINE_H
TARGET_X = FINE_W / 2.0   # laser aim x in crop space
TARGET_Y = FINE_H / 2.0   # laser aim y in crop space

LK_PARAMS = dict(
    winSize=(FINE_ALIGN_LK_WIN_SIZE, FINE_ALIGN_LK_WIN_SIZE),
    maxLevel=FINE_ALIGN_LK_MAX_LEVEL,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def _img_msg_to_array(msg: Image) -> np.ndarray:
    return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
        (msg.height, msg.width, 3)
    )


def _full_to_crop(x: float, y: float):
    return x - CROP_X0, y - CROP_Y0


def _crop_frame(frame: np.ndarray) -> np.ndarray:
    return frame[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1].copy()


def _inside_crop(cx: float, cy: float, margin: float = 8.0) -> bool:
    return margin <= cx < (FINE_W - margin) and margin <= cy < (FINE_H - margin)


def _compute_errors(xl, yl, xr, yr):
    """PD error in crop space.  Mirror of fine_align_motion._compute_errors."""
    err_x = (xl + xr) - (2.0 * TARGET_X)
    err_y = TARGET_Y - ((yl + yr) / 2.0)
    return float(err_x), float(err_y)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class _Session:
    """All state for a single fine-align run (lives inside a lock)."""
    __slots__ = (
        "active", "t0",
        "track_l", "track_r",
        "old_gray_l", "old_gray_r",
        "prev_ex", "prev_ey",
        "inside_cnt",
        "deadzone", "max_time", "settle_frames",
        "done",
    )

    def __init__(
        self,
        left_x, left_y, right_x, right_y,
        frame_l: np.ndarray, frame_r: np.ndarray,
        deadzone: float, max_time: float, settle_frames: int,
    ):
        crop_l = _crop_frame(frame_l)
        crop_r = _crop_frame(frame_r)
        cl_x, cl_y = _full_to_crop(left_x,  left_y)
        cr_x, cr_y = _full_to_crop(right_x, right_y)

        self.track_l = np.array([[[cl_x, cl_y]]], dtype=np.float32)
        self.track_r = np.array([[[cr_x, cr_y]]], dtype=np.float32)
        self.old_gray_l = cv2.cvtColor(crop_l, cv2.COLOR_BGR2GRAY)
        self.old_gray_r = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)
        self.prev_ex    = 0.0
        self.prev_ey    = 0.0
        self.inside_cnt = 0
        self.deadzone      = deadzone
        self.max_time      = max_time
        self.settle_frames = settle_frames
        self.t0     = time.monotonic()
        self.active = True
        self.done   = False


class FineAlignNode(Node):
    def __init__(self):
        super().__init__("fine_align_node")

        self._session: _Session | None = None
        self._sess_lock = threading.Lock()

        # Buffered latest frames for session initialisation
        self._latest_left:  np.ndarray | None = None
        self._latest_right: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        # Publishers
        self._pub_error = self.create_publisher(
            PixelError, "/lw/fine_align/pixel_error", 10
        )
        self._pub_cmd = self.create_publisher(
            GantryCommand, "/lw/gantry/command", 10
        )

        # Synchronised stereo frame subscriber
        self._sub_left  = Subscriber(self, Image, "/lw/left/image_raw")
        self._sub_right = Subscriber(self, Image, "/lw/right/image_raw")
        self._sync = ApproximateTimeSynchronizer(
            [self._sub_left, self._sub_right], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._on_pair)

        # Goal subscriber
        self._sub_goal = self.create_subscription(
            FineAlignGoal, "/lw/fine_align/goal", self._on_goal, 10
        )

        self.get_logger().info("FineAlignNode ready — waiting for /lw/fine_align/goal")

    # ── goal handler ──────────────────────────────────────────────────────────

    def _on_goal(self, msg: FineAlignGoal):
        if msg.cancel:
            with self._sess_lock:
                if self._session is not None:
                    self._session.active = False
                    self._session = None
            self.get_logger().info("FineAlign: session cancelled")
            return

        deadzone      = msg.deadzone_px    if msg.deadzone_px    > 0 else FINE_ALIGN_DEADZONE_PX
        max_time      = msg.max_time_sec   if msg.max_time_sec   > 0 else FINE_ALIGN_MAX_TIME_SEC
        settle_frames = msg.settle_frames  if msg.settle_frames  > 0 else FINE_ALIGN_SETTLE_FRAMES

        # Grab the most recent frame pair for tracker initialisation
        with self._frame_lock:
            frame_l = self._latest_left
            frame_r = self._latest_right

        if frame_l is None or frame_r is None:
            self.get_logger().warning("FineAlign: no frames available yet — goal ignored")
            return

        cl_x, cl_y = _full_to_crop(msg.left_x_px,  msg.left_y_px)
        cr_x, cr_y = _full_to_crop(msg.right_x_px, msg.right_y_px)
        if not (_inside_crop(cl_x, cl_y) and _inside_crop(cr_x, cr_y)):
            self.get_logger().warning(
                f"FineAlign: initial points outside centre crop — "
                f"L=({cl_x:.0f},{cl_y:.0f}) R=({cr_x:.0f},{cr_y:.0f}). Goal ignored."
            )
            return

        sess = _Session(
            left_x=msg.left_x_px, left_y=msg.left_y_px,
            right_x=msg.right_x_px, right_y=msg.right_y_px,
            frame_l=frame_l, frame_r=frame_r,
            deadzone=deadzone, max_time=max_time, settle_frames=settle_frames,
        )
        with self._sess_lock:
            self._session = sess

        self.get_logger().info(
            f"FineAlign: session started  "
            f"L=({msg.left_x_px:.0f},{msg.left_y_px:.0f})  "
            f"R=({msg.right_x_px:.0f},{msg.right_y_px:.0f})  "
            f"deadzone={deadzone:.1f}px max_time={max_time:.1f}s settle={settle_frames}"
        )

    # ── frame callback ────────────────────────────────────────────────────────

    def _on_pair(self, left_msg: Image, right_msg: Image):
        frame_l = _img_msg_to_array(left_msg)
        frame_r = _img_msg_to_array(right_msg)

        # Always cache latest frames for session init
        with self._frame_lock:
            self._latest_left  = frame_l
            self._latest_right = frame_r

        with self._sess_lock:
            sess = self._session

        if sess is None or not sess.active:
            return

        self._step(sess, frame_l, frame_r)

    # ── tracking + PD step ───────────────────────────────────────────────────

    def _step(self, sess: _Session, frame_l: np.ndarray, frame_r: np.ndarray):
        elapsed = time.monotonic() - sess.t0
        stamp   = self.get_clock().now().to_msg()

        crop_l = _crop_frame(frame_l)
        crop_r = _crop_frame(frame_r)
        gray_l = cv2.cvtColor(crop_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)

        # LK tracking
        new_l, st_l, _ = cv2.calcOpticalFlowPyrLK(
            sess.old_gray_l, gray_l, sess.track_l, None, **LK_PARAMS
        )
        new_r, st_r, _ = cv2.calcOpticalFlowPyrLK(
            sess.old_gray_r, gray_r, sess.track_r, None, **LK_PARAMS
        )

        tracking_ok = (
            st_l is not None and st_r is not None
            and st_l[0][0] != 0 and st_r[0][0] != 0
        )

        if not tracking_ok:
            self.get_logger().warning("FineAlign: LK tracking lost — ending session")
            self._end_session(sess, timed_out=False, locked=False, elapsed=elapsed, stamp=stamp)
            return

        xl = float(new_l[0, 0, 0])
        yl = float(new_l[0, 0, 1])
        xr = float(new_r[0, 0, 0])
        yr = float(new_r[0, 0, 1])

        if not (_inside_crop(xl, yl) and _inside_crop(xr, yr)):
            self.get_logger().warning("FineAlign: tracked point left crop boundary — ending session")
            self._end_session(sess, timed_out=False, locked=False, elapsed=elapsed, stamp=stamp)
            return

        # Update trackers
        sess.track_l = new_l
        sess.track_r = new_r
        sess.old_gray_l = gray_l
        sess.old_gray_r = gray_r

        err_x, err_y = _compute_errors(xl, yl, xr, yr)
        dex = err_x - sess.prev_ex
        dey = err_y - sess.prev_ey
        sess.prev_ex = err_x
        sess.prev_ey = err_y

        in_dz = abs(err_x) <= sess.deadzone and abs(err_y) <= sess.deadzone
        if in_dz:
            sess.inside_cnt += 1
        else:
            sess.inside_cnt = 0

        locked = sess.inside_cnt >= sess.settle_frames

        # Publish pixel error
        pe = PixelError()
        pe.header.stamp = stamp
        pe.dx_px       = err_x
        pe.dy_px       = err_y
        pe.in_deadzone = in_dz
        pe.locked      = locked
        pe.timed_out   = False
        pe.elapsed_sec = float(elapsed)
        self._pub_error.publish(pe)

        if locked:
            self.get_logger().info(
                f"FineAlign: LOCKED after {elapsed:.2f}s  err=({err_x:.1f},{err_y:.1f})px"
            )
            self._end_session(sess, timed_out=False, locked=True, elapsed=elapsed, stamp=stamp)
            return

        if elapsed >= sess.max_time:
            self.get_logger().warning(f"FineAlign: timeout after {elapsed:.2f}s")
            self._end_session(sess, timed_out=True, locked=False, elapsed=elapsed, stamp=stamp)
            return

        # PD jog command
        jog_x = _clamp(-(FINE_ALIGN_KP_X * err_x + FINE_ALIGN_KD_X * dex) * FINE_ALIGN_STEP_MM,
                        -FINE_ALIGN_MAX_JOG_MM, FINE_ALIGN_MAX_JOG_MM)
        jog_y = _clamp( (FINE_ALIGN_KP_Y * err_y + FINE_ALIGN_KD_Y * dey) * FINE_ALIGN_STEP_MM,
                        -FINE_ALIGN_MAX_JOG_MM, FINE_ALIGN_MAX_JOG_MM)

        cmd = GantryCommand()
        cmd.cmd       = "jog"
        cmd.x_mm      = float(jog_x)
        cmd.y_mm      = float(jog_y)
        cmd.feed_rate = float(FINE_ALIGN_FEED)
        self._pub_cmd.publish(cmd)

    def _end_session(
        self, sess: _Session, *,
        timed_out: bool, locked: bool, elapsed: float, stamp
    ):
        with self._sess_lock:
            if self._session is sess:
                self._session = None
        sess.active = False
        sess.done   = True

        # Final pixel error message
        pe = PixelError()
        pe.header.stamp = stamp
        pe.dx_px       = sess.prev_ex
        pe.dy_px       = sess.prev_ey
        pe.in_deadzone = (
            abs(sess.prev_ex) <= sess.deadzone and abs(sess.prev_ey) <= sess.deadzone
        )
        pe.locked      = locked
        pe.timed_out   = timed_out
        pe.elapsed_sec = float(elapsed)
        self._pub_error.publish(pe)


def main(args=None):
    rclpy.init(args=args)
    node = FineAlignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
