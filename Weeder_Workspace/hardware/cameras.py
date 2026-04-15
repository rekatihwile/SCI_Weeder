import cv2
import time
import os
import sys
import json
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

from config import (
    LEFT_CAMERA_INDEX,
    RIGHT_CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_SETTINGS,
    IS_WINDOWS,
    AUTO_MODE,
    BASE_DIR,
    TRIAL_RECORDINGS_DIR,
    RECORD_VIDEO_FPS,
    RECORD_VIDEO_SCALE,
)

# Windows uses Media Foundation, Linux (Jetson) uses V4L2
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2


def _rprint(msg):
    """Always prints to the real terminal, even if sys.stdout has been redirected."""
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()


def apply_camera_settings(cap, props, dev_path=None):
    if not props:
        return

    auto_exposure = float(props.get("auto_exposure", 1))
    auto_wb = float(props.get("auto_wb", 1))

    if os.name == "nt":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
        cap.set(cv2.CAP_PROP_AUTO_WB, auto_wb)
        if "exposure" in props: cap.set(cv2.CAP_PROP_EXPOSURE, float(props["exposure"]))
        if "gain" in props: cap.set(cv2.CAP_PROP_GAIN, float(props["gain"]))
        if "brightness" in props: cap.set(cv2.CAP_PROP_BRIGHTNESS, float(props["brightness"]))
        if "contrast" in props: cap.set(cv2.CAP_PROP_CONTRAST, float(props["contrast"]))
        if "saturation" in props: cap.set(cv2.CAP_PROP_SATURATION, float(props["saturation"]))
        if "white_balance" in props: cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(props["white_balance"]))
    else:
        if dev_path is not None:
            _rprint(f"[V4L2 Target] Applying startup settings strictly to -> {dev_path}")
            v4l2_auto_exp = 1 if auto_exposure == 1 else 3
            v4l2_auto_wb = 0 if auto_wb == 1 else 1

            os.system(f"v4l2-ctl -d {dev_path} -c exposure_auto={v4l2_auto_exp} > /dev/null 2>&1")
            os.system(f"v4l2-ctl -d {dev_path} -c white_balance_temperature_auto={v4l2_auto_wb} > /dev/null 2>&1")

            if "exposure" in props: os.system(f"v4l2-ctl -d {dev_path} -c exposure_absolute={int(props['exposure'])} > /dev/null 2>&1")
            if "gain" in props: os.system(f"v4l2-ctl -d {dev_path} -c gain={int(props['gain'])} > /dev/null 2>&1")
            if "brightness" in props: os.system(f"v4l2-ctl -d {dev_path} -c brightness={int(props['brightness'])} > /dev/null 2>&1")
            if "contrast" in props: os.system(f"v4l2-ctl -d {dev_path} -c contrast={int(props['contrast'])} > /dev/null 2>&1")
            if "saturation" in props: os.system(f"v4l2-ctl -d {dev_path} -c saturation={int(props['saturation'])} > /dev/null 2>&1")
            if "white_balance" in props: os.system(f"v4l2-ctl -d {dev_path} -c white_balance_temperature={int(props['white_balance'])} > /dev/null 2>&1")


class TrialRecorder:
    """
    Records stereo frames to SCI_Weeder/trial_recordings/<timestamp>.mp4

    Strategy (same as original BronnyJr.py):
      - Background thread samples latest_fL/fR at steady FPS → appends to frame_buffer
      - write() is non-blocking: just swaps the shared frame reference
      - release() stops the thread, then encodes the whole buffer to disk at once

    All output goes through _rprint() so it always appears in the terminal
    even when sys.stdout is redirected to a log file.
    """

    def __init__(self, filepath, fps=15.0, scale=0.5):
        self.filepath = str(filepath)
        self.fps = fps
        self.scale = scale

        self._frame_buffer = []
        self._latest_fl = None
        self._latest_fr = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        _rprint(f"[VIDEO] Capture thread started — saving to {self.filepath}")

    def _capture_loop(self):
        interval = 1.0 / self.fps
        while not self._stop_event.is_set():
            t0 = time.time()

            with self._lock:
                fl = self._latest_fl
                fr = self._latest_fr

            if fl is not None and fr is not None:
                h, w = fl.shape[:2]
                sh, sw = int(h * self.scale), int(w * self.scale)
                combined = np.hstack([
                    cv2.resize(fl, (sw, sh)),
                    cv2.resize(fr, (sw, sh)),
                ])
                self._frame_buffer.append(combined)

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    def write(self, frame_l, frame_r):
        """Non-blocking — just updates the shared frames for the capture thread."""
        with self._lock:
            self._latest_fl = frame_l
            self._latest_fr = frame_r

    def release(self):
        """Stop capture thread, encode buffered frames to disk."""
        _rprint("[VIDEO] Stopping capture thread...")
        self._stop_event.set()
        self._thread.join(timeout=3.0)

        n = len(self._frame_buffer)
        _rprint(f"[VIDEO] {n} frames captured. Encoding → {self.filepath}")

        if n == 0:
            _rprint("[VIDEO] WARNING: 0 frames in buffer — nothing to save.")
            _rprint("[VIDEO] Check that RECORD_TRIAL=True and that read_pair() was called after start_recording().")
            return

        h, w = self._frame_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            _rprint(f"[VIDEO] ERROR: VideoWriter could not open {self.filepath}")
            _rprint(f"[VIDEO] Check that the directory exists and is writable.")
            return

        for i, frame in enumerate(self._frame_buffer):
            writer.write(frame)
            if i % 30 == 0:
                sys.__stdout__.write(f"\r[VIDEO] Encoding {i}/{n}...")
                sys.__stdout__.flush()

        writer.release()
        _rprint(f"\n[VIDEO] Done! Saved {n} frames to:\n        {self.filepath}")


class StereoCameras:
    def __init__(self):
        self.left = None
        self.right = None
        self._recorder = None

        self.dev_paths = {
            "left": f"/dev/video{LEFT_CAMERA_INDEX}",
            "right": f"/dev/video{RIGHT_CAMERA_INDEX}"
        }

        hw_cfg = BASE_DIR / "params" / "hardware_config.json"
        if hw_cfg.exists():
            with open(hw_cfg, "r") as f:
                hw = json.load(f)
                if "cameras" in hw:
                    if hw["cameras"].get("left") and "device" in hw["cameras"]["left"]:
                        self.dev_paths["left"] = hw["cameras"]["left"]["device"]
                    if hw["cameras"].get("right") and "device" in hw["cameras"]["right"]:
                        self.dev_paths["right"] = hw["cameras"]["right"]["device"]

    def open(self):
        _rprint("\n=== OPENING CAMERAS ===")
        _rprint(f"Opening Left : {LEFT_CAMERA_INDEX}")
        _rprint(f"Opening Right: {RIGHT_CAMERA_INDEX}")

        self.left = cv2.VideoCapture(LEFT_CAMERA_INDEX, BACKEND)
        self.right = cv2.VideoCapture(RIGHT_CAMERA_INDEX, BACKEND)

        for name, cap in [("Left", self.left), ("Right", self.right)]:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {name} camera.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        time.sleep(0.5)

        apply_camera_settings(self.left, CAMERA_SETTINGS.get("left"), self.dev_paths["left"])
        apply_camera_settings(self.right, CAMERA_SETTINGS.get("right"), self.dev_paths["right"])

        if AUTO_MODE:
            if os.name == "nt":
                self.left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.left.set(cv2.CAP_PROP_AUTO_WB, 1)
                self.right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.right.set(cv2.CAP_PROP_AUTO_WB, 1)
            else:
                for side in ["left", "right"]:
                    dev = self.dev_paths[side]
                    if dev:
                        os.system(f"v4l2-ctl -d {dev} -c exposure_auto=3 > /dev/null 2>&1")
                        os.system(f"v4l2-ctl -d {dev} -c white_balance_temperature_auto=1 > /dev/null 2>&1")

        for _ in range(5):
            self.left.grab()
            self.right.grab()

        _rprint("Stereo cameras opened.")

    def _flip_frame(self, frame):
        return cv2.rotate(frame, cv2.ROTATE_180)

    def read_pair(self, retries=5):
        if self.left is None or self.right is None:
            raise RuntimeError("Cameras are not open.")

        for attempt in range(retries):
            self.left.grab()
            self.right.grab()

            ret_l, frame_l = self.left.retrieve()
            ret_r, frame_r = self.right.retrieve()

            if ret_l and ret_r:
                fl = self._flip_frame(frame_l)
                fr = self._flip_frame(frame_r)
                if self._recorder is not None:
                    self._recorder.write(fl, fr)
                return fl, fr

            _rprint(f"[WARN] Frame dropped by USB bus (attempt {attempt+1}/{retries}). Retrying...")
            time.sleep(0.05)

        raise RuntimeError("Failed to read stereo pair after multiple retries.")

    # ------------------------------------------------------------------
    # Trial recording
    # ------------------------------------------------------------------

    def start_recording(self):
        """
        Start recording to SCI_Weeder/trial_recordings/<timestamp>.mp4
        Uses RECORD_VIDEO_FPS and RECORD_VIDEO_SCALE from config.py.
        Call this once after survey confirm.
        """
        if self._recorder is not None:
            _rprint("[VIDEO] Recording already active — ignoring duplicate start_recording() call.")
            return

        TRIAL_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = TRIAL_RECORDINGS_DIR / f"{timestamp}.mp4"

        self._recorder = TrialRecorder(
            filepath=path,
            fps=RECORD_VIDEO_FPS,
            scale=RECORD_VIDEO_SCALE,
        )

    def stop_recording(self):
        """Encode and save the video. Safe to call even if not recording."""
        if self._recorder is not None:
            self._recorder.release()
            self._recorder = None

    def close(self):
        self.stop_recording()
        if self.left is not None:
            self.left.release()
        if self.right is not None:
            self.right.release()