import cv2
import re
import time
import os
import sys
import json
import queue
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
# change path to import config from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
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
    RECORD_TRIAL,
    RECORD_RAW_FRAMES_ONLY,
    RECORD_FRAME_FORMAT,
    RECORD_JPEG_QUALITY,
    RECORD_EVERY_N_FRAMES,
    RECORD_MAX_FPS,
    RECORD_MIN_INTERVAL_SEC,
    RECORD_LIVE_VIDEO,
    RECORD_LIVE_OVERLAYS,
    RECORD_VIDEO_FPS,
    RECORD_VIDEO_SCALE,
    RECORD_VIDEO_TIMESTAMP,
    RECORD_VIDEO_OVERLAY,
    RECORD_VIDEO_DEBUG,
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


def _next_recording_index(recordings_dir):
    """Return the next sequential trial index by scanning existing trial_NNN outputs."""
    max_idx = 0
    for p in Path(recordings_dir).glob("trial_*"):
        m = re.match(r"trial_(\d+)_", p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


class LiveVideoRecorder:
    """
    Async frame queue that encodes to MP4 at the end of a trial.

    Design:
      - write() is non-blocking: just updates a shared "latest frame" reference.
      - A background thread wakes every 1/fps seconds, snapshots the latest frame,
        and appends (timestamp, combined_frame) to an in-memory queue.
      - release() stops the thread, prints timing stats, then encodes the queue
        to disk in one pass using the target fps written into the file header.

    Timestamps let you verify actual vs target sample rate after the fact.
    All output goes through _rprint() so it always appears in the terminal
    even when sys.stdout is redirected to a log file.
    """

    def __init__(self, filepath, fps=15.0, scale=0.5):
        self.filepath = str(filepath)
        self.fps      = fps
        self.scale    = scale

        self._frame_queue      = []
        self._latest_fl        = None
        self._latest_fr        = None
        self._latest_overlay_l = None   # annotated frames from the CV pipeline
        self._latest_overlay_r = None
        self._status_lines     = []     # text lines burned into every recorded frame
        self._lock             = threading.Lock()
        self._stop_event       = threading.Event()
        self._start_time       = time.time()

        self._ckpt_path   = self.filepath.replace(".mp4", ".partial.avi")
        self._ckpt_writer = None

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        _rprint(f"[VIDEO] Recorder started — target {fps:.1f} fps  scale={scale}")
        _rprint(f"[VIDEO] Final:      {self.filepath}")
        _rprint(f"[VIDEO] Crash-safe: {self._ckpt_path}")

    def set_status_lines(self, lines):
        """Set text lines burned into every recorded frame (top-left corner)."""
        with self._lock:
            self._status_lines = list(lines)

    def set_context(self, **context):
        """Compatibility hook used by the raw recorder."""
        return

    def _burn_status(self, frame):
        with self._lock:
            lines = list(self._status_lines)
        y = 26
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y += 24

    def _burn_timestamp(self, frame, t0):
        elapsed = t0 - self._start_time
        wall    = datetime.now().strftime("%H:%M:%S")
        label   = f"T+{elapsed:.1f}s  {wall}"
        y       = frame.shape[0] - 8
        cv2.putText(frame, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,   0,   0), 3)
        cv2.putText(frame, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    def _capture_loop(self):
        interval      = 1.0 / self.fps
        debug_counter = 0
        while not self._stop_event.is_set():
            t0 = time.time()

            with self._lock:
                if (RECORD_LIVE_OVERLAYS or RECORD_VIDEO_OVERLAY) and self._latest_overlay_l is not None:
                    fl = self._latest_overlay_l
                    fr = self._latest_overlay_r
                else:
                    fl = self._latest_fl
                    fr = self._latest_fr

            if fl is not None and fr is not None:
                h, w = fl.shape[:2]
                if self.scale != 1.0:
                    sh, sw = int(h * self.scale), int(w * self.scale)
                    combined = np.hstack([
                        cv2.resize(fl, (sw, sh)),
                        cv2.resize(fr, (sw, sh)),
                    ])
                else:
                    combined = np.hstack([fl.copy(), fr.copy()])

                self._burn_status(combined)
                if RECORD_VIDEO_TIMESTAMP:
                    self._burn_timestamp(combined, t0)

                self._frame_queue.append((t0, combined))

                if self._ckpt_writer is None:
                    out_h, out_w = combined.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    self._ckpt_writer = cv2.VideoWriter(
                        self._ckpt_path, fourcc, self.fps, (out_w, out_h)
                    )
                self._ckpt_writer.write(combined)

                debug_counter += 1
                if RECORD_VIDEO_DEBUG and debug_counter % 30 == 0:
                    n, dur, actual_fps = self.stats()
                    expected_t = self._frame_queue[0][0] + (n - 1) * interval
                    drift_ms   = (t0 - expected_t) * 1000.0
                    _rprint(
                        f"[VIDEO DEBUG] queue={n} | elapsed={dur:.1f}s | "
                        f"fps={actual_fps:.2f}/{self.fps:.1f} | drift={drift_ms:+.1f}ms"
                    )

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    def write(self, frame_l, frame_r):
        """Non-blocking — updates raw frame reference for the capture thread.
        Frames with unexpected dimensions (e.g., from an HD survey burst) are
        silently dropped so the video stream stays at a consistent resolution."""
        if frame_l is None or frame_r is None:
            return
        if frame_l.shape[1] != FRAME_WIDTH or frame_l.shape[0] != FRAME_HEIGHT:
            return
        with self._lock:
            self._latest_fl = frame_l
            self._latest_fr = frame_r

    def write_overlay(self, frame_l, frame_r):
        """Push annotated frames (bboxes, keypoints, LK vectors) for recording.
        Pass None for both to clear the overlay and fall back to raw frames."""
        with self._lock:
            self._latest_overlay_l = frame_l
            self._latest_overlay_r = frame_r

    def stats(self):
        """Return (frame_count, actual_duration_sec, actual_fps) from the current queue."""
        n = len(self._frame_queue)
        if n < 2:
            return n, 0.0, 0.0
        duration   = self._frame_queue[-1][0] - self._frame_queue[0][0]
        actual_fps = (n - 1) / duration if duration > 0 else 0.0
        return n, duration, actual_fps

    def release(self):
        """Stop the capture thread, print timing stats, encode queue to MP4, clean up checkpoint."""
        _rprint("[VIDEO] Stopping capture thread...")
        self._stop_event.set()
        self._thread.join(timeout=3.0)

        # Close checkpoint writer so the AVI is finalised on disk.
        if self._ckpt_writer is not None:
            self._ckpt_writer.release()
            self._ckpt_writer = None

        n = len(self._frame_queue)

        if n == 0:
            _rprint("[VIDEO] WARNING: 0 frames in queue — nothing to save.")
            _rprint("[VIDEO] Check that RECORD_TRIAL=True and read_pair() was called after open().")
            return

        _, duration, actual_fps = self.stats()
        _rprint(
            f"[VIDEO] {n} frames | duration {duration:.1f}s | "
            f"actual {actual_fps:.2f} fps (target {self.fps:.1f} fps)"
        )
        _rprint(f"[VIDEO] Encoding → {self.filepath}")

        h, w = self._frame_queue[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            _rprint(f"[VIDEO] ERROR: VideoWriter could not open {self.filepath}")
            _rprint(f"[VIDEO] Crash-safe partial video is still at: {self._ckpt_path}")
            return

        for i, (ts, frame) in enumerate(self._frame_queue):
            writer.write(frame)
            if i % 30 == 0:
                sys.__stdout__.write(f"\r[VIDEO] Encoding {i}/{n}...")
                sys.__stdout__.flush()

        writer.release()
        _rprint(f"\n[VIDEO] Done! Saved {n} frames to:\n        {self.filepath}")

        # Delete crash-safe file now that final encode succeeded.
        try:
            os.remove(self._ckpt_path)
        except FileNotFoundError:
            pass


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class RawFrameRecorder:
    """
    Lightweight live recorder.

    The live path only enqueues left/right frames and writes them as raw image files
    on a background thread. It does not draw overlays, stitch frames, resize, or encode
    a video while the experiment is running.
    """

    def __init__(
        self,
        run_dir,
        fps=15.0,
        frame_format="jpg",
        jpeg_quality=90,
        every_n_frames=1,
        min_interval_sec=0.0,
        queue_size=16,
    ):
        self.run_dir = Path(run_dir)
        self.left_dir = self.run_dir / "left"
        self.right_dir = self.run_dir / "right"
        self.left_dir.mkdir(parents=True, exist_ok=True)
        self.right_dir.mkdir(parents=True, exist_ok=True)

        fmt = str(frame_format or "jpg").lower().lstrip(".")
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt not in ("jpg", "png"):
            raise ValueError(f"Unsupported RECORD_FRAME_FORMAT: {frame_format}")

        self.frame_format = fmt
        self.jpeg_quality = int(jpeg_quality)
        self.every_n_frames = max(1, int(every_n_frames or 1))
        if RECORD_MAX_FPS:
            self.min_interval_sec = 1.0 / float(RECORD_MAX_FPS)
        else:
            self.min_interval_sec = max(0.0, float(min_interval_sec or 0.0))
        self.fps = fps

        self.manifest_path = self.run_dir / "manifest.jsonl"
        self.metadata_path = self.run_dir / "recording_meta.json"
        self._manifest_f = open(self.manifest_path, "a", buffering=1)
        self._queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._context = {}
        self._stop_event = threading.Event()
        self._start_mono = time.monotonic()
        self._last_saved_mono = 0.0
        self._seen_frames = 0
        self._saved_frames = 0
        self._dropped_frames = 0
        self._next_frame_index = 1
        self._total_save_time_s = 0.0

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        self._write_metadata(status="running")
        _rprint(
            f"[REC] Raw recorder started: {self.run_dir} "
            f"format={self.frame_format} every={self.every_n_frames} "
            f"min_interval={self.min_interval_sec:.3f}s"
        )

    def _write_metadata(self, status):
        data = {
            "status": status,
            "started_monotonic": self._start_mono,
            "frame_format": self.frame_format,
            "jpeg_quality": self.jpeg_quality,
            "every_n_frames": self.every_n_frames,
            "min_interval_sec": self.min_interval_sec,
            "manifest": self.manifest_path.name,
            "left_dir": self.left_dir.relative_to(self.run_dir).as_posix(),
            "right_dir": self.right_dir.relative_to(self.run_dir).as_posix(),
            "stats": self.stats(),
        }
        with open(self.metadata_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def set_status_lines(self, lines):
        self.set_context(status_lines=list(lines))

    def set_context(self, **context):
        with self._lock:
            self._context.update(_json_safe(context))

    def write_overlay(self, frame_l, frame_r):
        # Raw recording intentionally ignores live overlays.
        return

    def _should_record_locked(self, timestamp_monotonic):
        self._seen_frames += 1
        if (self._seen_frames - 1) % self.every_n_frames != 0:
            return False
        if self.min_interval_sec > 0.0:
            if timestamp_monotonic - self._last_saved_mono < self.min_interval_sec:
                return False
        self._last_saved_mono = timestamp_monotonic
        return True

    def write(self, frame_l, frame_r):
        if frame_l is None or frame_r is None:
            return

        timestamp_monotonic = time.monotonic()
        with self._lock:
            if not self._should_record_locked(timestamp_monotonic):
                return
            context = dict(self._context)
            idx = self._next_frame_index
            self._next_frame_index += 1

        left_rel = Path("left") / f"left_{idx:06d}.{self.frame_format}"
        right_rel = Path("right") / f"right_{idx:06d}.{self.frame_format}"
        record = {
            "frame_index": idx,
            "left_image_path": left_rel.as_posix(),
            "right_image_path": right_rel.as_posix(),
            "timestamp_monotonic": timestamp_monotonic,
            "timestamp_wall": datetime.now().isoformat(timespec="milliseconds"),
            **context,
        }

        try:
            self._queue.put_nowait((frame_l, frame_r, left_rel, right_rel, record))
        except queue.Full:
            self._dropped_frames += 1
            if RECORD_VIDEO_DEBUG and self._dropped_frames % 20 == 1:
                _rprint(f"[REC DEBUG] raw recorder queue full; dropped={self._dropped_frames}")

    def _imwrite_params(self):
        if self.frame_format == "jpg":
            return [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        if self.frame_format == "png":
            return [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
        return []

    def _writer_loop(self):
        params = self._imwrite_params()
        _io_error = False
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frame_l, frame_r, left_rel, right_rel, record = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if _io_error:
                self._dropped_frames += 1
                self._queue.task_done()
                continue

            try:
                t_save = time.perf_counter()
                ok_l = cv2.imwrite(str(self.run_dir / left_rel), frame_l, params)
                ok_r = cv2.imwrite(str(self.run_dir / right_rel), frame_r, params)
                save_dt = time.perf_counter() - t_save
                self._total_save_time_s += save_dt

                record["recording_save_time_s"] = round(save_dt, 6)
                record["recording_ok"] = bool(ok_l and ok_r)
                self._manifest_f.write(json.dumps(_json_safe(record), default=str) + "\n")
                self._saved_frames += 1
            except OSError as e:
                _rprint(f"[REC] Write error ({e}); dropping remaining frames.")
                _io_error = True
                self._dropped_frames += 1
            finally:
                self._queue.task_done()

    def stats(self):
        elapsed = max(0.0, time.monotonic() - self._start_mono)
        return {
            "recording_dir": str(self.run_dir),
            "recording_manifest_path": str(self.manifest_path),
            "recording_frames_seen": int(self._seen_frames),
            "recording_frames_saved": int(self._saved_frames),
            "recording_frames_dropped": int(self._dropped_frames),
            "recording_frame_save_time_s": round(float(self._total_save_time_s), 3),
            "recording_elapsed_s": round(elapsed, 3),
        }

    def release(self):
        _rprint("[REC] Stopping raw recorder...")
        self._stop_event.set()
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            _rprint("[REC] WARNING: raw recorder still flushing in background.")
        try:
            self._manifest_f.flush()
            self._manifest_f.close()
        except Exception:
            pass
        self._write_metadata(status="complete")
        stats = self.stats()
        _rprint(
            f"[REC] Saved {stats['recording_frames_saved']} frame pair(s), "
            f"dropped {stats['recording_frames_dropped']}, "
            f"frame-save time {stats['recording_frame_save_time_s']:.3f}s"
        )
        _rprint(f"[REC] Manifest: {self.manifest_path}")


class StereoCameras:
    def __init__(self):
        self.left = None
        self.right = None
        self._recorder = None
        self._last_recording_dir = None
        self._last_recording_stats = {}
        self._cam_lock = threading.Lock()
        self._bg_stop = threading.Event()
        self._bg_thread = None
        self._last_read_time = 0.0  # tracks when read_pair() last ran

        self.dev_paths = {
            "left": f"/dev/video{LEFT_CAMERA_INDEX}",
            "right": f"/dev/video{RIGHT_CAMERA_INDEX}"
        }

        hw_cfg = BASE_DIR / "params" / "hardware" / "hardware_config.json"
        if hw_cfg.exists():
            with open(hw_cfg, "r") as f:
                hw = json.load(f)
                if "cameras" in hw:
                    if hw["cameras"].get("left") and "device" in hw["cameras"]["left"]:
                        self.dev_paths["left"] = hw["cameras"]["left"]["device"]
                    if hw["cameras"].get("right") and "device" in hw["cameras"]["right"]:
                        self.dev_paths["right"] = hw["cameras"]["right"]["device"]

    def open(self, start_recorder=True):
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

        self._bg_stop.clear()
        self._bg_thread = threading.Thread(target=self._bg_grab_loop, daemon=True)
        self._bg_thread.start()

        if RECORD_TRIAL and start_recorder:
            _rprint("[REC] RECORD_TRIAL=True — auto-starting recorder.")
            self.start_recording()

    def set_resolution(self, width, height):
        """Switch capture resolution on both cameras. Flushes stale frames after switch."""
        # Keep the bg thread backed off for the entire duration of the switch.
        self._last_read_time = time.time()
        with self._cam_lock:
            for cap in (self.left, self.right):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # V4L2 needs a few frames to flush the old-resolution buffer.
        time.sleep(0.1)
        for _ in range(5):
            self._last_read_time = time.time()
            self.left.grab()
            self.right.grab()

    def _flip_frame(self, frame):
        return cv2.rotate(frame, cv2.ROTATE_180)

    def _bg_grab_loop(self):
        """Grab frames during gantry transit so the recorder has fresh content.
        Backs off when read_pair() is being called actively (detection/tracking)
        to avoid consuming frames that belong to the CV pipeline."""
        interval = 1.0 / RECORD_VIDEO_FPS
        while not self._bg_stop.is_set():
            t0 = time.time()
            # Skip if read_pair was called recently — let the CV pipeline own the bus.
            if time.time() - self._last_read_time > 0.15:
                try:
                    with self._cam_lock:
                        self.left.grab()
                        self.right.grab()
                        ret_l, fl = self.left.retrieve()
                        ret_r, fr = self.right.retrieve()
                    if ret_l and ret_r and self._recorder is not None:
                        fl_f = self._flip_frame(fl)
                        fr_f = self._flip_frame(fr)
                        if isinstance(self._recorder, LiveVideoRecorder):
                            # Legacy stitched video needs a fixed frame size.
                            if fl_f.shape[1] == FRAME_WIDTH and fl_f.shape[0] == FRAME_HEIGHT:
                                self._recorder.write(fl_f, fr_f)
                        else:
                            self._recorder.write(fl_f, fr_f)
                except Exception:
                    pass
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    def read_pair(self, retries=5):
        if self.left is None or self.right is None:
            raise RuntimeError("Cameras are not open.")

        for attempt in range(retries):
            with self._cam_lock:
                self.left.grab()
                self.right.grab()
                ret_l, frame_l = self.left.retrieve()
                ret_r, frame_r = self.right.retrieve()

            if ret_l and ret_r:
                self._last_read_time = time.time()
                fl = self._flip_frame(frame_l)
                fr = self._flip_frame(frame_r)
                if self._recorder is not None:
                    self._recorder.write(fl, fr)
                return fl, fr

            _rprint(f"[WARN] Frame dropped by USB bus (attempt {attempt+1}/{retries}). Retrying...")
            time.sleep(0.05)

        raise RuntimeError("Failed to read stereo pair after multiple retries.")

    def clear_overlay(self):
        """Reset the recording overlay so raw frames are shown until the next CV annotation."""
        if self._recorder is not None:
            self._recorder.write_overlay(None, None)

    def set_recording_status(self, lines):
        """Update recording status context. Legacy video burns it into frames."""
        if self._recorder is not None:
            self._recorder.set_status_lines(lines)

    def set_recording_context(self, **context):
        """Attach JSON-serializable state to future raw frame manifest records."""
        if self._recorder is not None and hasattr(self._recorder, "set_context"):
            self._recorder.set_context(**context)

    def get_recording_dir(self):
        if self._recorder is None:
            return self._last_recording_dir
        run_dir = getattr(self._recorder, "run_dir", None)
        if run_dir is not None:
            return Path(run_dir)
        filepath = getattr(self._recorder, "filepath", None)
        return Path(filepath).parent if filepath else None

    def get_recording_stats(self):
        if self._recorder is None or not hasattr(self._recorder, "stats"):
            return dict(self._last_recording_stats)
        stats = self._recorder.stats()
        if isinstance(stats, tuple):
            n, duration, actual_fps = stats
            return {
                "recording_frames_saved": n,
                "recording_elapsed_s": round(duration, 3),
                "recording_actual_fps": round(actual_fps, 3),
            }
        return dict(stats)

    # ------------------------------------------------------------------
    # Trial recording
    # ------------------------------------------------------------------

    def start_recording(self):
        """
        Start trial recording.

        Default: raw left/right image files plus manifest.jsonl.
        Optional legacy path: stitched live MP4 when RECORD_RAW_FRAMES_ONLY=False
        and RECORD_LIVE_VIDEO=True.

        No-ops if RECORD_TRIAL=False or a recorder is already active.
        Called automatically by open() when RECORD_TRIAL=True; also safe
        to call manually (e.g. to start recording only from survey confirm).
        """
        if not RECORD_TRIAL:
            _rprint("[REC] RECORD_TRIAL=False — recording skipped.")
            return

        if self._recorder is not None:
            _rprint("[REC] Recording already active — ignoring duplicate start_recording() call.")
            return

        TRIAL_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        idx       = _next_recording_index(TRIAL_RECORDINGS_DIR)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if RECORD_RAW_FRAMES_ONLY or not RECORD_LIVE_VIDEO:
            run_dir = TRIAL_RECORDINGS_DIR / f"trial_{idx:03d}_{timestamp}"
            self._recorder = RawFrameRecorder(
                run_dir=run_dir,
                fps=RECORD_VIDEO_FPS,
                frame_format=RECORD_FRAME_FORMAT,
                jpeg_quality=RECORD_JPEG_QUALITY,
                every_n_frames=RECORD_EVERY_N_FRAMES,
                min_interval_sec=RECORD_MIN_INTERVAL_SEC,
            )
        else:
            path = TRIAL_RECORDINGS_DIR / f"trial_{idx:03d}_{timestamp}.mp4"
            self._recorder = LiveVideoRecorder(
                filepath=path,
                fps=RECORD_VIDEO_FPS,
                scale=RECORD_VIDEO_SCALE,
            )

    def stop_recording(self):
        """Flush and close the active recorder. Safe to call even if not recording."""
        if self._recorder is not None:
            self._last_recording_dir = self.get_recording_dir()
            self._recorder.release()
            self._last_recording_stats = self.get_recording_stats()
            self._recorder = None

    def close(self):
        self._bg_stop.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=2.0)
            self._bg_thread = None
        self.stop_recording()
        if self.left is not None:
            self.left.release()
        if self.right is not None:
            self.right.release()

# -----------------------------------------------------------------------------
# Standalone camera utility
# -----------------------------------------------------------------------------

def _resolve_weight_path(model_name_or_path, model_map, weights_dir):
    if model_name_or_path is None:
        return None
    filename = model_map.get(model_name_or_path, model_name_or_path)
    path = Path(filename)
    if not path.is_absolute():
        path = Path(weights_dir) / path
    return path


def _draw_text_lines(frame, lines, x=10, y=26, color=(0, 255, 255)):
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)
        y += 25


def _display_pair(left, right, display_scale):
    if display_scale != 1.0:
        left = cv2.resize(
            left,
            (int(left.shape[1] * display_scale), int(left.shape[0] * display_scale)),
        )
        right = cv2.resize(
            right,
            (int(right.shape[1] * display_scale), int(right.shape[0] * display_scale)),
        )
    return np.hstack([left, right])


def _build_ai_debug_detector(display_scale):
    from config import (
        AI_CONFIDENCE,
        AI_TARGET_CLASS,
        CV_WEIGHTS_DIR,
        DEFAULT_MODEL,
        DEFAULT_QPOINT_MODEL,
        MODEL_MAP,
    )
    try:
        from vision.detectors.ai_detector import AIDetector
    except ModuleNotFoundError as exc:
        _rprint(f"[AI DEBUG] Missing Python package while importing AI detector: {exc.name}")
        _rprint("[AI DEBUG] Install requirements.txt or select the IDE interpreter that has the AI stack.")
        raise

    yolo_path = _resolve_weight_path(DEFAULT_MODEL, MODEL_MAP, CV_WEIGHTS_DIR)
    qpoint_path = _resolve_weight_path(DEFAULT_QPOINT_MODEL, MODEL_MAP, CV_WEIGHTS_DIR)

    _rprint("\n=== AI DEBUG MODEL ===")
    _rprint(f"[AI DEBUG] YOLO model:   {DEFAULT_MODEL} -> {yolo_path}")
    _rprint(f"[AI DEBUG] QPoint model: {DEFAULT_QPOINT_MODEL} -> {qpoint_path}")

    if yolo_path is None or not yolo_path.exists():
        _rprint(f"[AI DEBUG] WARNING: YOLO weight file not found: {yolo_path}")
    if qpoint_path is None or not qpoint_path.exists():
        _rprint(f"[AI DEBUG] WARNING: QPoint weight file not found: {qpoint_path}")

    return AIDetector(
        display_scale=display_scale,
        yolo_path=yolo_path,
        qpoint_path=qpoint_path,
        conf=AI_CONFIDENCE,
        target_class=AI_TARGET_CLASS,
    )


def _scan_ai_debug_side(core, frame):
    boxes, masks = core._get_filtered_results(frame)
    qpoints = {}
    qpoint_error = None

    if boxes and core.qpoint_model is not None:
        try:
            for gx, gy, box_idx, peak_conf in core._run_qpoints_batch(frame, boxes, masks):
                qpoints[box_idx] = {
                    "point": (int(gx), int(gy)),
                    "peak": float(peak_conf),
                    "source": "heatmap",
                }
        except Exception as exc:
            qpoint_error = str(exc)

    detections = []
    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        cls_id = int(box.cls[0].cpu().item())
        cls_name = core.yolo.names.get(cls_id, str(cls_id))
        conf = float(box.conf[0].cpu().item())

        qpoint = qpoints.get(i)
        if qpoint is None:
            qpoint = {
                "point": (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
                "peak": None,
                "source": "box_center" if core.qpoint_model is None else "fallback_center",
            }

        detections.append({
            "box": (x1, y1, x2, y2),
            "cls": cls_id,
            "name": cls_name,
            "conf": conf,
            "point": qpoint["point"],
            "peak": qpoint["peak"],
            "source": qpoint["source"],
        })

    raw_counts = []
    if not detections:
        conf_values = []
        for conf in (0.05, 0.10, 0.20, core.conf, 0.50):
            conf = max(0.001, min(0.999, float(conf)))
            if all(abs(conf - existing) > 1e-6 for existing in conf_values):
                conf_values.append(conf)
        for conf in sorted(conf_values):
            try:
                raw_counts.append((conf, core.count_at_conf(frame, conf)))
            except Exception as exc:
                raw_counts.append((conf, f"err: {exc}"))
                break

    return {
        "detections": detections,
        "raw_counts": raw_counts,
        "qpoint_error": qpoint_error,
    }


def _print_ai_debug_side_report(side_name, info):
    detections = info["detections"]
    _rprint(f"[AI DEBUG] {side_name}: {len(detections)} filtered detection(s)")

    if info["qpoint_error"]:
        _rprint(f"  QPoint inference error: {info['qpoint_error']}")

    if info["raw_counts"]:
        counts = " | ".join(f"conf {conf:.2f}: {count}" for conf, count in info["raw_counts"])
        _rprint(f"  Raw YOLO count check (no qpoint / no IoM): {counts}")

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        peak = "n/a" if det["peak"] is None else f"{det['peak']:.4f}"
        _rprint(
            f"  det {i}: {det['name']} cls={det['cls']} "
            f"yolo={det['conf']:.3f} heatmap_peak={peak} "
            f"qpoint={det['point']} source={det['source']} "
            f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
        )


def _draw_ai_debug_side(frame, side_name, info, scan_label):
    detections = info["detections"]
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        px, py = det["point"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.circle(out, (int(px), int(py)), 7, (0, 0, 255), -1)
        cv2.circle(out, (int(px), int(py)), 11, (255, 255, 255), 1)

        label = f"{det['name']} {det['conf']:.2f}"
        if det["peak"] is not None:
            label += f" peak {det['peak']:.2f}"
        elif det["source"] != "heatmap":
            label += " center"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(th + 6, y1 - 5)
        cv2.rectangle(out, (x1, text_y - th - 5), (x1 + tw + 4, text_y + 2), (0, 220, 0), -1)
        cv2.putText(out, label, (x1 + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    lines = [
        f"{side_name}: {len(detections)} detections",
        scan_label,
    ]
    if not detections:
        lines.append("No filtered boxes")
        if info["raw_counts"]:
            counts = " ".join(f"{conf:.2f}:{count}" for conf, count in info["raw_counts"])
            lines.append(f"raw YOLO {counts}")
    if info["qpoint_error"]:
        lines.append("QPoint error - see terminal")

    _draw_text_lines(out, lines)
    return out


def _run_ai_debug_scan(detector, left, right):
    scan_label = datetime.now().strftime("scan %H:%M:%S")
    _rprint(f"\n[AI DEBUG] {scan_label}")
    _rprint(f"[AI DEBUG] Frame shape: left={left.shape[1]}x{left.shape[0]} right={right.shape[1]}x{right.shape[0]}")

    left_info = _scan_ai_debug_side(detector.cv_left, left)
    right_info = _scan_ai_debug_side(detector.cv_right, right)

    _print_ai_debug_side_report("LEFT", left_info)
    _print_ai_debug_side_report("RIGHT", right_info)

    return (
        _draw_ai_debug_side(left, "LEFT", left_info, scan_label),
        _draw_ai_debug_side(right, "RIGHT", right_info, scan_label),
    )


def _draw_match_labels(frame, side, matched_targets):
    out = frame.copy()
    key = "left_px" if side == "left" else "right_px"

    for i, target in enumerate(matched_targets):
        px, py = target[key]
        label = f"M{i}"
        x = int(px) + 12
        y = max(18, int(py) - 12)
        cv2.circle(out, (int(px), int(py)), 14, (255, 0, 255), 2)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    return out


def _print_burst_debug_report(stable_left, stable_right, matched_targets, unmatched_left, unmatched_right):
    _rprint(
        f"[AI DEBUG] Burst stable: left={len(stable_left)} right={len(stable_right)} "
        f"matched={len(matched_targets)}"
    )
    if unmatched_left or unmatched_right:
        _rprint(f"[AI DEBUG] Unmatched: left={len(unmatched_left)} right={len(unmatched_right)}")

    for i, target in enumerate(matched_targets):
        score = target.get("score", 0.0)
        iou = target.get("box_iou")
        cls_id = target.get("left_cls", target.get("right_cls"))
        left_conf = target.get("left_conf")
        right_conf = target.get("right_conf")
        parts = [
            f"  M{i}: L={target['left_px']} R={target['right_px']}",
            f"score={score:.3f}",
        ]
        if iou is not None:
            parts.append(f"box_iou={iou:.3f}")
        if cls_id is not None:
            parts.append(f"cls={cls_id}")
        if left_conf is not None and right_conf is not None:
            parts.append(f"conf L/R={left_conf:.3f}/{right_conf:.3f}")
        _rprint(" | ".join(parts))


def _run_ai_debug_burst_match(cams, detector, coarse_mover):
    from config import (
        SURVEY_BURST_COUNT,
        SURVEY_CLUSTER_RADIUS_PX,
        SURVEY_MIN_HITS,
        SURVEY_TARGET_CLASSES,
    )
    from vision.matching import match_points

    scan_label = datetime.now().strftime("burst %H:%M:%S")
    _rprint(f"\n[AI DEBUG] {scan_label}")
    _rprint(
        f"[AI DEBUG] Running burst matching: frames={SURVEY_BURST_COUNT} "
        f"min_hits={SURVEY_MIN_HITS} cluster_radius={SURVEY_CLUSTER_RADIUS_PX} "
        f"survey_classes={SURVEY_TARGET_CLASSES}"
    )

    stable_left, stable_right = coarse_mover.detect_stable_points(
        cams,
        detector,
        detector_mode="ai",
        burst_count=SURVEY_BURST_COUNT,
        min_hits=SURVEY_MIN_HITS,
        cluster_radius_px=SURVEY_CLUSTER_RADIUS_PX,
        survey_classes=SURVEY_TARGET_CLASSES,
    )

    matched_targets, unmatched_left, unmatched_right = match_points(
        stable_left,
        stable_right,
        verbose=True,
    )
    _print_burst_debug_report(stable_left, stable_right, matched_targets, unmatched_left, unmatched_right)

    left_frame = coarse_mover.last_survey_frameL
    right_frame = coarse_mover.last_survey_frameR
    if left_frame is None or right_frame is None:
        left_frame, right_frame = cams.read_pair()

    left_out = detector.cv_left.draw_stable_detections(left_frame, stable_left)
    right_out = detector.cv_right.draw_stable_detections(right_frame, stable_right)
    left_out = _draw_match_labels(left_out, "left", matched_targets)
    right_out = _draw_match_labels(right_out, "right", matched_targets)

    _draw_text_lines(
        left_out,
        [
            f"LEFT burst stable: {len(stable_left)}",
            f"matched pairs: {len(matched_targets)}",
            scan_label,
        ],
    )
    _draw_text_lines(
        right_out,
        [
            f"RIGHT burst stable: {len(stable_right)}",
            f"matched pairs: {len(matched_targets)}",
            scan_label,
        ],
    )

    return left_out, right_out


def _show_raw_preview(cams, display_scale):
    window_name = "Stereo camera preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _rprint("\n[CAMERA] Raw preview. Press q or Esc in the image window to quit.")

    while True:
        left, right = cams.read_pair()
        display_left = left.copy()
        display_right = right.copy()
        _draw_text_lines(display_left, ["LEFT live"])
        _draw_text_lines(display_right, ["RIGHT live"])
        cv2.imshow(window_name, _display_pair(display_left, display_right, display_scale))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyWindow(window_name)


def _show_ai_debug_preview(cams, display_scale):
    detector = _build_ai_debug_detector(display_scale)
    from control.coarse_move import TriangulationCoarseMover
    coarse_mover = TriangulationCoarseMover()

    window_name = "Stereo AI debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    _rprint("\n[AI DEBUG] Focus the OpenCV image window.")
    _rprint("[AI DEBUG] SPACE = burst-stable YOLO + QPoint scan, then stereo match")
    _rprint("[AI DEBUG] l = return to live preview | q/Esc = quit")

    show_last_scan = False
    last_scan_left = None
    last_scan_right = None

    while True:
        left, right = cams.read_pair()

        if show_last_scan and last_scan_left is not None and last_scan_right is not None:
            display_left = last_scan_left.copy()
            display_right = last_scan_right.copy()
            frame_y = left.shape[0] - 48
            _draw_text_lines(display_left, ["Last scan frozen", "SPACE rescan | l live"], y=frame_y)
            _draw_text_lines(display_right, ["Last scan frozen", "SPACE rescan | l live"], y=frame_y)
        else:
            display_left = left.copy()
            display_right = right.copy()
            _draw_text_lines(display_left, ["LEFT live", "SPACE burst match | q quit"])
            _draw_text_lines(display_right, ["RIGHT live", "SPACE burst match | q quit"])

        cv2.imshow(window_name, _display_pair(display_left, display_right, display_scale))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            busy_left = left.copy()
            busy_right = right.copy()
            _draw_text_lines(busy_left, ["Running burst match..."])
            _draw_text_lines(busy_right, ["Running burst match..."])
            cv2.imshow(window_name, _display_pair(busy_left, busy_right, display_scale))
            cv2.waitKey(1)
            last_scan_left, last_scan_right = _run_ai_debug_burst_match(cams, detector, coarse_mover)
            show_last_scan = True
        elif key == ord("l"):
            show_last_scan = False

    cv2.destroyWindow(window_name)


def _camera_cli_main():
    """Small CLI for checking camera IDs, resolution, FPS, and applied settings."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Stereo camera utility")
    parser.add_argument("--probe", action="store_true", help="Probe camera indices 0-9")
    parser.add_argument("--open", action="store_true", help="Open the configured stereo pair and show current properties")
    parser.add_argument("--view", action="store_true", help="Show live left/right preview until q is pressed")
    parser.add_argument(
        "--ai-debug",
        action="store_true",
        help="Show live preview; press Space to run burst-stable YOLO + QPoint scan and stereo matching",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.75,
        help="Scale for the side-by-side OpenCV preview window",
    )
    args = parser.parse_args()

    no_args = len(sys.argv) == 1
    if no_args:
        args.ai_debug = True

    if args.display_scale <= 0:
        parser.error("--display-scale must be greater than 0")

    def props(cap):
        return {
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "gain": cap.get(cv2.CAP_PROP_GAIN),
            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": cap.get(cv2.CAP_PROP_SATURATION),
        }

    if args.probe:
        for idx in range(10):
            cap = cv2.VideoCapture(idx, BACKEND)
            ok = cap.isOpened()
            print(f"camera {idx}: {'OPEN' if ok else 'not found'}")
            if ok:
                print(json.dumps(props(cap), indent=2))
            cap.release()
        return

    cams = StereoCameras()
    try:
        cams.open(start_recorder=not args.ai_debug)
        print("\nConfigured stereo cameras opened.")
        print("Left camera:", json.dumps(props(cams.left), indent=2))
        print("Right camera:", json.dumps(props(cams.right), indent=2))

        if args.ai_debug:
            _show_ai_debug_preview(cams, args.display_scale)
        elif args.view:
            _show_raw_preview(cams, args.display_scale)
    finally:
        cams.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _camera_cli_main()
