"""Survey-position stereo photo burst routes for the remote dashboard."""

import math
import re
import time
from pathlib import Path

import cv2
from flask import jsonify, render_template, request

from config import BASE_DIR, GRBL_PORT, SURVEY_BURST_COUNT, SURVEY_POS_X, SURVEY_POS_Y
from config.survey_params import resolve_burst_count
from dashboard_camera import ensure_cameras, parse_bool
from dashboard_camera_tuner import _apply_live, _settings_from_payload
from dashboard_gantry import ensure_gantry
from dashboard_images import b64_img
from dashboard_rectify import maybe_rectify_pair
from dashboard_state import state


SURVEY_PICS_DIR = BASE_DIR / "SurveyPics"
PASS_THRESHOLD_MM = 5.0
DEFAULT_WARMUP_FRAMES = 8


def _coerce_int(value, default, min_value, max_value):
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _coerce_float(value, default, min_value, max_value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _next_start_index(save_dir: Path) -> int:
    max_idx = -1
    pattern = re.compile(r"^left_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    for path in save_dir.glob("left_*"):
        match = pattern.match(path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _save_burst(cameras, count, warmup, interval_s, save_dir, use_rectification=False):
    save_dir.mkdir(parents=True, exist_ok=True)

    warmup_none_count = 0
    for _ in range(warmup):
        left, right = cameras.read_pair()
        if left is None or right is None:
            warmup_none_count += 1

    idx = _next_start_index(save_dir)
    saved = []
    last_left = None
    last_right = None
    max_attempts = count * 3
    started = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        if len(saved) >= count:
            break

        left, right = cameras.read_pair()
        if left is None or right is None:
            continue
        left, right, frame_mode = maybe_rectify_pair(left, right, use_rectification)

        left_path = save_dir / f"left_{idx:04d}.jpg"
        right_path = save_dir / f"right_{idx:04d}.jpg"

        left_ok = cv2.imwrite(str(left_path), left)
        right_ok = cv2.imwrite(str(right_path), right)
        if not left_ok or not right_ok:
            raise RuntimeError(f"Failed to write stereo pair {idx} to {save_dir}")

        saved.append({
            "index": idx,
            "left": str(left_path),
            "right": str(right_path),
            "left_name": left_path.name,
            "right_name": right_path.name,
            "frame_mode": frame_mode,
        })
        last_left = left
        last_right = right
        idx += 1

        if interval_s > 0 and len(saved) < count:
            time.sleep(interval_s)

    return {
        "saved": saved,
        "saved_count": len(saved),
        "attempts": max_attempts,
        "warmup_none_count": warmup_none_count,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "frame_mode": "rectified" if use_rectification else "raw",
        "last_left_image": b64_img(last_left) if last_left is not None else None,
        "last_right_image": b64_img(last_right) if last_right is not None else None,
    }


def register_survey_photo_routes(app):
    @app.route("/survey_photos")
    def survey_photos_page():
        return render_template(
            "survey_photos.html",
            grbl_port=GRBL_PORT,
            survey_pos_x=SURVEY_POS_X,
            survey_pos_y=SURVEY_POS_Y,
            default_count=resolve_burst_count(SURVEY_BURST_COUNT),
            default_warmup=DEFAULT_WARMUP_FRAMES,
            survey_pics_dir=str(SURVEY_PICS_DIR),
        )

    @app.route("/api/survey_photos/capture", methods=["POST"])
    def api_survey_photos_capture():
        try:
            payload = request.get_json(silent=True) or {}
            count = _coerce_int(
                payload.get("count"),
                resolve_burst_count(SURVEY_BURST_COUNT),
                1,
                100,
            )
            warmup = _coerce_int(payload.get("warmup"), DEFAULT_WARMUP_FRAMES, 0, 100)
            interval_s = _coerce_float(payload.get("interval"), 0.0, 0.0, 10.0)
            use_rectification = parse_bool(payload.get("rectified", False))
            settings = _settings_from_payload(payload)

            with state.gantry_lock:
                gantry = ensure_gantry()
                pos_before = gantry.get_position()
                gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
                pos_after = gantry.get_position()

            if pos_after is not None:
                dx = pos_after["x"] - SURVEY_POS_X
                dy = pos_after["y"] - SURVEY_POS_Y
                dist = math.hypot(dx, dy)
            else:
                dist = None

            with state.camera_lock:
                cameras = ensure_cameras()
                _apply_live(settings)
                burst = _save_burst(
                    cameras,
                    count,
                    warmup,
                    interval_s,
                    SURVEY_PICS_DIR,
                    use_rectification=use_rectification,
                )

            ok = burst["saved_count"] == count and (dist is not None and dist <= PASS_THRESHOLD_MM)
            return jsonify({
                "ok": ok,
                "save_dir": str(SURVEY_PICS_DIR),
                "requested_count": count,
                "warmup": warmup,
                "interval_s": interval_s,
                "rectified": use_rectification,
                "gantry": {
                    "port": GRBL_PORT,
                    "survey_target": {"x": SURVEY_POS_X, "y": SURVEY_POS_Y},
                    "position_before": pos_before,
                    "position_after": pos_after,
                    "distance_from_target_mm": round(dist, 3) if dist is not None else None,
                    "pass_threshold_mm": PASS_THRESHOLD_MM,
                },
                **burst,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500
