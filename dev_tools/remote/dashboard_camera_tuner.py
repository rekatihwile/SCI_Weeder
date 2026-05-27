"""Camera settings tuner routes for the remote dashboard."""

import json

from flask import jsonify, render_template, request

from config import BASE_DIR
from dashboard_camera import ensure_cameras
from dashboard_state import state
from dashboard_yolo import get_debug_defaults
from hardware import cameras as camera_hw


CAMERA_CONFIG_PATH = BASE_DIR / "params" / "hardware" / "camera_config.json"

_SIDE_DEFAULTS = {
    "auto_exposure": 0,
    "auto_wb": 1,
    "exposure": 300,
    "gain": 50,
    "brightness": 0,
    "contrast": 35,
    "saturation": 70,
    "white_balance": 2800,
}

_INT_RANGES = {
    "auto_exposure": (0, 1),
    "auto_wb": (0, 1),
    "exposure": (1, 1000),
    "gain": (0, 255),
    "brightness": (-64, 64),
    "contrast": (0, 100),
    "saturation": (0, 100),
    "white_balance": (2000, 6500),
}


def _coerce_int(value, default, min_value, max_value):
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _sanitize_side(raw, previous=None):
    raw = raw if isinstance(raw, dict) else {}
    out = dict(previous or {})

    for key, default in _SIDE_DEFAULTS.items():
        min_value, max_value = _INT_RANGES[key]
        out[key] = _coerce_int(raw.get(key, out.get(key, default)), default, min_value, max_value)

    return out


def _read_camera_config():
    data = {}
    if CAMERA_CONFIG_PATH.exists():
        try:
            data = json.loads(CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    left = data.get("left") if isinstance(data.get("left"), dict) else {}
    right = data.get("right") if isinstance(data.get("right"), dict) else {}
    return {
        "left": _sanitize_side(left, left),
        "right": _sanitize_side(right, right),
    }


def _settings_from_payload(payload):
    payload = payload if isinstance(payload, dict) else {}
    current = _read_camera_config()
    incoming = payload.get("settings", payload)
    incoming = incoming if isinstance(incoming, dict) else {}

    return {
        "left": _sanitize_side(incoming.get("left"), current.get("left")),
        "right": _sanitize_side(incoming.get("right"), current.get("right")),
    }


def _write_camera_config(settings):
    CAMERA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CAMERA_CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, indent=4) + "\n", encoding="utf-8")
    tmp.replace(CAMERA_CONFIG_PATH)

    # Keep camera resets in this dashboard process using the newly saved values.
    camera_hw.CAMERA_SETTINGS = settings


def _apply_live(settings):
    cam = ensure_cameras()
    camera_hw.apply_camera_settings(cam.left, settings.get("left"), cam.dev_paths.get("left"))
    camera_hw.apply_camera_settings(cam.right, settings.get("right"), cam.dev_paths.get("right"))


def register_camera_tuner_routes(app):
    @app.route("/camera_tuner")
    def camera_tuner_page():
        return render_template(
            "camera_tuner.html",
            survey_defaults=get_debug_defaults("survey"),
        )

    @app.route("/api/camera_tuner/config", methods=["GET"])
    def api_camera_tuner_config():
        try:
            return jsonify({
                "ok": True,
                "path": str(CAMERA_CONFIG_PATH),
                "settings": _read_camera_config(),
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    @app.route("/api/camera_tuner/apply", methods=["POST"])
    def api_camera_tuner_apply():
        try:
            settings = _settings_from_payload(request.get_json(silent=True) or {})
            with state.camera_lock:
                _apply_live(settings)
            return jsonify({
                "ok": True,
                "message": "Applied live camera settings.",
                "settings": settings,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    @app.route("/api/camera_tuner/save", methods=["POST"])
    def api_camera_tuner_save():
        try:
            settings = _settings_from_payload(request.get_json(silent=True) or {})
            _write_camera_config(settings)
            with state.camera_lock:
                _apply_live(settings)
            return jsonify({
                "ok": True,
                "path": str(CAMERA_CONFIG_PATH),
                "message": f"Saved camera settings to {CAMERA_CONFIG_PATH}",
                "settings": settings,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500
