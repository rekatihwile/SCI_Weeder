"""Persistent local settings for dashboard pages.

Dashboard UI state is stored in dashboard_settings.json (auto-saved on every change).
Survey runtime parameters are written to config.py only when the user presses
"Save Config Settings" — nothing is overwritten automatically.
"""

import json
import re
from copy import deepcopy
from pathlib import Path
from threading import Lock

from config import BASE_DIR


SETTINGS_PATH = BASE_DIR / "dev_tools" / "cache" / "dashboard_settings.json"

_DEFAULT_SETTINGS = {
    "survey": {},
    "fine": {},
    "fine_align": {},
    "match": {},
    "workspace3d": {},
    "gantry": {},
    "camera": {},
}

_lock = Lock()


def _default_settings():
    return deepcopy(_DEFAULT_SETTINGS)


def _normalize_settings_dict(data):
    out = _default_settings()
    if isinstance(data, dict):
        for page_name, page_settings in data.items():
            if page_name in out and isinstance(page_settings, dict):
                out[page_name] = dict(page_settings)
            elif isinstance(page_name, str) and isinstance(page_settings, dict):
                out[page_name] = dict(page_settings)
    return out


def load_dashboard_settings() -> dict:
    """Load persisted dashboard settings from disk, creating defaults if missing."""
    with _lock:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not SETTINGS_PATH.exists():
            defaults = _default_settings()
            SETTINGS_PATH.write_text(json.dumps(defaults, indent=2))
            return defaults

        try:
            raw = SETTINGS_PATH.read_text()
            data = json.loads(raw)
        except Exception:
            data = _default_settings()
            SETTINGS_PATH.write_text(json.dumps(data, indent=2))
            return data

        normalized = _normalize_settings_dict(data)
        if normalized != data:
            SETTINGS_PATH.write_text(json.dumps(normalized, indent=2))
        return normalized


def save_dashboard_settings(settings: dict) -> None:
    """Persist the full dashboard settings object to disk."""
    with _lock:
        normalized = _normalize_settings_dict(settings)
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(normalized, indent=2))


def get_page_settings(page_name: str) -> dict:
    """Return settings for one page, always a dict."""
    key = str(page_name or "").strip()
    all_settings = load_dashboard_settings()
    page = all_settings.get(key)
    return dict(page) if isinstance(page, dict) else {}


def update_page_settings(page_name: str, values: dict) -> dict:
    """Merge incoming values into one page settings and persist.

    Returns the updated page settings.
    """
    key = str(page_name or "").strip()
    incoming = dict(values or {})

    all_settings = load_dashboard_settings()
    current = all_settings.get(key)
    if not isinstance(current, dict):
        current = {}

    current.update(incoming)
    all_settings[key] = current
    save_dashboard_settings(all_settings)
    return dict(current)


# =============================================================================
# config.py writer — "Save Config Settings"
# =============================================================================

_CONFIG_PY_PATH = BASE_DIR / "config" / "survey_params.py"
_config_write_lock = Lock()

# Maps suggested_config keys (from run_debug_scan) → config.py variable names.
# Only keys listed here are ever touched; everything else in config.py is left alone.
_SURVEY_CONFIG_MAP = {
    "BURST_COUNT":      "SURVEY_BURST_COUNT",
    "MIN_HITS":         "SURVEY_MIN_HITS",
    "POINT_MODE":       "SURVEY_POINT_MODE",
    "TARGET_CLASSES":   "SURVEY_TARGET_CLASSES",
    "YOLO_IMGSZ_USED":  "SURVEY_YOLO_IMGSZ",
    "CONFIDENCE_OVERRIDE": "SURVEY_CONFIDENCE_OVERRIDE",
    "CROP_MODE":        "SURVEY_CROP_MODE",
    "CROP_W":           "SURVEY_CROP_W",
    "CROP_H":           "SURVEY_CROP_H",
    "LEFT_OFFSET_X":    "SURVEY_LEFT_OFFSET_X",
    "LEFT_OFFSET_Y":    "SURVEY_LEFT_OFFSET_Y",
    "RIGHT_OFFSET_X":   "SURVEY_RIGHT_OFFSET_X",
    "RIGHT_OFFSET_Y":   "SURVEY_RIGHT_OFFSET_Y",
}


def _fmt_config_val(v) -> str:
    """Format a Python value as a config.py literal (single line)."""
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_config_val(x) for x in v) + "]"
    if isinstance(v, dict):
        pairs = ", ".join(f'"{k}": {_fmt_config_val(vv)}' for k, vv in v.items())
        return "{" + pairs + "}"
    return repr(v)


def save_survey_config_to_config_py(suggested_config: dict) -> list:
    """Write survey tuning values from suggested_config into config.py in-place.

    Only the variables listed in _SURVEY_CONFIG_MAP are touched — all other
    config.py content (comments, formatting, unrelated vars) is preserved.

    Returns a list of {"var": name, "value": repr} dicts for each updated var.
    """
    with _config_write_lock:
        content = _CONFIG_PY_PATH.read_text()
        updated = []

        for sc_key, cfg_var in _SURVEY_CONFIG_MAP.items():
            if sc_key not in suggested_config:
                continue

            new_val = suggested_config[sc_key]
            val_repr = _fmt_config_val(new_val)

            # Match the exact assignment line: VAR_NAME = <anything to end of line>
            pattern = re.compile(
                r"^(" + re.escape(cfg_var) + r"\s*=\s*).*$",
                re.MULTILINE,
            )
            new_content, n = pattern.subn(lambda m, vr=val_repr: m.group(1) + vr, content)

            if n == 0:
                continue  # var not found — skip rather than append a stray line

            content = new_content
            updated.append({"var": cfg_var, "value": val_repr})

        # Atomic write: write to .tmp then rename so a crash mid-write can't corrupt config.py
        tmp = _CONFIG_PY_PATH.with_suffix(".py.tmp")
        tmp.write_text(content)
        tmp.replace(_CONFIG_PY_PATH)

        return updated
