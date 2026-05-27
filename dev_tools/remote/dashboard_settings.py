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

from config import BASE_DIR, AVOID_CLASSES, AI_CLASS_CONFIDENCE, MODEL_MAP


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

_CONFIG_PY_PATH        = BASE_DIR / "config" / "survey_params.py"
_VISION_CONFIG_PY_PATH = BASE_DIR / "config" / "vision.py"
_FINE_ALIGN_CONFIG_PATH = BASE_DIR / "config" / "alignment_params.py"
_config_write_lock = Lock()

# Maps suggested_config keys → (config_file_path, variable_name_in_file).
# Only keys listed here are ever touched; all other content is preserved.
_SURVEY_CONFIG_MAP = {
    # written to config/survey_params.py
    "BURST_COUNT":         (_CONFIG_PY_PATH,        "SURVEY_BURST_COUNT"),
    "MIN_HITS":            (_CONFIG_PY_PATH,         "SURVEY_MIN_HITS"),
    "POINT_MODE":          (_CONFIG_PY_PATH,         "SURVEY_POINT_MODE"),
    "YOLO_IMGSZ_USED":     (_CONFIG_PY_PATH,         "SURVEY_YOLO_IMGSZ"),
    "CROP_MODE":           (_CONFIG_PY_PATH,         "SURVEY_CROP_MODE"),
    "CROP_W":              (_CONFIG_PY_PATH,         "SURVEY_CROP_W"),
    "CROP_H":              (_CONFIG_PY_PATH,         "SURVEY_CROP_H"),
    "LEFT_OFFSET_X":       (_CONFIG_PY_PATH,         "SURVEY_LEFT_OFFSET_X"),
    "LEFT_OFFSET_Y":       (_CONFIG_PY_PATH,         "SURVEY_LEFT_OFFSET_Y"),
    "RIGHT_OFFSET_X":      (_CONFIG_PY_PATH,         "SURVEY_RIGHT_OFFSET_X"),
    "RIGHT_OFFSET_Y":      (_CONFIG_PY_PATH,         "SURVEY_RIGHT_OFFSET_Y"),
    # written to config/vision.py
    "TARGET_CLASSES":      (_VISION_CONFIG_PY_PATH,  "TARGET_CLASSES"),
    "AVOID_CLASSES":       (_VISION_CONFIG_PY_PATH,  "AVOID_CLASSES"),
    "CONFIDENCE_OVERRIDE": (_VISION_CONFIG_PY_PATH,  "AI_CONFIDENCE"),
    "AI_CLASS_CONFIDENCE": (_VISION_CONFIG_PY_PATH,  "AI_CLASS_CONFIDENCE"),
    "DEFAULT_MODEL":       (_VISION_CONFIG_PY_PATH,  "DEFAULT_MODEL"),
    "DEFAULT_MODEL_PT":    (_VISION_CONFIG_PY_PATH,  "DEFAULT_MODEL_PT"),
    "DEFAULT_MODEL_ENGINE": (_VISION_CONFIG_PY_PATH, "DEFAULT_MODEL_ENGINE"),
    "YOLO_BACKEND":        (_VISION_CONFIG_PY_PATH,  "YOLO_BACKEND"),
    "USE_TENSORRT_ENGINE": (_VISION_CONFIG_PY_PATH,  "USE_TENSORRT_ENGINE"),
}


_FINE_ALIGN_CONFIG_MAP = {
    "burst_count":  (_FINE_ALIGN_CONFIG_PATH, "FINE_ALIGN_BURST_COUNT"),
    "min_hits":     (_FINE_ALIGN_CONFIG_PATH, "FINE_ALIGN_MIN_HITS"),
    "crop_half_px": (_FINE_ALIGN_CONFIG_PATH, "FINE_ALIGN_REID_CROP_HALF_PX"),
    "y_gate_px":    (_FINE_ALIGN_CONFIG_PATH, "FINE_ALIGN_REID_MAX_Y_DIFF_PX"),
    "min_disp_px":  (_FINE_ALIGN_CONFIG_PATH, "FINE_ALIGN_REID_MIN_DISPARITY_PX"),
    "max_disp_px":  (_FINE_ALIGN_CONFIG_PATH, "FINE_ALIGN_REID_MAX_DISPARITY_PX"),
    "point_mode":   (_CONFIG_PY_PATH,          "FINE_ALIGN_REID_POINT_MODE"),
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


def _write_config_vars(file_path: Path, var_updates: dict) -> list:
    """Rewrite specific variable assignments in a config .py file in-place.

    var_updates: {var_name: new_value}
    Returns list of {"var": name, "value": repr} for each var actually updated.
    """
    content = file_path.read_text()
    updated = []
    for cfg_var, new_val in var_updates.items():
        val_repr = _fmt_config_val(new_val)
        pattern = re.compile(
            r"^(" + re.escape(cfg_var) + r"\s*=\s*).*$",
            re.MULTILINE,
        )
        new_content, n = pattern.subn(lambda m, vr=val_repr: m.group(1) + vr, content)
        if n == 0:
            continue
        content = new_content
        updated.append({"var": cfg_var, "value": val_repr})

    tmp = file_path.with_suffix(".py.tmp")
    tmp.write_text(content)
    tmp.replace(file_path)
    return updated


def save_survey_config_to_config_py(suggested_config: dict) -> list:
    """Write survey tuning values from suggested_config into config files in-place.

    Entries in _SURVEY_CONFIG_MAP map suggested_config keys to (file, var_name).
    Only the listed variables are touched; all other file content is preserved.

    Returns a list of {"var": name, "value": repr} dicts for each updated var.
    """
    with _config_write_lock:
        payload = dict(suggested_config or {})

        if "AVOID_CONFIDENCE_OVERRIDE" in payload:
            avoid_conf = payload.get("AVOID_CONFIDENCE_OVERRIDE")
            avoid_classes = payload.get("AVOID_CLASSES", AVOID_CLASSES)
            class_conf = {
                int(k): float(v)
                for k, v in dict(AI_CLASS_CONFIDENCE or {}).items()
            }
            for cls_id in list(avoid_classes or []):
                cls_id = int(cls_id)
                if avoid_conf is None:
                    class_conf.pop(cls_id, None)
                else:
                    class_conf[cls_id] = float(avoid_conf)
            payload["AI_CLASS_CONFIDENCE"] = class_conf or None

        if "SEGMENTATION_MODEL" in payload:
            model_choice = str(payload.get("SEGMENTATION_MODEL") or "").strip()
            if model_choice and model_choice != "__config__":
                resolved_model = MODEL_MAP.get(model_choice, model_choice)
                model_suffix = Path(str(resolved_model)).suffix.lower()

                payload["DEFAULT_MODEL"] = model_choice if model_choice in MODEL_MAP else resolved_model
                if model_suffix == ".engine":
                    payload["DEFAULT_MODEL_ENGINE"] = resolved_model
                    payload["YOLO_BACKEND"] = "engine"
                    payload["USE_TENSORRT_ENGINE"] = True
                else:
                    payload["DEFAULT_MODEL_PT"] = resolved_model
                    payload["YOLO_BACKEND"] = "pt"
                    payload["USE_TENSORRT_ENGINE"] = False

        # Group updates by target file.
        by_file: dict[Path, dict] = {}
        for sc_key, (cfg_path, cfg_var) in _SURVEY_CONFIG_MAP.items():
            if sc_key not in payload:
                continue
            by_file.setdefault(cfg_path, {})[cfg_var] = payload[sc_key]

        updated = []
        for file_path, var_updates in by_file.items():
            updated.extend(_write_config_vars(file_path, var_updates))

        return updated


def save_fine_align_config_to_config_py(config_values: dict) -> list:
    """Write fine-align Re-ID tuning values into alignment_params.py and survey_params.py in-place."""
    with _config_write_lock:
        by_file: dict[Path, dict] = {}
        for key, (cfg_path, cfg_var) in _FINE_ALIGN_CONFIG_MAP.items():
            if key not in config_values:
                continue
            by_file.setdefault(cfg_path, {})[cfg_var] = config_values[key]

        updated = []
        for file_path, var_updates in by_file.items():
            updated.extend(_write_config_vars(file_path, var_updates))

        return updated
