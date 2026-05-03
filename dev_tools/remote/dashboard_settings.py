"""Persistent local settings for dashboard pages.

This stores debug/dev UI state only. It does not write config.py or params/*.json.
"""

import json
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
