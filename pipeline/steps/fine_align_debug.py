"""
pipeline/steps/fine_align_debug.py
------------------------------------
Modular Fine Align debug helpers.

Provides:
  - save/load cached survey/match/plan results
  - normalize pipeline planned targets into a flat plan JSON
  - coarse move to a cached target
  - run one Fine Align Re-ID attempt (sequential YOLO, no PD loop)
"""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
# change path to parent folder to import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from control.fine_align_reid import run_fine_align_reid

from config import (
    BASE_DIR,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    RECT_NPZ_PATH,
)


# =============================================================================
# Cache directory helpers  (Part A)
# =============================================================================

def cache_dir() -> Path:
    return BASE_DIR / "dev_tools" / "cache"


def latest_plan_path() -> Path:
    return cache_dir() / "latest_plan.json"


def fine_align_debug_dir() -> Path:
    d = cache_dir() / "fine_align_debug"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_latest_plan(plan: dict) -> Path:
    cache_dir().mkdir(parents=True, exist_ok=True)
    path = latest_plan_path()
    path.write_text(json.dumps(plan, indent=2))
    print(f"[FINE ALIGN DEBUG] Saved latest plan -> {path}")
    return path


def load_latest_plan() -> dict:
    path = latest_plan_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No cached plan found at {path}. "
            "Run bringup/07_match_plan_only.py first to generate one."
        )
    return json.loads(path.read_text())


def list_cached_targets(plan: dict) -> list:
    return plan["targets"]


def get_cached_target(plan: dict, target_id: int) -> dict:
    for t in plan["targets"]:
        if t["target_id"] == target_id:
            return t
    raise KeyError(f"No target with target_id={target_id} in plan.")


# =============================================================================
# Plan normalization — convert pipeline output to flat plan JSON  (Part B helper)
# =============================================================================

def _json_safe(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_safe(w) for k, w in v.items()}
    return v


def normalize_planned_targets_to_plan(
    planned_targets: list,
    survey_ref_xy: tuple,
    frame_mode: str = "raw",
) -> dict:
    """
    Convert pipeline planned_targets list into the flat plan JSON format.

    Each planned target from pipeline/steps/match_plan.py has:
      {
        "source_target": {"left_px": ..., "right_px": ..., "score": ..., ...},
        "target_xy_mm":  (x_mm, y_mm),
        ...
      }

    Returns a dict ready for save_latest_plan().
    """
    targets = []
    for i, pt in enumerate(planned_targets):
        src = pt.get("source_target", {})
        if isinstance(src, (list, tuple)) and len(src) >= 2:
            left_px  = list(_json_safe(src[0]))
            right_px = list(_json_safe(src[1]))
            score    = float(src[2]) if len(src) >= 3 else 1.0
            cls_id   = None
        else:
            left_px  = list(_json_safe(src.get("left_px",  [None, None])))
            right_px = list(_json_safe(src.get("right_px", [None, None])))
            score    = float(src.get("score", 1.0))
            cls_id   = src.get("cls", None)
            if cls_id is not None:
                cls_id = int(cls_id)

        xy = _json_safe(pt.get("target_xy_mm", [0.0, 0.0]))

        targets.append({
            "target_id":      i,
            "coarse_x_mm":    float(xy[0]),
            "coarse_y_mm":    float(xy[1]),
            "left_px_survey":  left_px,
            "right_px_survey": right_px,
            "match_score":    score,
            "class_id":       cls_id,
            "source":         "survey_match_plan",
            "raw":            _json_safe(pt),
        })

    return {
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "frame_mode":     frame_mode,
        "survey_ref_xy":  list(_json_safe(survey_ref_xy)),
        "targets":        targets,
    }


# =============================================================================
# Standalone stereo rectification  (no dashboard_state dependency)
# =============================================================================

_rect_maps_cache: dict = {}


def _load_rect_maps() -> dict:
    """Load stereo rectification maps from RECT_NPZ_PATH. Cached after first load."""
    if _rect_maps_cache:
        return _rect_maps_cache

    data = np.load(str(RECT_NPZ_PATH))

    def _find(candidates):
        for k in candidates:
            if k in data:
                return k
        return None

    lx = _find(["map1L", "left_map_x", "map1_left", "left_map1", "mapLx", "mapxL", "lmapx"])
    ly = _find(["map2L", "left_map_y", "map2_left", "left_map2", "mapLy", "mapyL", "lmapy"])
    rx = _find(["map1R", "right_map_x", "map1_right", "right_map1", "mapRx", "mapxR", "rmapx"])
    ry = _find(["map2R", "right_map_y", "map2_right", "right_map2", "mapRy", "mapyR", "rmapy"])

    if None in (lx, ly, rx, ry):
        raise RuntimeError(
            f"Could not find rectification map keys in {RECT_NPZ_PATH}. "
            f"Available keys: {list(data.keys())}"
        )

    _rect_maps_cache["left_map_x"]  = data[lx]
    _rect_maps_cache["left_map_y"]  = data[ly]
    _rect_maps_cache["right_map_x"] = data[rx]
    _rect_maps_cache["right_map_y"] = data[ry]
    return _rect_maps_cache


def _rectify_frame_pair(left_frame, right_frame):
    """Apply stereo rectification maps. Returns (left_rect, right_rect)."""
    maps = _load_rect_maps()
    left_rect = cv2.remap(
        left_frame,
        maps["left_map_x"],
        maps["left_map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rect = cv2.remap(
        right_frame,
        maps["right_map_x"],
        maps["right_map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left_rect, right_rect


# =============================================================================
# Re-ID once helper  (Part C)
# =============================================================================

def run_reid_once(
    cameras,
    detector,
    target: dict,
    use_rectified: bool = True,
    crop_w: int = 384,
    crop_h: int = 384,
    burst_count: int = 5,
    min_hits: int = 1,
    cluster_radius_px=None,
    point_mode: str = "box_center",
    expected_cls=None,
    conf_override=None,
    y_gate_px: float = 5,
    min_disp_px: float = 10,
    max_disp_px: float = 500,
    imgsz=None,
) -> dict:
    """
    Thin wrapper around control.fine_align_reid.run_fine_align_reid().
    Saves debug artifacts and returns JSON-safe fields for bringup/dashboard.
    """
    t_start = time.perf_counter()
    target_id = target.get("target_id", None)
    debug_dir = fine_align_debug_dir()
    timing: dict = {}
    debug_files: dict = {}

    try:
        reid_result = run_fine_align_reid(
            cameras=cameras,
            detector=detector,
            target=target,
            crop_w=crop_w,
            crop_h=crop_h,
            burst_count=burst_count,
            min_hits=min_hits,
            cluster_radius_px=cluster_radius_px,
            point_mode=point_mode,
            class_filter=expected_cls,
            conf_override=conf_override,
            imgsz=imgsz,
            use_rectified=use_rectified,
            y_gate_px=y_gate_px,
            min_disp_px=min_disp_px,
            max_disp_px=max_disp_px,
            return_debug=True,
        )

        timing = dict(reid_result.get("timing", {}))
        frame_mode = reid_result.get("frame_mode", "rectified" if use_rectified else "raw")
        crop_bounds = reid_result.get("crop", {})
        left_dets = reid_result.get("left_detections", [])
        right_dets = reid_result.get("right_detections", [])
        candidates = reid_result.get("matches", [])
        chosen = reid_result.get("chosen")
        debug_frames = reid_result.get("debug_frames") or {}

        # ------------------------------------------------------------------ #
        # 6. Save debug artifacts to dev_tools/cache/fine_align_debug         #
        # ------------------------------------------------------------------ #
        def _save(img, name):
            p = debug_dir / name
            cv2.imwrite(str(p), img)
            return str(p)

        if debug_frames.get("left_full") is not None:
            debug_files["full_left"] = _save(debug_frames["left_full"], "latest_full_left.jpg")
        if debug_frames.get("right_full") is not None:
            debug_files["full_right"] = _save(debug_frames["right_full"], "latest_full_right.jpg")
        if debug_frames.get("left_crop") is not None:
            debug_files["crop_left"] = _save(debug_frames["left_crop"], "latest_crop_left.jpg")
        if debug_frames.get("right_crop") is not None:
            debug_files["crop_right"] = _save(debug_frames["right_crop"], "latest_crop_right.jpg")
        if debug_frames.get("left_overlay") is not None:
            debug_files["overlay_left"] = _save(debug_frames["left_overlay"], "latest_reid_overlay_left.jpg")
        if debug_frames.get("right_overlay") is not None:
            debug_files["overlay_right"] = _save(debug_frames["right_overlay"], "latest_reid_overlay_right.jpg")

        # ------------------------------------------------------------------ #
        # 7. Build and return result dict                                      #
        # ------------------------------------------------------------------ #
        timing["total_s"] = float(timing.get("total_s", round(time.perf_counter() - t_start, 6)))
        result = {
            "ok":               bool(reid_result.get("ok", False)),
            "target_id":        target_id,
            "frame_mode":       frame_mode,
            "crop":             crop_bounds,
            "timing":           timing,
            "left_detections":  left_dets,
            "right_detections": right_dets,
            "matches":          candidates,
            "chosen":           chosen,
            "debug_files":      debug_files,
            "error":            reid_result.get("error"),
        }

        result_path = debug_dir / "latest_reid_result.json"
        result_path.write_text(json.dumps(result, indent=2))
        debug_files["result_json"] = str(result_path)

        print(
            f"[FINE ALIGN RE-ID] Done.  total={float(timing['total_s']):.3f}s  "
            f"L={len(left_dets)} R={len(right_dets)}  "
            f"matches={len(candidates)}  chosen={'YES' if chosen else 'NONE'}"
        )
        return result

    except Exception as exc:
        import traceback
        err_str = traceback.format_exc()
        print(f"[FINE ALIGN RE-ID] ERROR: {exc}")
        timing["total_s"] = round(time.perf_counter() - t_start, 6)

        return {
            "ok":               False,
            "target_id":        target_id,
            "frame_mode":       "rectified" if use_rectified else "raw",
            "crop":             {},
            "timing":           timing,
            "left_detections":  [],
            "right_detections": [],
            "matches":          [],
            "chosen":           None,
            "debug_files":      debug_files,
            "error":            err_str,
        }


# =============================================================================
# Coarse move helper  (Part D)
# =============================================================================

def move_to_cached_target(gantry, target: dict, feed=None) -> dict:
    """Move gantry to the coarse position stored in a cached target dict."""
    x_mm      = float(target["coarse_x_mm"])
    y_mm      = float(target["coarse_y_mm"])
    target_id = target.get("target_id", None)

    if feed is not None:
        gantry.move_absolute(x_mm, y_mm, feed=float(feed))
    else:
        gantry.move_absolute(x_mm, y_mm)

    return {
        "ok":              True,
        "target_id":       target_id,
        "commanded_x_mm":  x_mm,
        "commanded_y_mm":  y_mm,
        "feed":            feed,
    }
