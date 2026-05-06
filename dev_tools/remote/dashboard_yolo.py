"""YOLO detection helpers: ensure detector, parameter parsers, debug scan, cached match."""

import json
import time

from dashboard_state import (
    state,
    SURVEY_BURST_COUNT,
    SURVEY_MIN_HITS,
    SURVEY_YOLO_IMGSZ,
    SURVEY_TARGET_CLASSES,
    SURVEY_AVOID_CLASSES,
    SURVEY_POINT_MODE,
    AVOID_CLASSES,
)
from dashboard_images import b64_img, draw_crop, draw_detections, draw_matches
from dashboard_camera import parse_bool, crop_bounds_for_side, ensure_cameras
from dashboard_rectify import maybe_rectify_pair

# Patch broken Jetson torchvision NMS before importing Ultralytics/YOLO.
# This matches the working bringup scripts.
import importlib.util
from pathlib import Path

import cv2

_NMS_PATCH_PATH = Path(__file__).resolve().parents[2] / "bringup" / "_nms_patch.py"

if _NMS_PATCH_PATH.exists():
    _spec = importlib.util.spec_from_file_location("_nms_patch", _NMS_PATCH_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
else:
    print(f"[WARN] NMS patch not found: {_NMS_PATCH_PATH}")
    
from vision.detectors.ai_detector import AIDetector
from vision.matching import match_points

_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
_SURVEY_CACHE_DIR = _CACHE_DIR / "survey"

# =============================================================================
# Parameter parsers
# =============================================================================

def snap32(value):
    value = int(value)
    return max(32, int(round(value / 32.0) * 32))


def parse_classes(value):
    if value is None:
        return SURVEY_TARGET_CLASSES
    text = str(value).strip().lower()
    if text in ("", "all", "none", "null"):
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_avoid_classes(value):
    """Parse avoid_classes param. Empty/none = use detector's configured avoid_classes."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "none", "null"):
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_conf(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


# =============================================================================
# YOLO inference parameters
# =============================================================================

def yolo_imgsz_from_params(params):
    mode = params.get("imgsz_mode", "snap_crop")
    if mode == "config":
        return SURVEY_YOLO_IMGSZ
    crop_w = int(params.get("crop_w", 704))
    crop_h = int(params.get("crop_h", 704))
    return (snap32(crop_h), snap32(crop_w))


def get_debug_defaults(kind):
    if kind == "fine":
        return {
            "title": "Fine Align / Re-ID YOLO Debug",
            "default_mode": "center",
            "default_crop_w": 384,
            "default_crop_h": 384,
            "default_burst": 5,
            "default_min_hits": 1,
        }
    return {
        "title": "Survey YOLO Debug",
        "default_mode": "center_facing",
        "default_crop_w": 704,
        "default_crop_h": 704,
        "default_burst": SURVEY_BURST_COUNT,
        "default_min_hits": SURVEY_MIN_HITS,
    }


def translate_detections_to_full(dets, crop):
    x0, y0 = crop["x0"], crop["y0"]
    out = []
    for d in dets:
        px, py = d["point"]
        x1, y1, x2, y2 = d["box"]
        out.append({
            **d,
            "point": (int(px + x0), int(py + y0)),
            "box": (
                float(x1 + x0),
                float(y1 + y0),
                float(x2 + x0),
                float(y2 + y0),
            ),
        })
    return out


# =============================================================================
# Detector lifecycle
# =============================================================================

def ensure_detector():
    if state.detector is None:
        state.detector = AIDetector()
        state.detector.warmup()
    return state.detector


def get_class_names(det=None):
    if det is None:
        if state.detector is None:
            return {}
        det = state.detector
    names = getattr(det.cv_left.yolo, "names", {})
    return {str(k): v for k, v in dict(names).items()}


# =============================================================================
# Debug scan
# =============================================================================

def run_debug_scan(params):
    cam = ensure_cameras()
    det = ensure_detector()

    use_rectification = parse_bool(params.get("rectified", False))
    burst_count = int(params.get("burst_count", SURVEY_BURST_COUNT))
    min_hits = int(params.get("min_hits", SURVEY_MIN_HITS))
    point_mode = params.get("point_mode", SURVEY_POINT_MODE or "box_center")
    class_filter = parse_classes(params.get("classes", ""))
    avoid_filter = parse_avoid_classes(params.get("avoid_classes", ""))
    conf_override = parse_conf(params.get("conf", ""))
    imgsz = yolo_imgsz_from_params(params)

    left_crop_box = crop_bounds_for_side(params, "left")
    right_crop_box = crop_bounds_for_side(params, "right")

    left_frames = []
    right_frames = []
    frame_mode = "raw"

    t_capture = time.perf_counter()
    for _ in range(burst_count):
        fL, fR = cam.read_pair()
        fL, fR, frame_mode = maybe_rectify_pair(fL, fR, use_rectification)
        left_frames.append(fL)
        right_frames.append(fR)
    capture_s = time.perf_counter() - t_capture

    lx0, ly0, lx1, ly1 = left_crop_box["x0"], left_crop_box["y0"], left_crop_box["x1"], left_crop_box["y1"]
    rx0, ry0, rx1, ry1 = right_crop_box["x0"], right_crop_box["y0"], right_crop_box["x1"], right_crop_box["y1"]

    left_crop_frames = [f[ly0:ly1, lx0:lx1] for f in left_frames]
    right_crop_frames = [f[ry0:ry1, rx0:rx1] for f in right_frames]

    old_left_conf  = det.cv_left.conf
    old_right_conf = det.cv_right.conf
    old_left_avoid  = det.cv_left.avoid_classes
    old_right_avoid = det.cv_right.avoid_classes

    if conf_override is not None:
        det.cv_left.conf  = conf_override
        det.cv_right.conf = conf_override
    if avoid_filter is not None:
        det.cv_left.avoid_classes  = avoid_filter
        det.cv_right.avoid_classes = avoid_filter

    try:
        t_yolo = time.perf_counter()

        stable_left = det.cv_left.return_burst_stable(
            left_crop_frames,
            min_stable_views=min_hits,
            classes_override=class_filter,
            debug_label="[WEB LEFT]",
            imgsz=imgsz,
            heatmap_final=(point_mode != "box_center"),
            point_mode=point_mode,
        )

        stable_right = det.cv_right.return_burst_stable(
            right_crop_frames,
            min_stable_views=min_hits,
            classes_override=class_filter,
            debug_label="[WEB RIGHT]",
            imgsz=imgsz,
            heatmap_final=(point_mode != "box_center"),
            point_mode=point_mode,
        )

        yolo_s = time.perf_counter() - t_yolo

    finally:
        det.cv_left.conf  = old_left_conf
        det.cv_right.conf = old_right_conf
        det.cv_left.avoid_classes  = old_left_avoid
        det.cv_right.avoid_classes = old_right_avoid

    stable_left_full = translate_detections_to_full(stable_left, left_crop_box)
    stable_right_full = translate_detections_to_full(stable_right, right_crop_box)

    left_base = draw_crop(left_frames[-1], left_crop_box, "left crop")
    right_base = draw_crop(right_frames[-1], right_crop_box, "right crop")

    left_overlay = draw_detections(left_base, stable_left_full, color=(0, 0, 255))
    right_overlay = draw_detections(right_base, stable_right_full, color=(0, 0, 255))

    class_names = get_class_names(det)

    state.last_scan = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_mode": frame_mode,
        "rectified": use_rectification,
        "left_frame": left_frames[-1],
        "right_frame": right_frames[-1],
        "left_detections": stable_left_full,
        "right_detections": stable_right_full,
        "left_crop": left_crop_box,
        "right_crop": right_crop_box,
        "params": dict(params),
        "class_names": class_names,
    }

    # Effective avoid list used in this scan (explicit override or detector default).
    effective_avoid = avoid_filter if avoid_filter is not None else old_left_avoid

    return {
        "left_count": len(stable_left_full),
        "right_count": len(stable_right_full),
        "capture_s": round(capture_s, 3),
        "yolo_s": round(yolo_s, 3),
        "left_crop": left_crop_box,
        "right_crop": right_crop_box,
        "left_image": b64_img(left_overlay),
        "right_image": b64_img(right_overlay),
        "class_names": class_names,
        "requested_classes": class_filter,
        "avoid_classes": effective_avoid,
        "confidence_override": conf_override,
        "imgsz": imgsz,
        "frame_mode": frame_mode,
        "cached": True,
        "suggested_config": {
            "BURST_COUNT": burst_count,
            "MIN_HITS": min_hits,
            "POINT_MODE": point_mode,
            "TARGET_CLASSES": class_filter,
            "AVOID_CLASSES": effective_avoid,
            "CONFIDENCE_OVERRIDE": conf_override,
            "RECTIFIED_DEBUG": use_rectification,
            "CROP_MODE": params.get("mode", "center"),
            "CROP_W": int(params.get("crop_w", 704)),
            "CROP_H": int(params.get("crop_h", 704)),
            "LEFT_OFFSET_X": int(params.get("left_offset_x", 0)),
            "LEFT_OFFSET_Y": int(params.get("left_offset_y", 0)),
            "RIGHT_OFFSET_X": int(params.get("right_offset_x", 0)),
            "RIGHT_OFFSET_Y": int(params.get("right_offset_y", 0)),
            "LEFT_CROP_FULL_FRAME": left_crop_box,
            "RIGHT_CROP_FULL_FRAME": right_crop_box,
            "YOLO_IMGSZ_USED": imgsz,
        },
    }


# =============================================================================
# Cached match
# =============================================================================

from pipeline.steps.match_plan import normalize_match


def _fallback_box(point, half=12.0):
    x, y = map(float, point)
    return (x - half, y - half, x + half, y + half)


def _det_from_cached_match(src, side):
    px_key = f"{side}_px"
    box_key = f"{side}_box"
    cls_key = f"{side}_cls"
    conf_key = f"{side}_conf"
    px_rect_key = f"{side}_px_rect"
    box_rect_key = f"{side}_box_rect"

    point = tuple(src[px_key])
    det = {
        "point": point,
        "box": tuple(src.get(box_key) or _fallback_box(point)),
        "cls": src.get(cls_key),
        "conf": src.get(conf_key),
    }
    if px_rect_key in src:
        det["point_rectified"] = tuple(src[px_rect_key])
    if box_rect_key in src:
        det["box_rectified"] = tuple(src[box_rect_key])
    return det


def _load_disk_cached_scan():
    plan_path = _CACHE_DIR / "latest_plan.json"
    left_path = _CACHE_DIR / "fine_align_debug" / "latest_full_left.jpg"
    right_path = _CACHE_DIR / "fine_align_debug" / "latest_full_right.jpg"

    if not plan_path.exists():
        raise RuntimeError(
            "No in-memory cached scan and no dev_tools/cache/latest_plan.json fallback found."
        )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    sources = [
        t["raw"]["source_target"]
        for t in plan.get("targets", [])
        if isinstance(t, dict)
        and isinstance(t.get("raw"), dict)
        and isinstance(t["raw"].get("source_target"), dict)
    ]
    if not sources:
        raise RuntimeError(f"Cached plan has no source matches: {plan_path}")

    left_frame = cv2.imread(str(left_path)) if left_path.exists() else None
    right_frame = cv2.imread(str(right_path)) if right_path.exists() else None
    if left_frame is None or right_frame is None:
        raise RuntimeError(
            "Cached plan exists, but the saved full-frame images are missing "
            f"or unreadable: {left_path}, {right_path}"
        )

    return {
        "timestamp": plan.get("created_at", plan_path.stat().st_mtime),
        "frame_mode": plan.get("frame_mode", "raw"),
        "rectified": plan.get("frame_mode") == "rectified",
        "left_frame": left_frame,
        "right_frame": right_frame,
        "left_detections": [_det_from_cached_match(src, "left") for src in sources],
        "right_detections": [_det_from_cached_match(src, "right") for src in sources],
        "left_crop": None,
        "right_crop": None,
        "params": {"source": str(plan_path)},
        "class_names": {},
        "disk_fallback": True,
    }


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def run_survey_and_cache(params):
    """Detect + stereo-match + persist raw frames and matched pixel coords to disk."""
    scan_result = run_debug_scan(params)
    match_result = run_cached_match()

    _SURVEY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(_SURVEY_CACHE_DIR / "left.jpg"), state.last_scan["left_frame"])
    cv2.imwrite(str(_SURVEY_CACHE_DIR / "right.jpg"), state.last_scan["right_frame"])

    meta = {
        "timestamp": state.last_scan["timestamp"],
        "frame_mode": state.last_scan["frame_mode"],
        "left_count": scan_result["left_count"],
        "right_count": scan_result["right_count"],
        "matched_count": match_result["matched_count"],
        "matches": _make_serializable(match_result["matches"]),
        "left_detections": _make_serializable(state.last_scan["left_detections"]),
        "right_detections": _make_serializable(state.last_scan["right_detections"]),
        "left_crop": state.last_scan.get("left_crop"),
        "right_crop": state.last_scan.get("right_crop"),
    }
    (_SURVEY_CACHE_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    return {
        **scan_result,
        "matched_count": match_result["matched_count"],
        "match_s": match_result["match_s"],
        "match_left_image": match_result["left_image"],
        "match_right_image": match_result["right_image"],
        "matches": match_result["matches"],
        "cached_to_disk": True,
    }


def run_cached_match():
    scan = state.last_scan or _load_disk_cached_scan()

    t0 = time.perf_counter()

    left_dets = list(scan["left_detections"])
    right_dets = list(scan["right_detections"])

    # Cached scan points are already in rectified pixel space when frame_mode=rectified.
    # Mark them explicitly so vision.matching does not apply raw->rectified remapping again.
    if scan.get("frame_mode") == "rectified":
        def _tag_rectified(dets):
            out = []
            for d in dets:
                if not isinstance(d, dict):
                    out.append(d)
                    continue
                item = dict(d)
                if "point" in item and "point_rectified" not in item:
                    item["point_rectified"] = tuple(item["point"])
                if "box" in item and "box_rectified" not in item:
                    item["box_rectified"] = tuple(item["box"])
                out.append(item)
            return out

        left_dets = _tag_rectified(left_dets)
        right_dets = _tag_rectified(right_dets)

    result = match_points(
        left_dets,
        right_dets,
    )

    raw_matches = result[0] if isinstance(result, tuple) else result
    matches = [normalize_match(m) for m in raw_matches]

    match_s = time.perf_counter() - t0

    left_det_img = draw_detections(scan["left_frame"], scan["left_detections"])
    right_det_img = draw_detections(scan["right_frame"], scan["right_detections"])

    left_match_img, right_match_img = draw_matches(left_det_img, right_det_img, matches)

    return {
        "frame_mode": scan["frame_mode"],
        "scan_timestamp": scan["timestamp"],
        "left_count": len(scan["left_detections"]),
        "right_count": len(scan["right_detections"]),
        "matched_count": len(matches),
        "match_s": round(match_s, 3),
        "matches": matches,
        "left_image": b64_img(left_match_img),
        "right_image": b64_img(right_match_img),
        "class_names": scan.get("class_names", {}),
        "disk_fallback": bool(scan.get("disk_fallback", False)),
    }
