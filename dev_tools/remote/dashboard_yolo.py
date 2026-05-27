"""YOLO detection helpers: ensure detector, parameter parsers, debug scan, cached match."""

import json
import time
import numpy as np

from dashboard_state import (
    state,
    SURVEY_BURST_COUNT,
    SURVEY_MIN_HITS,
    SURVEY_YOLO_IMGSZ,
    SURVEY_POINT_MODE,
    AI_CONFIDENCE,
    AI_CLASS_CONFIDENCE,
    TARGET_CLASSES,
    AVOID_CLASSES,
    SURVEY_TARGET_CLASSES,
    SURVEY_EVAL_MODEL_CHOICE,
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
    
from config import (
    CV_WEIGHTS_DIR,
    DEFAULT_MODEL_ENGINE,
    DEFAULT_MODEL_PT,
    MODEL_MAP,
    SURVEY_POS_X,
    SURVEY_POS_Y,
)
from vision.detectors.ai_detector import AIDetector
from vision.matching import match_points

_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
_SURVEY_CACHE_DIR = _CACHE_DIR / "survey"

# =============================================================================
# Parameter parsers
# =============================================================================

_SEG_MODEL_EXTS = {".pt", ".engine", ".onnx"}


def _configured_model_label():
    return DEFAULT_MODEL_ENGINE or DEFAULT_MODEL_PT or "configured default"


def _normalise_model_choice(value):
    text = str(value or "").strip()
    return text or "__config__"


def get_segmentation_model_options():
    options = [
        {
            "value": "__config__",
            "label": f"config default ({_configured_model_label()})",
        }
    ]

    seen = {"__config__"}
    for alias, filename in sorted(MODEL_MAP.items()):
        suffix = Path(str(filename)).suffix.lower()
        if suffix not in _SEG_MODEL_EXTS:
            continue
        options.append({
            "value": alias,
            "label": f"{alias} ({filename})",
        })
        seen.add(alias)
        seen.add(Path(str(filename)).name)

    if CV_WEIGHTS_DIR.exists():
        for path in sorted(CV_WEIGHTS_DIR.iterdir()):
            if path.suffix.lower() not in _SEG_MODEL_EXTS:
                continue
            if path.name in seen:
                continue
            options.append({
                "value": path.name,
                "label": path.name,
            })
            seen.add(path.name)

    return options

def snap32(value):
    value = int(value)
    return max(32, int(round(value / 32.0) * 32))


def parse_classes(value):
    if value is None:
        return TARGET_CLASSES
    text = str(value).strip().lower()
    if text in ("", "all", "none", "null"):
        return "all"
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_avoid_classes(value):
    """Parse avoid_classes param.

    Semantics:
    - blank/omitted -> use detector configured avoid_classes
    - "none"/"null" -> disable avoid classes for this run
    - "0,1" -> explicit avoid class list
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in ("none", "null"):
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_conf(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _default_avoid_confidence():
    conf_map = dict(AI_CLASS_CONFIDENCE or {})
    vals = []
    for cls_id in list(AVOID_CLASSES or []):
        if int(cls_id) in conf_map:
            vals.append(float(conf_map[int(cls_id)]))
    if vals:
        return min(vals)
    return None


def _round_timing(value, digits=4):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _sum_timing(left, right, key):
    lv = _round_timing((left or {}).get(key), 6) or 0.0
    rv = _round_timing((right or {}).get(key), 6) or 0.0
    return round(lv + rv, 6)


def _build_timing_summary(left_timing, right_timing, wall_s, point_mode):
    combined = {
        "yolo_time_s": _sum_timing(left_timing, right_timing, "yolo_time_s"),
        "boxpoint_time_s": _sum_timing(left_timing, right_timing, "boxpoint_time_s"),
        "grouping_time_s": _sum_timing(left_timing, right_timing, "grouping_time_s"),
        "merge_time_s": _sum_timing(left_timing, right_timing, "merge_time_s"),
        "qpoint_time_s": _sum_timing(left_timing, right_timing, "qpoint_time_s"),
        "total_time_s": _sum_timing(left_timing, right_timing, "total_time_s"),
        "wall_time_s": round(float(wall_s), 6),
    }
    combined["box_center_path_s"] = round(
        combined["yolo_time_s"] + combined["boxpoint_time_s"] + combined["merge_time_s"],
        6,
    )
    combined["qpoint_extra_s"] = combined["qpoint_time_s"]

    return {
        "point_mode": point_mode,
        "left": dict(left_timing or {}),
        "right": dict(right_timing or {}),
        "combined": combined,
        "qpoint_model_batched": bool(
            (left_timing or {}).get("qpoint_batched") or (right_timing or {}).get("qpoint_batched")
        ),
    }


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
    model_options = get_segmentation_model_options()
    if kind == "fine":
        return {
            "title": "Fine Align / Re-ID YOLO Debug",
            "default_mode": "center",
            "default_crop_w": 384,
            "default_crop_h": 384,
            "default_burst": 5,
            "default_min_hits": 1,
            "model_options": model_options,
            "default_model_choice": "__config__",
        }
    return {
        "title": "Survey YOLO Debug",
        "default_mode": "center_facing",
        "default_crop_w": 704,
        "default_crop_h": 704,
        "default_burst": SURVEY_BURST_COUNT,
        "default_min_hits": SURVEY_MIN_HITS,
        "model_options": model_options,
        "default_model_choice": "__config__",
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

def ensure_detector(model_choice=None):
    selected = _normalise_model_choice(model_choice)
    yolo_path = None if selected == "__config__" else selected

    if state.detector is None or state.detector_model_choice != selected:
        state.detector = AIDetector(yolo_path=yolo_path)
        state.detector.warmup()
        state.detector_model_choice = selected
    return state.detector


def get_class_names(det=None):
    if det is None:
        if state.detector is None:
            return {}
        det = state.detector
    names = getattr(det.cv_left.yolo, "names", {})
    return {str(k): v for k, v in dict(names).items()}


def get_detector_info(det=None):
    det = det or state.detector
    if det is None:
        return {
            "loaded": False,
            "choice": state.detector_model_choice,
            "model": None,
            "backend": None,
        }
    return {
        "loaded": True,
        "choice": state.detector_model_choice,
        "model": str(getattr(det, "yolo_path", "")),
        "backend": getattr(det, "yolo_backend", None),
    }


# =============================================================================
# Debug scan
# =============================================================================

def run_debug_scan(params):
    cam = ensure_cameras()
    model_choice = _normalise_model_choice(params.get("model", "__config__"))
    det = ensure_detector(model_choice)

    use_rectification = parse_bool(params.get("rectified", False))
    burst_count = int(params.get("burst_count", SURVEY_BURST_COUNT))
    min_hits = int(params.get("min_hits", SURVEY_MIN_HITS))
    point_mode = params.get("point_mode", SURVEY_POINT_MODE or "box_center")
    class_filter = parse_classes(params.get("classes", ""))
    avoid_filter = parse_avoid_classes(params.get("avoid_classes", ""))
    conf_override = parse_conf(params.get("conf", ""))
    avoid_conf_override = parse_conf(params.get("avoid_conf", ""))
    iom_enabled = parse_bool(params.get("iom", "true"))
    if avoid_conf_override is None:
        avoid_conf_override = _default_avoid_confidence()
    requested_imgsz = yolo_imgsz_from_params(params)

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
    imgsz = det.cv_left._resolve_imgsz(left_crop_frames, requested_imgsz)

    old_left_conf  = det.cv_left.conf
    old_right_conf = det.cv_right.conf
    old_left_avoid  = det.cv_left.avoid_classes
    old_right_avoid = det.cv_right.avoid_classes
    old_left_class_conf = dict(det.cv_left.class_conf)
    old_right_class_conf = dict(det.cv_right.class_conf)
    old_left_avoid_conf = det.cv_left.avoid_confidence_override
    old_right_avoid_conf = det.cv_right.avoid_confidence_override
    old_left_iom = getattr(det.cv_left, "iom_enabled", True)
    old_right_iom = getattr(det.cv_right, "iom_enabled", True)

    if conf_override is not None:
        det.cv_left.conf  = conf_override
        det.cv_right.conf = conf_override

    det.cv_left.iom_enabled = iom_enabled
    det.cv_right.iom_enabled = iom_enabled

    # Effective avoid classes for this scan:
    # - explicit avoid_classes param wins;
    # - otherwise use detector defaults.
    effective_avoid = list(avoid_filter) if avoid_filter is not None else list(old_left_avoid or [])

    det.cv_left.avoid_classes = effective_avoid
    det.cv_right.avoid_classes = effective_avoid
    if avoid_conf_override is not None and effective_avoid:
        left_class_conf = dict(old_left_class_conf)
        right_class_conf = dict(old_right_class_conf)
        for cls_id in effective_avoid:
            left_class_conf[int(cls_id)] = float(avoid_conf_override)
            right_class_conf[int(cls_id)] = float(avoid_conf_override)
        det.cv_left.class_conf = left_class_conf
        det.cv_right.class_conf = right_class_conf
        det.cv_left.avoid_confidence_override = float(avoid_conf_override)
        det.cv_right.avoid_confidence_override = float(avoid_conf_override)

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
        left_timing = dict(getattr(det.cv_left, "last_burst_timing", {}) or {})
        right_timing = dict(getattr(det.cv_right, "last_burst_timing", {}) or {})
        timing = _build_timing_summary(left_timing, right_timing, yolo_s, point_mode)

    finally:
        det.cv_left.conf  = old_left_conf
        det.cv_right.conf = old_right_conf
        det.cv_left.avoid_classes  = old_left_avoid
        det.cv_right.avoid_classes = old_right_avoid
        det.cv_left.class_conf = old_left_class_conf
        det.cv_right.class_conf = old_right_class_conf
        det.cv_left.avoid_confidence_override = old_left_avoid_conf
        det.cv_right.avoid_confidence_override = old_right_avoid_conf
        det.cv_left.iom_enabled = old_left_iom
        det.cv_right.iom_enabled = old_right_iom

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
        "model_info": get_detector_info(det),
    }

    # Effective avoid list used in this scan (after removing UI-requested classes).

    return {
        "left_count": len(stable_left_full),
        "right_count": len(stable_right_full),
        "capture_s": round(capture_s, 3),
        "yolo_s": round(yolo_s, 3),
        "total_runtime_s": _round_timing((timing["combined"] or {}).get("wall_time_s"), 3),
        "boxpoint_runtime_s": _round_timing((timing["combined"] or {}).get("boxpoint_time_s"), 3),
        "qpoint_runtime_s": _round_timing((timing["combined"] or {}).get("qpoint_time_s"), 3),
        "timing": timing,
        "left_crop": left_crop_box,
        "right_crop": right_crop_box,
        "left_image": b64_img(left_overlay),
        "right_image": b64_img(right_overlay),
        "class_names": class_names,
        "model_info": get_detector_info(det),
        "requested_classes": class_filter,
        "avoid_classes": effective_avoid,
        "confidence_override": conf_override,
        "avoid_confidence_override": avoid_conf_override,
        "imgsz": imgsz,
        "requested_imgsz": requested_imgsz,
        "frame_mode": frame_mode,
        "cached": True,
        "suggested_config": {
            "BURST_COUNT": burst_count,
            "MIN_HITS": min_hits,
            "POINT_MODE": point_mode,
            "TARGET_CLASSES": class_filter,
            "AVOID_CLASSES": effective_avoid,
            "CONFIDENCE_OVERRIDE": conf_override,
            "AVOID_CONFIDENCE_OVERRIDE": avoid_conf_override,
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
            "SEGMENTATION_MODEL": model_choice,
        },
    }


def _crop_bounds_for_image(params, width, height):
    mode = str(params.get("mode", "center") or "center").strip().lower()
    crop_w = int(params.get("crop_w", min(704, width)))
    crop_h = int(params.get("crop_h", min(704, height)))
    offset_x = int(params.get("offset_x", params.get("left_offset_x", 0)))
    offset_y = int(params.get("offset_y", params.get("left_offset_y", 0)))

    crop_w = max(32, min(width, crop_w))
    crop_h = max(32, min(height, crop_h))

    if mode == "full":
        return {"x0": 0, "y0": 0, "x1": width, "y1": height}

    cx = width // 2 + offset_x
    cy = height // 2 + offset_y

    if mode == "left":
        cx = width // 4 + offset_x
    elif mode == "right":
        cx = (3 * width) // 4 + offset_x
    elif mode == "top":
        cy = height // 4 + offset_y
    elif mode == "bottom":
        cy = (3 * height) // 4 + offset_y

    x0 = int(max(0, min(width - crop_w, cx - crop_w // 2)))
    y0 = int(max(0, min(height - crop_h, cy - crop_h // 2)))
    return {"x0": x0, "y0": y0, "x1": x0 + crop_w, "y1": y0 + crop_h}


def run_validation_scan(params, uploaded_files):
    model_choice = _normalise_model_choice(params.get("model", "__config__"))
    det = ensure_detector(model_choice)

    burst_count = max(1, int(params.get("burst_count", SURVEY_BURST_COUNT)))
    min_hits = max(1, int(params.get("min_hits", SURVEY_MIN_HITS)))
    point_mode = params.get("point_mode", SURVEY_POINT_MODE or "box_center")
    class_filter = parse_classes(params.get("classes", ""))
    avoid_filter = parse_avoid_classes(params.get("avoid_classes", ""))
    conf_override = parse_conf(params.get("conf", ""))
    avoid_conf_override = parse_conf(params.get("avoid_conf", ""))
    iom_enabled = parse_bool(params.get("iom", "true"))
    if avoid_conf_override is None:
        avoid_conf_override = _default_avoid_confidence()
    requested_imgsz = yolo_imgsz_from_params(params)

    old_conf = det.cv_left.conf
    old_avoid = det.cv_left.avoid_classes
    old_class_conf = dict(det.cv_left.class_conf)
    old_avoid_conf = det.cv_left.avoid_confidence_override
    old_iom = getattr(det.cv_left, "iom_enabled", True)

    if conf_override is not None:
        det.cv_left.conf = conf_override

    det.cv_left.iom_enabled = iom_enabled

    effective_avoid = list(avoid_filter) if avoid_filter is not None else list(old_avoid or [])
    det.cv_left.avoid_classes = effective_avoid

    if avoid_conf_override is not None and effective_avoid:
        class_conf = dict(old_class_conf)
        for cls_id in effective_avoid:
            class_conf[int(cls_id)] = float(avoid_conf_override)
        det.cv_left.class_conf = class_conf
        det.cv_left.avoid_confidence_override = float(avoid_conf_override)

    try:
        results = []
        t_all = time.perf_counter()
        for file_storage in uploaded_files:
            raw = file_storage.read()
            if not raw:
                continue

            arr = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                continue

            h, w = image.shape[:2]
            crop_box = _crop_bounds_for_image(params, w, h)
            x0, y0, x1, y1 = crop_box["x0"], crop_box["y0"], crop_box["x1"], crop_box["y1"]
            crop_img = image[y0:y1, x0:x1]
            crop_frames = [crop_img for _ in range(burst_count)]

            imgsz = det.cv_left._resolve_imgsz(crop_frames, requested_imgsz)
            t_img = time.perf_counter()
            stable = det.cv_left.return_burst_stable(
                crop_frames,
                min_stable_views=min_hits,
                classes_override=class_filter,
                debug_label=f"[VALIDATION {file_storage.filename}]",
                imgsz=imgsz,
                heatmap_final=(point_mode != "box_center"),
                point_mode=point_mode,
            )
            infer_s = time.perf_counter() - t_img

            full_dets = translate_detections_to_full(stable, crop_box)
            overlay = draw_detections(draw_crop(image, crop_box, "crop"), full_dets, color=(0, 0, 255))

            results.append({
                "filename": file_storage.filename,
                "count": len(full_dets),
                "image": b64_img(overlay),
                "detections": full_dets,
                "crop": crop_box,
                "imgsz": imgsz,
                "inference_s": round(infer_s, 4),
            })

        total_s = time.perf_counter() - t_all
    finally:
        det.cv_left.conf = old_conf
        det.cv_left.avoid_classes = old_avoid
        det.cv_left.class_conf = old_class_conf
        det.cv_left.avoid_confidence_override = old_avoid_conf
        det.cv_left.iom_enabled = old_iom

    class_names = get_class_names(det)
    return {
        "files_processed": len(results),
        "total_detections": int(sum(item["count"] for item in results)),
        "total_runtime_s": round(total_s if 'total_s' in locals() else 0.0, 4),
        "results": results,
        "class_names": class_names,
        "model_info": get_detector_info(det),
        "requested_classes": class_filter,
        "avoid_classes": effective_avoid,
        "confidence_override": conf_override,
        "avoid_confidence_override": avoid_conf_override,
        "burst_count": burst_count,
        "min_hits": min_hits,
        "point_mode": point_mode,
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


def _cache_fine_align_plan_from_survey(match_result):
    """Build the Fine Align latest_plan cache from the current survey scan."""
    from control.coarse_move import TriangulationCoarseMover
    from pipeline.steps.fine_align_debug import (
        fine_align_debug_dir,
        normalize_planned_targets_to_plan,
        save_latest_plan,
    )
    from pipeline.steps.match_plan import run_match_and_plan

    scan = state.last_scan or {}
    matches = list(match_result.get("matches") or [])
    frame_mode = scan.get("frame_mode", match_result.get("frame_mode", "raw"))
    survey_ref_xy = (SURVEY_POS_X, SURVEY_POS_Y)
    survey_plan_path = _SURVEY_CACHE_DIR / "match_plan.json"

    debug_dir = fine_align_debug_dir()
    debug_files = {}
    if scan.get("left_frame") is not None:
        left_path = debug_dir / "latest_full_left.jpg"
        cv2.imwrite(str(left_path), scan["left_frame"])
        debug_files["full_left"] = str(left_path)
    if scan.get("right_frame") is not None:
        right_path = debug_dir / "latest_full_right.jpg"
        cv2.imwrite(str(right_path), scan["right_frame"])
        debug_files["full_right"] = str(right_path)

    if matches:
        coarse_mover = TriangulationCoarseMover()
        matched_targets, solved_targets, planned_targets = run_match_and_plan(
            scan.get("left_detections", []),
            scan.get("right_detections", []),
            coarse_mover,
            start_xy=survey_ref_xy,
            output_path=survey_plan_path,
            precomputed_matches=matches,
        )
    else:
        matched_targets, solved_targets, planned_targets = [], [], []
        survey_plan_path.write_text(
            json.dumps({
                "matched_targets": [],
                "unmatched_left": [],
                "unmatched_right": [],
                "solved_targets": [],
                "planned_targets": [],
                "grid_filter": {},
                "grid_summary": {},
            }, indent=2),
            encoding="utf-8",
        )

    fine_align_plan = normalize_planned_targets_to_plan(
        planned_targets,
        survey_ref_xy=survey_ref_xy,
        frame_mode=frame_mode,
    )
    latest_plan = save_latest_plan(fine_align_plan)

    return {
        "latest_plan_path": str(latest_plan),
        "survey_match_plan_path": str(survey_plan_path),
        "frame_mode": frame_mode,
        "survey_ref_xy": list(survey_ref_xy),
        "matched_count": len(matched_targets),
        "solved_count": len(solved_targets),
        "planned_count": len(planned_targets),
        "target_count": len(fine_align_plan.get("targets", [])),
        "debug_files": debug_files,
    }


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
        "model_info": _make_serializable(state.last_scan.get("model_info")),
    }
    (_SURVEY_CACHE_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    fine_align_cache = _cache_fine_align_plan_from_survey(match_result)

    return {
        **scan_result,
        "matched_count": match_result["matched_count"],
        "match_s": match_result["match_s"],
        "match_left_image": match_result["left_image"],
        "match_right_image": match_result["right_image"],
        "matches": match_result["matches"],
        "cached_to_disk": True,
        "fine_align_cache": fine_align_cache,
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
        "model_info": scan.get("model_info"),
        "disk_fallback": bool(scan.get("disk_fallback", False)),
    }
