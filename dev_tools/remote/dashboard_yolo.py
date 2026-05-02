"""YOLO detection helpers: ensure detector, parameter parsers, debug scan, cached match."""

import time

from dashboard_state import (
    state,
    SURVEY_BURST_COUNT,
    SURVEY_MIN_HITS,
    SURVEY_YOLO_IMGSZ,
    SURVEY_TARGET_CLASSES,
    SURVEY_POINT_MODE,
)
from dashboard_images import b64_img, draw_crop, draw_detections, draw_matches
from dashboard_camera import parse_bool, crop_bounds_for_side, ensure_cameras
from dashboard_rectify import maybe_rectify_pair

# Patch broken Jetson torchvision NMS before importing Ultralytics/YOLO.
# This matches the working bringup scripts.
import importlib.util
from pathlib import Path

_NMS_PATCH_PATH = Path(__file__).resolve().parents[2] / "bringup" / "_nms_patch.py"

if _NMS_PATCH_PATH.exists():
    _spec = importlib.util.spec_from_file_location("_nms_patch", _NMS_PATCH_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
else:
    print(f"[WARN] NMS patch not found: {_NMS_PATCH_PATH}")
    
from vision.detectors.ai_detector import AIDetector
from vision.matching import match_points

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

    old_left_conf = det.cv_left.conf
    old_right_conf = det.cv_right.conf

    if conf_override is not None:
        det.cv_left.conf = conf_override
        det.cv_right.conf = conf_override

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
        det.cv_left.conf = old_left_conf
        det.cv_right.conf = old_right_conf

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
        "confidence_override": conf_override,
        "imgsz": imgsz,
        "frame_mode": frame_mode,
        "cached": True,
        "suggested_config": {
            "BURST_COUNT": burst_count,
            "MIN_HITS": min_hits,
            "POINT_MODE": point_mode,
            "TARGET_CLASSES": class_filter,
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

def normalize_match(match):
    if isinstance(match, dict):
        if "left_px" in match and "right_px" in match:
            return match
    if isinstance(match, (list, tuple)) and len(match) >= 2:
        return {
            "left_px": tuple(match[0]),
            "right_px": tuple(match[1]),
            "score": float(match[2]) if len(match) >= 3 and isinstance(match[2], (int, float)) else 1.0,
        }
    raise ValueError(f"Unknown match format: {type(match)} {match}")


def run_cached_match():
    if state.last_scan is None:
        raise RuntimeError("No cached scan yet. Run Scan / Save Points first.")

    t0 = time.perf_counter()

    result = match_points(
        state.last_scan["left_detections"],
        state.last_scan["right_detections"],
    )

    raw_matches = result[0] if isinstance(result, tuple) else result
    matches = [normalize_match(m) for m in raw_matches]

    match_s = time.perf_counter() - t0

    left_det_img = draw_detections(state.last_scan["left_frame"], state.last_scan["left_detections"])
    right_det_img = draw_detections(state.last_scan["right_frame"], state.last_scan["right_detections"])

    left_match_img, right_match_img = draw_matches(left_det_img, right_det_img, matches)

    return {
        "frame_mode": state.last_scan["frame_mode"],
        "scan_timestamp": state.last_scan["timestamp"],
        "left_count": len(state.last_scan["left_detections"]),
        "right_count": len(state.last_scan["right_detections"]),
        "matched_count": len(matches),
        "match_s": round(match_s, 3),
        "matches": matches,
        "left_image": b64_img(left_match_img),
        "right_image": b64_img(right_match_img),
        "class_names": state.last_scan["class_names"],
    }
