#!/usr/bin/env python3
"""Render an annotated video from a raw trial recording folder."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import (  # noqa: E402
    AI_CONFIDENCE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FINE_ALIGN_CROP_SCALE,
    FINE_ALIGN_REID_CROP_HALF_PX,
    LASER_FIRE_DURATION_SEC,
    CALIB_NPZ_PATH,
    RECT_NPZ_PATH,
    CALIBRATION_EXPECTS_UNFLIPPED,
    TRI_X_SIGN,
    TRI_Y_SIGN,
    TRI_X_GAIN,
    TRI_Y_GAIN,
    LASER_OFFSET_X_MM,
    LASER_OFFSET_Y_MM,
    SURVEY_POS_X,
    SURVEY_POS_Y,
    WORKSPACE_X_MIN,
    WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Y_MAX,
    SURVEY_CROP_MODE,
    SURVEY_CROP_HALF_X_PX,
    SURVEY_CROP_HALF_Y_PX,
    SURVEY_PROJECT_CROP_MARGIN_PX,
    SURVEY_PROJECT_Z_MIN_MM,
    SURVEY_PROJECT_Z_MAX_MM,
    SURVEY_PROJECT_Z_SAMPLES,
)
from vision.workspace_crop import project_workspace_crop_left_right


# =============================================================================
# Quick Settings
# =============================================================================

# Put the trial number here, then run:
#   python dev_tools/render_trial_video.py
#
# Example: TRIAL_NUMBER = 7 matches trial_recordings/trial_007_*
TRIAL_NUMBER = 69

# Optional fallback if you prefer a path instead of a number.
TRIAL_RUN_DIR = None

# Output can be a folder or a full .mp4/.avi path.
# Default saves into REPO_ROOT/rendered_videos/.
OUTPUT_PATH = "rendered_videos"

RUN_YOLO = True
# Offline YOLO is only used for RE-ID visualization, not every rendered frame.
YOLO_MODEL = None
YOLO_IMGSZ = 640
YOLO_CONF = AI_CONFIDENCE

# Precompute expensive RE-ID crop detections before the video write loop.
REID_PRECOMPUTE = True
# Keep this at 1 by default; batched per-side inference is already fast, and
# parallel model execution is less reliable on this CPU-only machine.
REID_PRECOMPUTE_SIDE_WORKERS = 1

# Render independent/re-ID chunks in parallel, but keep LK/lock/fire chunks ordered.
PARALLEL_RENDER = True
PARALLEL_FRAME_WORKERS = 4
PARALLEL_MIN_CHUNK_FRAMES = 8

OUTPUT_FPS = None
MAX_FRAMES = None

PD_FLOW_OVERLAY = True
PD_FLOW_TRAIL_LEN = 28
PD_CV_FLASH_SEC = 0.45
REID_CV_FLASH_SEC = 0.9
SURVEY_CV_FLASH_SEC = 2.0
PD_MINIMAP_SIZE = 172


# =============================================================================
# Layout constants
# =============================================================================

HEADER_H = 46  # pixels reserved at top for progress bars

# LK optical-flow parameters used in the renderer for visual tracking only.
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Fine-align crop geometry (mirrors fine_align.py constants)
_FINE_W = int(FRAME_WIDTH * FINE_ALIGN_CROP_SCALE)
_FINE_H = int(FRAME_HEIGHT * FINE_ALIGN_CROP_SCALE)
_CROP_X0 = (FRAME_WIDTH - _FINE_W) // 2
_CROP_Y0 = (FRAME_HEIGHT - _FINE_H) // 2
_CROP_X1 = _CROP_X0 + _FINE_W
_CROP_Y1 = _CROP_Y0 + _FINE_H

# BGR colors for each state segment in the timeline bar.
_STATE_COLORS = {
    'HOME':              ( 70,  70,  70),
    'SURVEY':            (  0, 180, 210),
    'SURVEY_CONFIRM':    (  0, 180, 210),
    'DETECT':            (  0, 180, 210),
    'MATCH':             (  0, 180, 210),
    'PLAN':              (  0, 180, 210),
    'TARGET':            (  0, 150, 230),
    'TRAVEL':            (  0, 150, 230),
    'POST_TRAVEL':       (  0, 150, 230),
    'FINE_ALIGN':        ( 40, 200,  40),
    'LOCKED':            ( 40, 200,  40),
    'FIRE':              ( 30,  30, 220),
    'FIRED':             ( 30,  30, 220),
    'FAILED_FINE_ALIGN': ( 30, 100, 220),
}
_DEFAULT_STATE_COLOR = (50, 50, 50)

_SURVEY_CROP_OVERLAY_CACHE = {}


def _centered_survey_crop_rect(frame_w, frame_h):
    half_x = SURVEY_CROP_HALF_X_PX
    half_y = SURVEY_CROP_HALF_Y_PX

    if half_x is None or half_y is None:
        return None

    cx, cy = frame_w // 2, frame_h // 2
    x0 = max(0, int(cx - half_x))
    x1 = min(frame_w, int(cx + half_x))
    y0 = max(0, int(cy - half_y))
    y1 = min(frame_h, int(cy + half_y))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _get_survey_crop_overlay_rects(frame_w, frame_h):
    mode = str(SURVEY_CROP_MODE or "projected").strip().lower()
    if mode not in ("projected", "centered"):
        mode = "projected"

    key = (int(frame_w), int(frame_h), mode)
    cached = _SURVEY_CROP_OVERLAY_CACHE.get(key)
    if cached is not None:
        return cached

    out = {
        "mode": "none",
        "left_rect": None,
        "right_rect": None,
        "reason": None,
    }

    if mode == "projected":
        left_rect, right_rect, info = project_workspace_crop_left_right(
            frame_width=frame_w,
            frame_height=frame_h,
            calib_npz_path=CALIB_NPZ_PATH,
            rect_npz_path=RECT_NPZ_PATH,
            workspace_x_min=WORKSPACE_X_MIN,
            workspace_x_max=WORKSPACE_X_MAX,
            workspace_y_min=WORKSPACE_Y_MIN,
            workspace_y_max=WORKSPACE_Y_MAX,
            survey_pos_x=SURVEY_POS_X,
            survey_pos_y=SURVEY_POS_Y,
            tri_sign_x=TRI_X_SIGN,
            tri_sign_y=TRI_Y_SIGN,
            tri_x_gain=TRI_X_GAIN,
            tri_y_gain=TRI_Y_GAIN,
            laser_offset_x_mm=LASER_OFFSET_X_MM,
            laser_offset_y_mm=LASER_OFFSET_Y_MM,
            z_min_mm=SURVEY_PROJECT_Z_MIN_MM,
            z_max_mm=SURVEY_PROJECT_Z_MAX_MM,
            z_samples=SURVEY_PROJECT_Z_SAMPLES,
            margin_px=SURVEY_PROJECT_CROP_MARGIN_PX,
            calibration_expects_unflipped=CALIBRATION_EXPECTS_UNFLIPPED,
        )
        if left_rect is not None and right_rect is not None:
            out["mode"] = "projected"
            out["left_rect"] = left_rect
            out["right_rect"] = right_rect
            _SURVEY_CROP_OVERLAY_CACHE[key] = out
            return out
        out["reason"] = (info or {}).get("reason", "project failed")

    centered = _centered_survey_crop_rect(frame_w, frame_h)
    if centered is not None:
        out["mode"] = "centered"
        out["left_rect"] = centered
        out["right_rect"] = centered

    _SURVEY_CROP_OVERLAY_CACHE[key] = out
    return out


# =============================================================================
# CLI / path helpers
# =============================================================================

def _find_trial_dir(trial_number):
    recordings_dir = REPO_ROOT / "trial_recordings"
    trial_id = int(trial_number)
    matches = sorted(recordings_dir.glob(f"trial_{trial_id:03d}_*"))
    matches = [p for p in matches if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No recording folder found for trial {trial_id:03d} in {recordings_dir}")
    if len(matches) > 1:
        print(f"[render] Multiple folders matched trial {trial_id:03d}; using latest: {matches[-1]}")
    return matches[-1]


def _default_run_dir():
    if TRIAL_RUN_DIR:
        return Path(TRIAL_RUN_DIR)
    if TRIAL_NUMBER is not None:
        return _find_trial_dir(TRIAL_NUMBER)
    recordings_dir = REPO_ROOT / "trial_recordings"
    matches = sorted(p for p in recordings_dir.glob("trial_*") if p.is_dir())
    if not matches:
        raise FileNotFoundError(
            "No trial folder configured. Set TRIAL_NUMBER at the top of this file "
            "or pass a run folder on the command line."
        )
    print(f"[render] TRIAL_NUMBER is not set; using latest folder: {matches[-1]}")
    return matches[-1]


def _resolve_output_path(output_spec, run_dir):
    if output_spec:
        output = Path(output_spec)
        if not output.is_absolute():
            output = REPO_ROOT / output
        if output.suffix.lower() not in (".mp4", ".avi"):
            output.mkdir(parents=True, exist_ok=True)
            output = output / f"{run_dir.name}_annotated.mp4"
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
        return output

    output_dir = REPO_ROOT / "Rendered_Videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{run_dir.name}_annotated.mp4"


def _load_records(run_dir):
    jsonl_path = run_dir / "manifest.jsonl"
    if jsonl_path.exists():
        records = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    for name in ("manifest.json", "frames.json", "trial_manifest.json"):
        path = run_dir / name
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("frames", [])

    raise FileNotFoundError(f"No manifest.jsonl/json found in {run_dir}")


def _estimate_fps(records, fallback=15.0):
    stamps = [r.get("timestamp_monotonic") for r in records if r.get("timestamp_monotonic") is not None]
    if len(stamps) < 2:
        return fallback
    intervals = np.diff(np.asarray(stamps, dtype=float))
    intervals = intervals[intervals > 1e-4]
    if intervals.size == 0:
        return fallback
    return float(np.clip(1.0 / np.median(intervals), 1.0, 60.0))


# =============================================================================
# Pre-processing
# =============================================================================

def _preprocess_records(records):
    stamps = [r.get("timestamp_monotonic") for r in records]
    valid = [t for t in stamps if t is not None]
    first_ts = valid[0] if valid else 0.0
    last_ts = valid[-1] if valid else 0.0
    total_duration = max(last_ts - first_ts, 1e-3)

    total_targets = 0
    for r in records:
        ptl = r.get("planned_target_list")
        if ptl:
            total_targets = max(total_targets, len(ptl))

    segments = []
    n = len(records)
    for i, r in enumerate(records):
        t = r.get("timestamp_monotonic")
        if t is None:
            continue
        t_start = (float(t) - first_ts) / total_duration
        if i + 1 < n:
            t_next = records[i + 1].get("timestamp_monotonic")
            t_end = ((float(t_next) - first_ts) / total_duration) if t_next is not None else t_start
        else:
            t_end = 1.0
        segments.append((t_start, min(t_end, 1.0), r.get("state_name", "?")))

    return first_ts, total_duration, total_targets, segments


def _target_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_pt(value):
    if value is None or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _collect_pd_seed_points(records):
    """Map planned target id to the recorded re-ID box center used to seed PD/LK."""
    seeds = {}
    prev_hit_count = 0
    for record in records:
        hits = record.get("hit_targets_so_far") or []
        hit_count = len(hits)

        # Preferred path: explicitly tagged target id from main.py.
        for hit in hits:
            tid = _target_id(hit.get("target_id"))
            if tid is None or tid in seeds:
                continue
            left_pt = _float_pt(hit.get("left_px"))
            right_pt = _float_pt(hit.get("right_px"))
            if left_pt is not None and right_pt is not None:
                seeds[tid] = {"left": left_pt, "right": right_pt}

        # Legacy fallback: if a new hit appears while a target is active,
        # attribute that new hit to the active planned target id.
        active_id = _target_id(record.get("current_target_id"))
        if active_id is not None and hit_count > prev_hit_count and active_id not in seeds:
            newest = hits[-1] if hits else None
            if newest is not None:
                left_pt = _float_pt(newest.get("left_px"))
                right_pt = _float_pt(newest.get("right_px"))
                if left_pt is not None and right_pt is not None:
                    seeds[active_id] = {"left": left_pt, "right": right_pt}

        # Last-resort fallback for very old manifests where id happened to match.
        for hit in hits:
            tid = _target_id(hit.get("id"))
            if tid is None or tid in seeds:
                continue
            left_pt = _float_pt(hit.get("left_px"))
            right_pt = _float_pt(hit.get("right_px"))
            if left_pt is not None and right_pt is not None:
                seeds[tid] = {"left": left_pt, "right": right_pt}

        prev_hit_count = max(prev_hit_count, hit_count)
    return seeds


def _metric_float(record, key):
    metrics = record.get("metrics_timestamps") or {}
    try:
        return float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return None


def _load_summary_reid_delays(run_dir):
    path = run_dir / "trial_summary.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[render] trial_summary read warning: {exc}")
        return {}

    delays = {}
    for target in data.get("targets", []):
        tid = _target_id(target.get("target_id"))
        timing = ((target.get("reid_protocol") or {}).get("timing") or {})
        try:
            delay_s = float(timing["reid_total_time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if tid is not None:
            delays[tid] = max(0.0, delay_s)
    return delays


def _collect_metric_reid_delays(records):
    delays = {}
    active_tid = None
    start_total = None
    for record in records:
        state = record.get("state_name")
        tid = _target_id(record.get("current_target_id"))
        total = _metric_float(record, "total_fine_align_reid_time_s")

        if state == "FINE_ALIGN" and tid is not None:
            if active_tid != tid:
                active_tid = tid
                start_total = total
            continue

        if active_tid is not None:
            if total is not None and start_total is not None and total >= start_total:
                delays.setdefault(active_tid, max(0.0, total - start_total))
            active_tid = None
            start_total = None
    return delays


def _collect_reid_delays(run_dir, records):
    delays = _collect_metric_reid_delays(records)
    delays.update(_load_summary_reid_delays(run_dir))
    return delays


def _collect_survey_flash_start(records):
    for record in records:
        if record.get("planned_target_list"):
            ts = record.get("timestamp_monotonic")
            if ts is not None:
                return float(ts)
    return None


def _collect_fire_start_times(records):
    starts = {}
    for record in records:
        if record.get("state_name") != "FIRE":
            continue
        ts = record.get("timestamp_monotonic")
        if ts is None:
            continue
        tid = _target_id(record.get("current_target_id"))
        if tid is not None:
            starts.setdefault(tid, float(ts))
    return starts


def _reid_segment_key(target_id, timestamp):
    tid = _target_id(target_id)
    if tid is None or timestamp is None:
        return None
    return (tid, round(float(timestamp), 6))


def _collect_reid_segment_starts(records, seed_points):
    segments = []
    prev_state = None
    prev_tid = None
    for record in records:
        state = record.get("state_name")
        tid = _target_id(record.get("current_target_id"))
        ts = record.get("timestamp_monotonic")
        is_segment_start = state == "FINE_ALIGN" and tid is not None and (prev_state != "FINE_ALIGN" or prev_tid != tid)
        if is_segment_start:
            key = _reid_segment_key(tid, ts)
            seed = seed_points.get(tid)
            if key is not None and seed is not None:
                segments.append({
                    "key": key,
                    "record": record,
                    "target_id": tid,
                    "seed": seed,
                })
        prev_state = state
        prev_tid = tid if state == "FINE_ALIGN" else None
    return segments


def _build_render_plan(records, reid_delays):
    plan = []
    counts = {"independent": 0, "reid": 0, "lk": 0}
    prev_state = None
    prev_tid = None
    segment_start_ts = None
    dependent_states = {"FINE_ALIGN", "LOCKED", "FIRE", "FIRED"}

    for record in records:
        state = record.get("state_name") or "?"
        tid = _target_id(record.get("current_target_id"))
        ts = record.get("timestamp_monotonic")

        if state in dependent_states and tid is not None:
            if state == "FINE_ALIGN" and (prev_state not in dependent_states or prev_tid != tid):
                segment_start_ts = float(ts) if ts is not None else None
            elif segment_start_ts is None:
                segment_start_ts = float(ts) if ts is not None else None

            if state == "FINE_ALIGN":
                delay_s = float(reid_delays.get(tid, 0.0))
                elapsed_s = 0.0 if (segment_start_ts is None or ts is None) else max(0.0, float(ts) - float(segment_start_ts))
                has_pd = record.get("pd_error_x_px") is not None and record.get("pd_error_y_px") is not None
                mode = "lk" if has_pd and elapsed_s > (delay_s + REID_CV_FLASH_SEC) else "reid"
            else:
                mode = "lk"
        else:
            segment_start_ts = None
            mode = "independent"

        plan.append({
            "mode": mode,
            "target_id": tid,
            "segment_start_ts": segment_start_ts,
        })
        counts[mode] += 1
        prev_state = state
        prev_tid = tid if state in dependent_states else None

    return plan, counts


def _chunk_render_plan(plan):
    if not plan:
        return []

    chunks = []
    start = 0

    def _chunk_key(entry):
        if entry["mode"] == "lk":
            return (entry["mode"], entry.get("target_id"), entry.get("segment_start_ts"))
        if entry["mode"] == "reid":
            return (entry["mode"], entry.get("target_id"), entry.get("segment_start_ts"))
        return (entry["mode"],)

    current_key = _chunk_key(plan[0])
    for i in range(1, len(plan)):
        key = _chunk_key(plan[i])
        if key != current_key:
            chunks.append({
                "mode": plan[start]["mode"],
                "start": start,
                "end": i,
            })
            start = i
            current_key = key

    chunks.append({
        "mode": plan[start]["mode"],
        "start": start,
        "end": len(plan),
    })
    return chunks


# =============================================================================
# Drawing helpers
# =============================================================================

def _draw_text_lines(frame, lines, x=12, y=28, scale=0.62, color=(0, 255, 255)):
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
        y += int(26 * scale / 0.62)


def _as_int_pt(pt):
    if pt is None or len(pt) < 2:
        return None
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _draw_marker(frame, pt, color, label=None, radius=6):
    p = _as_int_pt(pt)
    if p is None:
        return
    cv2.circle(frame, p, radius, color, -1)
    cv2.circle(frame, p, radius + 4, (255, 255, 255), 1)
    if label:
        cv2.putText(frame, label, (p[0] + 9, p[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, label, (p[0] + 9, p[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _clamp_int(value, lo, hi):
    return int(max(lo, min(hi, round(float(value)))))


def _estimated_box(frame, pt, half=28):
    p = _as_int_pt(pt)
    if p is None:
        return None
    h, w = frame.shape[:2]
    x, y = p
    return (
        _clamp_int(x - half, 0, w - 1),
        _clamp_int(y - half, 0, h - 1),
        _clamp_int(x + half, 0, w - 1),
        _clamp_int(y + half, 0, h - 1),
    )


def _put_text(frame, text, org, scale=0.55, color=(0, 255, 255), thickness=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _label_origin_near(frame, pt, dx=12, dy=-12):
    p = _as_int_pt(pt)
    if p is None:
        return 12, 28
    h, w = frame.shape[:2]
    x = _clamp_int(p[0] + dx, 8, max(8, w - 180))
    y = _clamp_int(p[1] + dy, 22, max(22, h - 10))
    return x, y


def _draw_cv_box(frame, pt, label, color=(0, 220, 255), half=28, confidence=None, winner=False):
    p = _as_int_pt(pt)
    box = _estimated_box(frame, pt, half=half)
    if p is None or box is None:
        return
    x1, y1, x2, y2 = box
    thickness = 3 if winner else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.drawMarker(frame, p, color, cv2.MARKER_CROSS, 18 if winner else 14, 2, cv2.LINE_AA)
    cv2.circle(frame, p, 4 if winner else 3, color, -1, cv2.LINE_AA)
    if label:
        suffix = ""
        if confidence is not None:
            try:
                suffix = f" {float(confidence):.2f}"
            except (TypeError, ValueError):
                suffix = ""
        _put_text(frame, f"{label}{suffix}", _label_origin_near(frame, (x1, y1), dx=0, dy=-6),
                  scale=0.45, color=color)


def _planned_side_candidates(planned_targets, px_key):
    candidates = []
    for planned in planned_targets or []:
        pt = _float_pt(planned.get(px_key))
        tid = _target_id(planned.get("id"))
        if pt is None or tid is None:
            continue
        candidates.append({
            "id": tid,
            "pt": pt,
            "confidence": planned.get("confidence"),
        })
    return candidates


def _hit_for_target(hits, active_id):
    if active_id is None:
        return None
    # Preferred path: explicit target id stored in each hit.
    for hit in hits or []:
        if _target_id(hit.get("target_id")) == active_id:
            return hit

    # Legacy path: id used to mean planned target id.
    for hit in hits or []:
        if _target_id(hit.get("id")) == active_id:
            return hit

    # Fallback for legacy runs with skipped targets where hit ids were compacted.
    if hits:
        return hits[-1]
    return None


def _planned_target_for_id(record, active_id):
    if active_id is None:
        return None
    for target in record.get("planned_target_list") or []:
        if _target_id(target.get("id")) == active_id:
            return target
    return None


def _active_target_class_id(record, active_id):
    target = _planned_target_for_id(record, active_id)
    if target is None:
        return None
    cls = target.get("class_id")
    try:
        return int(cls) if cls is not None else None
    except (TypeError, ValueError):
        return None


def _draw_strike_marker(frame, pt, label, color=(0, 0, 235), fallback_center=False):
    p = _as_int_pt(pt)
    if p is None and fallback_center:
        h, w = frame.shape[:2]
        p = (w // 2, h // 2)
    if p is None:
        return

    radius = 16
    cv2.circle(frame, p, radius, color, 2, cv2.LINE_AA)
    cv2.line(frame, (p[0] - radius - 5, p[1]), (p[0] - 5, p[1]), color, 2, cv2.LINE_AA)
    cv2.line(frame, (p[0] + 5, p[1]), (p[0] + radius + 5, p[1]), color, 2, cv2.LINE_AA)
    cv2.line(frame, (p[0], p[1] - radius - 5), (p[0], p[1] - 5), color, 2, cv2.LINE_AA)
    cv2.line(frame, (p[0], p[1] + 5), (p[0], p[1] + radius + 5), color, 2, cv2.LINE_AA)
    _put_text(frame, label, _label_origin_near(frame, p, dx=18, dy=-18), scale=0.62, color=color, thickness=2)


def _fire_remaining_s(record, fire_start_times):
    ts = record.get("timestamp_monotonic")
    tid = _target_id(record.get("current_target_id"))
    if ts is None or tid is None:
        return None
    start_ts = fire_start_times.get(tid, float(ts))
    elapsed = max(0.0, float(ts) - float(start_ts))
    return max(0.0, float(LASER_FIRE_DURATION_SEC) - elapsed)


def _draw_fire_countdown(frame, remaining_s, target_id):
    if remaining_s is None:
        return
    total = max(0.001, float(LASER_FIRE_DURATION_SEC))
    frac = max(0.0, min(1.0, remaining_s / total))
    h, w = frame.shape[:2]
    panel_w = min(330, max(240, w // 3))
    panel_h = 70
    x0 = (w - panel_w) // 2
    y0 = h - panel_h - 18
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (24, 20, 18), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 80, 255), 2, cv2.LINE_AA)

    tgt = f"  T{target_id}" if target_id is not None else ""
    _put_text(frame, f"FIRING: {remaining_s:.2f}s{tgt}", (x0 + 14, y0 + 30),
              scale=0.78, color=(0, 120, 255), thickness=2)
    bar_x0, bar_y0 = x0 + 14, y0 + 46
    bar_x1, bar_y1 = x1 - 14, y0 + 58
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (90, 90, 90), 1, cv2.LINE_AA)
    fill_w = int((bar_x1 - bar_x0) * frac)
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + fill_w, bar_y1), (0, 80, 255), -1)


def _new_pd_flow_state():
    return {
        "target_id": None,
        "segment_start_ts": None,
        "initialized": False,
        "old_gray_left": None,
        "old_gray_right": None,
        "pt_left": None,
        "pt_right": None,
        "trail_left": [],
        "trail_right": [],
        "reid_detected": False,
        "reid_dets_left": [],
        "reid_dets_right": [],
        "reid_winner_idx_left": None,
        "reid_winner_idx_right": None,
    }


def _reset_pd_flow_state(state, target_id=None, timestamp=None):
    state.clear()
    state.update(_new_pd_flow_state())
    state["target_id"] = target_id
    state["segment_start_ts"] = timestamp


def _draw_cv_flash(frame, pt, label, phase):
    p = _as_int_pt(pt)
    if p is None:
        return
    x, y = p
    box = 24
    ring = 14 + (phase % 4) * 4
    color = (0, 220, 255)
    cv2.rectangle(frame, (x - box, y - box), (x + box, y + box), color, 2, cv2.LINE_AA)
    cv2.circle(frame, p, ring, color, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, p, (80, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
    _put_text(frame, label, (x + 12, max(18, y - 30)), scale=0.48, color=color)


def _crop_with_padding(frame, cx, cy, half):
    h, w = frame.shape[:2]
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)
    crop = frame[src_y0:src_y1, src_x0:src_x1]
    if crop.size == 0:
        crop = np.zeros((2 * half, 2 * half, 3), dtype=np.uint8)

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    rel_x = float(cx - x0)
    rel_y = float(cy - y0)
    return crop, rel_x, rel_y


def _draw_zoom_inset(frame, raw_frame, pt, title, color=(0, 255, 255), err=None, anchor="bottom_right"):
    p = _as_int_pt(pt)
    if p is None:
        return

    size = min(PD_MINIMAP_SIZE, max(96, frame.shape[1] // 5))
    crop, rel_x, rel_y = _crop_with_padding(raw_frame, p[0], p[1], half=56)
    inset = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    sx = size / max(1, crop.shape[1])
    sy = size / max(1, crop.shape[0])
    ip = (_clamp_int(rel_x * sx, 0, size - 1), _clamp_int(rel_y * sy, 0, size - 1))
    cv2.line(inset, (ip[0], 0), (ip[0], size - 1), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(inset, (0, ip[1]), (size - 1, ip[1]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(inset, ip, 9, color, 2, cv2.LINE_AA)

    h, w = frame.shape[:2]
    margin = 10
    title_h = 24
    if anchor == "top_right":
        x0 = w - size - margin
        y0 = 62
    else:
        x0 = w - size - margin
        y0 = h - size - title_h - 62
    x0 = max(margin, x0)
    y0 = max(margin, y0)
    x1 = min(w - margin, x0 + size)
    y1 = min(h - margin, y0 + title_h + size)
    if x1 - x0 != size or y1 - y0 != title_h + size:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)
    cv2.rectangle(frame, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), color, 1, cv2.LINE_AA)
    _put_text(frame, title, (x0 + 6, y0 + 17), scale=0.43, color=color)
    frame[y0 + title_h:y0 + title_h + size, x0:x0 + size] = inset
    if err is not None:
        ex, ey = err
        _put_text(frame, f"{ex:+.1f}, {ey:+.1f}px", (x0 + 6, y1 - 8), scale=0.42, color=color)


def _draw_flow_tracker(frame, trail, pt, label, color, flash=False, phase=0):
    p = _as_int_pt(pt)
    if p is None:
        return

    pts = [_as_int_pt(t) for t in trail]
    pts = [t for t in pts if t is not None]
    if len(pts) >= 2:
        for i in range(1, len(pts)):
            frac = i / max(1, len(pts) - 1)
            seg_color = tuple(int(c * (0.45 + 0.55 * frac)) for c in color)
            cv2.line(frame, pts[i - 1], pts[i], seg_color, 2, cv2.LINE_AA)
        cv2.arrowedLine(frame, pts[-2], pts[-1], color, 2, cv2.LINE_AA, tipLength=0.35)

    if flash:
        _draw_cv_flash(frame, pt, f"CV {label}", phase)
    cv2.circle(frame, p, 8, color, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, p, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    _put_text(frame, f"LK {label}", (p[0] + 12, min(frame.shape[0] - 12, p[1] + 24)), scale=0.46, color=color)


def _draw_pd_target_marker(frame, label=None, color=(255, 60, 0)):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    arm = 18
    gap = 7
    cv2.circle(frame, (cx, cy), 8, color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx - arm - gap, cy), (cx - gap, cy), color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + arm + gap, cy), color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm - gap), (cx, cy - gap), color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + arm + gap), color, 2, cv2.LINE_AA)
    if label:
        _put_text(frame, label, _label_origin_near(frame, (cx, cy), dx=20, dy=-16), scale=0.44, color=color)
    return (cx, cy)


def _draw_target_drive_arrow(frame, current_pt, target_pt, color=(255, 60, 0)):
    p = _as_int_pt(current_pt)
    t = _as_int_pt(target_pt)
    if p is None or t is None:
        return
    if abs(p[0] - t[0]) + abs(p[1] - t[1]) < 3:
        return
    cv2.arrowedLine(frame, p, t, color, 2, cv2.LINE_AA, tipLength=0.18)


def _draw_continuous_stereo_guides(video_row, left_width, record):
    state = record.get("state_name") or "?"
    if state not in ("FINE_ALIGN", "LOCKED", "FIRE", "FIRED"):
        return

    h, w = video_row.shape[:2]
    right_offset_x = int(left_width)
    pd_y = h // 2
    pd_color = (255, 60, 0)

    # This is the actual PD target row used by the controller: frame center in both cameras.
    cv2.line(video_row, (0, pd_y), (w - 1, pd_y), pd_color, 1, cv2.LINE_AA)
    cv2.drawMarker(video_row, (left_width // 2, pd_y), pd_color, cv2.MARKER_TILTED_CROSS, 14, 1, cv2.LINE_AA)
    cv2.drawMarker(video_row, (right_offset_x + (w - right_offset_x) // 2, pd_y), pd_color, cv2.MARKER_TILTED_CROSS, 14, 1, cv2.LINE_AA)
    _put_text(video_row, "PD target row", (12, max(24, pd_y - 10)), scale=0.42, color=pd_color)

    # Also connect the planned survey/reference point continuously across both views.
    active_id = _target_id(record.get("current_target_id"))
    planned = _planned_target_for_id(record, active_id)
    left_ref = _as_int_pt((planned or {}).get("left_px"))
    right_ref = _as_int_pt((planned or {}).get("right_px"))
    ref_color = (0, 165, 255)
    if left_ref is not None and right_ref is not None:
        lp = left_ref
        rp = (right_offset_x + right_ref[0], right_ref[1])
        cv2.line(video_row, (0, lp[1]), lp, ref_color, 1, cv2.LINE_AA)
        cv2.line(video_row, lp, rp, ref_color, 2, cv2.LINE_AA)
        cv2.line(video_row, rp, (w - 1, rp[1]), ref_color, 1, cv2.LINE_AA)
        cv2.drawMarker(video_row, lp, ref_color, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        cv2.drawMarker(video_row, rp, ref_color, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        _put_text(video_row, "survey ref", (12, max(44, min(lp[1], rp[1]) - 8)), scale=0.42, color=ref_color)


def _reid_crop_boxes():
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    half = max(1, int(FINE_ALIGN_REID_CROP_HALF_PX))
    box = (
        max(0, cx - half),
        max(0, cy - half),
        min(FRAME_WIDTH, cx + half),
        min(FRAME_HEIGHT, cy + half),
    )
    return box, box


def _draw_reid_crop_view(frame, raw_frame, crop_box, seed_pt, title, color, candidates=None, winner_id=None):
    x0, y0, x1, y1 = crop_box
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    _put_text(frame, title, (x0 + 6, max(22, y0 - 8)), scale=0.48, color=color)
    crop_candidates = []
    for candidate in candidates or []:
        pt = candidate.get("pt")
        p = _as_int_pt(pt)
        if p is None:
            continue
        if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
            continue
        is_winner = winner_id is not None and candidate.get("id") == winner_id
        crop_candidates.append((candidate, p, is_winner))
        if is_winner and seed_pt is not None:
            continue
        _draw_cv_box(
            frame,
            pt,
            "winner" if is_winner else None,
            color=color if is_winner else (70, 220, 255),
            half=22 if is_winner else 16,
            confidence=None,
            winner=is_winner,
        )
    if seed_pt is not None:
        _draw_cv_box(frame, seed_pt, "winner", color=color, half=22, winner=True)

    crop = raw_frame[y0:y1, x0:x1]
    if crop.size == 0:
        return
    h, w = frame.shape[:2]
    inset_w = min(230, max(150, w // 4))
    inset_h = max(90, int(inset_w * crop.shape[0] / max(1, crop.shape[1])))
    inset_h = min(inset_h, 180)
    inset = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)
    if seed_pt is not None:
        sx = inset_w / max(1, crop.shape[1])
        sy = inset_h / max(1, crop.shape[0])
        for candidate, p, is_winner in crop_candidates:
            if is_winner and seed_pt is not None:
                continue
            px = _clamp_int((p[0] - x0) * sx, 0, inset_w - 1)
            py = _clamp_int((p[1] - y0) * sy, 0, inset_h - 1)
            c = color if is_winner else (70, 220, 255)
            half = 18 if is_winner else 12
            cv2.rectangle(inset, (max(0, px - half), max(0, py - half)),
                          (min(inset_w - 1, px + half), min(inset_h - 1, py + half)), c, 1, cv2.LINE_AA)
            cv2.drawMarker(inset, (px, py), c, cv2.MARKER_CROSS, 11, 1, cv2.LINE_AA)

        px = _clamp_int((seed_pt[0] - x0) * sx, 0, inset_w - 1)
        py = _clamp_int((seed_pt[1] - y0) * sy, 0, inset_h - 1)
        cv2.rectangle(inset, (max(0, px - 18), max(0, py - 18)),
                      (min(inset_w - 1, px + 18), min(inset_h - 1, py + 18)), color, 2, cv2.LINE_AA)
        cv2.drawMarker(inset, (px, py), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    ox = w - inset_w - 12
    oy = 66
    overlay = frame.copy()
    cv2.rectangle(overlay, (ox - 4, oy - 28), (ox + inset_w + 4, oy + inset_h + 4), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (ox - 4, oy - 28), (ox + inset_w + 4, oy + inset_h + 4), color, 1, cv2.LINE_AA)
    _put_text(frame, title, (ox + 6, oy - 9), scale=0.45, color=color)
    frame[oy:oy + inset_h, ox:ox + inset_w] = inset


def _yolo_detections_in_box(core, raw_frame, crop_box, conf, class_filter=None):
    """Run YOLO on the crop region; return full-frame detection dicts {center, box, conf}.

    Uses the active target class when available so re-ID visuals match live behavior.
    """
    bx0, by0, bx1, by1 = crop_box
    crop = raw_frame[by0:by1, bx0:bx1]
    if crop.size == 0:
        return []
    classes_arg = None
    if class_filter is not None:
        try:
            classes_arg = [int(class_filter)]
        except (TypeError, ValueError):
            classes_arg = None
    try:
        results = core.yolo(
            crop,
            imgsz=640,
            verbose=False,
            conf=conf,
            retina_masks=False,
            classes=classes_arg,
            **core._yolo_predict_kwargs(),
        )
        raw_boxes = results[0].boxes if results and results[0].boxes is not None else []
    except Exception:
        return []
    dets = []
    for b in raw_boxes:
        xyxy = b.xyxy[0]
        x1, y1 = int(xyxy[0]) + bx0, int(xyxy[1]) + by0
        x2, y2 = int(xyxy[2]) + bx0, int(xyxy[3]) + by0
        dets.append({
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "box": (x1, y1, x2, y2),
            "conf": float(b.conf[0]),
            "cls": int(b.cls[0]) if getattr(b, "cls", None) is not None else None,
        })
    return dets


def _filtered_boxes_to_full_frame_dets(filtered_boxes, crop_box):
    bx0, by0, _, _ = crop_box
    dets = []
    for b in filtered_boxes or []:
        xyxy = b.xyxy[0].cpu().numpy()
        x1 = int(round(float(xyxy[0]))) + bx0
        y1 = int(round(float(xyxy[1]))) + by0
        x2 = int(round(float(xyxy[2]))) + bx0
        y2 = int(round(float(xyxy[3]))) + by0
        dets.append({
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "box": (x1, y1, x2, y2),
            "conf": float(b.conf[0]),
            "cls": int(b.cls[0]) if getattr(b, "cls", None) is not None else None,
        })
    return dets


def _winner_idx_from_seed(dets, seed_pt):
    sp = _as_int_pt(seed_pt)
    if not dets or sp is None:
        return None
    return min(
        range(len(dets)),
        key=lambda i: (dets[i]["center"][0] - sp[0]) ** 2 + (dets[i]["center"][1] - sp[1]) ** 2,
    )


def _precompute_reid_overlay_cache(run_dir, records, seed_points, detector, imgsz, conf):
    segments = _collect_reid_segment_starts(records, seed_points)
    if detector is None or not segments:
        return {}

    left_box, right_box = _reid_crop_boxes()
    left_jobs = []
    right_jobs = []
    skipped = 0
    for seg in segments:
        record = seg["record"]
        raw_left = cv2.imread(str(run_dir / record["left_image_path"]))
        raw_right = cv2.imread(str(run_dir / record["right_image_path"]))
        if raw_left is None or raw_right is None:
            skipped += 1
            continue
        lx0, ly0, lx1, ly1 = left_box
        rx0, ry0, rx1, ry1 = right_box
        class_filter = _active_target_class_id(record, seg["target_id"])
        left_jobs.append({
            "key": seg["key"],
            "crop": raw_left[ly0:ly1, lx0:lx1].copy(),
            "crop_box": left_box,
            "seed": seg["seed"].get("left"),
            "class_filter": class_filter,
        })
        right_jobs.append({
            "key": seg["key"],
            "crop": raw_right[ry0:ry1, rx0:rx1].copy(),
            "crop_box": right_box,
            "seed": seg["seed"].get("right"),
            "class_filter": class_filter,
        })

    def _run_side_jobs(side_name, core, jobs):
        side_results = {}
        grouped = {}
        total = len(jobs)
        for job in jobs:
            grouped.setdefault(job["class_filter"], []).append(job)

        done = 0
        for class_filter, bucket in grouped.items():
            crops = [job["crop"] for job in bucket]
            batched = core._get_filtered_results_batch(
                crops,
                classes_override=class_filter,
                conf_override=conf,
                imgsz=imgsz,
            )
            for job, (boxes, _masks) in zip(bucket, batched):
                dets = _filtered_boxes_to_full_frame_dets(boxes, job["crop_box"])
                side_results[job["key"]] = {
                    "dets": dets,
                    "winner_idx": _winner_idx_from_seed(dets, job["seed"]),
                }
                done += 1
            class_label = "all" if class_filter is None else str(class_filter)
            print(f"[render prep] {side_name} class={class_label} {done}/{total} segment(s)")
        return side_results

    started = time.perf_counter()
    workers = max(1, int(REID_PRECOMPUTE_SIDE_WORKERS or 1))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(2, workers), thread_name_prefix="render-reid") as pool:
            fut_left = pool.submit(_run_side_jobs, "left", detector.cv_left, left_jobs)
            fut_right = pool.submit(_run_side_jobs, "right", detector.cv_right, right_jobs)
            left_results = fut_left.result()
            right_results = fut_right.result()
    else:
        left_results = _run_side_jobs("left", detector.cv_left, left_jobs)
        right_results = _run_side_jobs("right", detector.cv_right, right_jobs)

    cache = {}
    for seg in segments:
        key = seg["key"]
        cache[key] = {
            "reid_dets_left": (left_results.get(key) or {}).get("dets", []),
            "reid_winner_idx_left": (left_results.get(key) or {}).get("winner_idx"),
            "reid_dets_right": (right_results.get(key) or {}).get("dets", []),
            "reid_winner_idx_right": (right_results.get(key) or {}).get("winner_idx"),
        }

    print(
        f"[render prep] cached RE-ID detections for {len(cache)}/{len(segments)} segment(s) "
        f"in {time.perf_counter() - started:.2f}s"
        + (f" ({skipped} skipped missing frame)" if skipped else "")
    )
    return cache


def _load_frame_pair(run_dir, record):
    left = cv2.imread(str(run_dir / record["left_image_path"]))
    right = cv2.imread(str(run_dir / record["right_image_path"]))
    if left is None or right is None:
        return None, None, None, None
    return left, right, left.copy(), right.copy()


def _compose_rendered_frame(run_dir, record, plan_entry, ctx, pd_flow_state=None):
    left, right, raw_left, raw_right = _load_frame_pair(run_dir, record)
    if left is None or right is None:
        return None

    lk_pt_left = pd_flow_state.get("pt_left") if pd_flow_state is not None else None
    lk_pt_right = pd_flow_state.get("pt_right") if pd_flow_state is not None else None
    _draw_manifest_overlays(
        left,
        right,
        record,
        ctx["first_ts"],
        survey_flash_start_ts=ctx["survey_flash_start_ts"],
        fire_start_times=ctx["fire_start_times"],
        lk_pt_left=lk_pt_left,
        lk_pt_right=lk_pt_right,
    )

    if ctx["pd_flow_overlay"] and (pd_flow_state is not None or plan_entry.get("mode") == "reid"):
        state = pd_flow_state
        if state is None:
            state = _new_pd_flow_state()
            state["target_id"] = plan_entry.get("target_id")
            state["segment_start_ts"] = plan_entry.get("segment_start_ts")
        _draw_pd_flow_overlay(
            left,
            right,
            raw_left,
            raw_right,
            record,
            ctx["pd_seed_points"],
            ctx["pd_reid_delays"],
            state,
            detector=ctx["detector"],
            yolo_conf=ctx["yolo_conf"],
            reid_overlay_cache=ctx["reid_overlay_cache"],
        )

    if left.shape[0] != right.shape[0]:
        right = cv2.resize(right, (right.shape[1], left.shape[0]))
    video_row = np.hstack([left, right])
    _draw_continuous_stereo_guides(video_row, left.shape[1], record)

    header = np.zeros((HEADER_H, video_row.shape[1], 3), dtype=np.uint8)
    combined = np.vstack([header, video_row])

    elapsed = 0.0
    if record.get("timestamp_monotonic") is not None:
        elapsed = float(record["timestamp_monotonic"]) - float(ctx["first_ts"])
    n_hits = len(record.get("hit_targets_so_far") or [])
    state_name = record.get("state_name") or "?"
    _draw_progress_header(
        combined,
        elapsed,
        ctx["total_duration"],
        n_hits,
        ctx["total_targets"],
        state_name,
        ctx["timeline_segments"],
    )
    return combined


def _draw_reid_overlay(left, right, raw_left, raw_right, record, active_id, seed,
                       segment_elapsed, reid_delay, detector=None, yolo_conf=None, state=None,
                       precomputed=None):
    left_box, right_box = _reid_crop_boxes()
    left_seed  = seed.get("left")  if seed else None
    right_seed = seed.get("right") if seed else None
    label = f"RE-ID T{active_id}  {segment_elapsed:.1f}/{max(reid_delay, 0.01):.1f}s"

    # Draw crop region outline + label; winner seed marker drawn after detection below.
    for frame, box in ((left, left_box), (right, right_box)):
        x0, y0, x1, y1 = box
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 80), 2, cv2.LINE_AA)
        _put_text(frame, label, (x0 + 6, max(22, y0 - 8)), scale=0.48, color=(0, 220, 80))

    if detector is None or state is None:
        # No YOLO — fall back to seed marker only.
        for frame, seed_pt in ((left, left_seed), (right, right_seed)):
            if seed_pt is not None:
                _draw_cv_box(frame, seed_pt, "winner", color=(0, 220, 80), half=22, winner=True)
        return

    # Detect once on the first RE-ID frame; cache for the whole segment.
    if not state.get("reid_detected"):
        if precomputed is not None:
            state["reid_dets_left"] = list(precomputed.get("reid_dets_left") or [])
            state["reid_winner_idx_left"] = precomputed.get("reid_winner_idx_left")
            state["reid_dets_right"] = list(precomputed.get("reid_dets_right") or [])
            state["reid_winner_idx_right"] = precomputed.get("reid_winner_idx_right")
        else:
            conf = yolo_conf if yolo_conf is not None else YOLO_CONF

            def _detect_side(core, raw_frame, box, seed_pt):
                dets = _yolo_detections_in_box(
                    core,
                    raw_frame,
                    box,
                    conf,
                    class_filter=_active_target_class_id(record, active_id),
                )
                return dets, _winner_idx_from_seed(dets, seed_pt)

            state["reid_dets_left"],  state["reid_winner_idx_left"]  = _detect_side(
                detector.cv_left,  raw_left,  left_box,  left_seed)
            state["reid_dets_right"], state["reid_winner_idx_right"] = _detect_side(
                detector.cv_right, raw_right, right_box, right_seed)
        state["reid_detected"] = True

    # Draw all detections: winner in green, losers in red.
    for frame, dets_key, winner_key in (
        (left,  "reid_dets_left",  "reid_winner_idx_left"),
        (right, "reid_dets_right", "reid_winner_idx_right"),
    ):
        dets = state.get(dets_key) or []
        winner_idx = state.get(winner_key)
        for i, det in enumerate(dets):
            is_winner = (i == winner_idx)
            color = (0, 220, 80) if is_winner else (0, 0, 220)
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_winner else 2, cv2.LINE_AA)
            cx, cy = det["center"]
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cls_tag = ""
            if det.get("cls") is not None:
                cls_tag = f" c{int(det['cls'])}"
            tag = f"{'winner' if is_winner else 'loser'} {det['conf']:.2f}{cls_tag}"
            _put_text(frame, tag, (x1, max(14, y1 - 4)), scale=0.42, color=color)


def _draw_pd_flow_overlay(left, right, raw_left, raw_right, record, seed_points, reid_delays, state,
                          detector=None, yolo_conf=None, reid_overlay_cache=None):
    active_id = _target_id(record.get("current_target_id"))
    timestamp = record.get("timestamp_monotonic")

    # Reset only when the tracked target changes — not just because we left FINE_ALIGN.
    # This keeps pt_left/right alive so the fire marker can use the last LK position.
    if active_id != state.get("target_id"):
        _reset_pd_flow_state(
            state, active_id,
            float(timestamp) if timestamp is not None else None,
        )

    if record.get("state_name") != "FINE_ALIGN":
        return

    if active_id is None or timestamp is None:
        return
    timestamp = float(timestamp)

    if state.get("target_id") != active_id:
        _reset_pd_flow_state(state, active_id, timestamp)

    seed = seed_points.get(active_id)
    if seed is None:
        return

    segment_start = state.get("segment_start_ts")
    segment_elapsed = 0.0 if segment_start is None else max(0.0, timestamp - float(segment_start))
    reid_delay = float(reid_delays.get(active_id, 0.0))
    pd_target_left = _draw_pd_target_marker(left, label="PD target")
    pd_target_right = _draw_pd_target_marker(right, label="PD target")

    # RE-ID phase + brief flash after it ends.
    if segment_elapsed <= reid_delay + REID_CV_FLASH_SEC:
        cache_key = _reid_segment_key(active_id, state.get("segment_start_ts"))
        precomputed = (reid_overlay_cache or {}).get(cache_key) if cache_key is not None else None
        _draw_reid_overlay(left, right, raw_left, raw_right, record, active_id, seed,
                           segment_elapsed, reid_delay, detector=detector, yolo_conf=yolo_conf,
                           state=state, precomputed=precomputed)
        if segment_elapsed < reid_delay:
            return  # still in RE-ID, skip PD display

    # PD phase: use LK to track the visual plant position frame-to-frame.
    # The manifest PD error values are accurate (from the live run); LK here
    # is only for keeping the on-screen marker on the plant as the gantry jogs.
    seed_left  = seed["left"]
    seed_right = seed["right"]

    gray_left  = cv2.cvtColor(raw_left,  cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(raw_right, cv2.COLOR_BGR2GRAY)

    def _lk_track(old_gray, new_gray, pt):
        arr = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        new_pt, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, arr, None, **_LK_PARAMS)
        if st is None or st[0][0] == 0:
            return pt
        nx, ny = float(new_pt[0, 0, 0]), float(new_pt[0, 0, 1])
        h, w = new_gray.shape[:2]
        return (nx, ny) if (0 <= nx < w and 0 <= ny < h) else pt

    if not state.get("initialized"):
        state["pt_left"]       = seed_left
        state["pt_right"]      = seed_right
        state["old_gray_left"] = gray_left
        state["old_gray_right"] = gray_right
        state["initialized"]   = True
    else:
        state["pt_left"]  = _lk_track(state["old_gray_left"],  gray_left,  state["pt_left"])
        state["pt_right"] = _lk_track(state["old_gray_right"], gray_right, state["pt_right"])
        state["old_gray_left"]  = gray_left
        state["old_gray_right"] = gray_right

    left_pt  = state["pt_left"]
    right_pt = state["pt_right"]

    # Accumulate trail; cap at PD_FLOW_TRAIL_LEN.
    state["trail_left"].append(left_pt)
    state["trail_right"].append(right_pt)
    if len(state["trail_left"]) > PD_FLOW_TRAIL_LEN:
        state["trail_left"] = state["trail_left"][-PD_FLOW_TRAIL_LEN:]
        state["trail_right"] = state["trail_right"][-PD_FLOW_TRAIL_LEN:]

    ex = record.get("pd_error_x_px")
    ey = record.get("pd_error_y_px")
    pd_locked = record.get("pd_locked")
    pd_settle = record.get("pd_settle_count")

    has_pd_data = ex is not None and ey is not None
    if has_pd_data:
        try:
            ex, ey = float(ex), float(ey)
        except (TypeError, ValueError):
            has_pd_data = False

    color_l = (120, 255, 120) if pd_locked else (0, 255, 255)
    color_r = (120, 255, 180) if pd_locked else (0, 255, 180)

    _draw_target_drive_arrow(left, left_pt, pd_target_left)
    _draw_target_drive_arrow(right, right_pt, pd_target_right)
    _draw_flow_tracker(left,  state["trail_left"],  left_pt,  "box center", color_l)
    _draw_flow_tracker(right, state["trail_right"], right_pt, "box center", color_r)
    _draw_zoom_inset(left,  raw_left,  left_pt,  "PD zoom L", color=color_l)
    _draw_zoom_inset(right, raw_right, right_pt, "PD zoom R", color=color_r)

    if has_pd_data:
        status = "DEADZONE" if pd_locked else "CORRECTING"
        settle_str = f"  settle {int(pd_settle)}" if pd_settle is not None else ""
        for frame, color in ((left, color_l), (right, color_r)):
            h, w = frame.shape[:2]
            x0, y0 = 12, max(92, h - 100)
            x1, y1 = min(w - 12, x0 + 316), y0 + 72
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 18, 18), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
            _put_text(frame, f"PD T{active_id}  {status}{settle_str}",
                      (x0 + 8, y0 + 20), scale=0.52, color=color)
            _put_text(frame, f"ex {ex:+5.1f}px",
                      (x0 + 8, y0 + 50), scale=0.70, color=(255, 255, 255), thickness=2)
            _put_text(frame, f"ey {ey:+5.1f}px",
                      (x0 + 166, y0 + 50), scale=0.70, color=(255, 255, 255), thickness=2)


def _draw_progress_header(canvas, elapsed, total_duration, n_hits, total_targets, current_state, timeline_segments):
    _, w = canvas.shape[:2]
    pad = 4

    # ── Full-width state timeline bar ──────────────────────────────────────
    tl_h  = 28
    tl_y0 = pad
    tl_y1 = tl_y0 + tl_h
    cv2.rectangle(canvas, (pad, tl_y0), (w - pad, tl_y1), (15, 15, 15), -1)
    tbar_w = (w - 2 * pad)
    for fs, fe, seg_state in timeline_segments:
        sx = pad + int(fs * tbar_w)
        ex = pad + int(fe * tbar_w)
        if ex <= sx:
            ex = sx + 1
        color = _STATE_COLORS.get(seg_state, _DEFAULT_STATE_COLOR)
        cv2.rectangle(canvas, (sx, tl_y0), (ex, tl_y1), color, -1)
    if total_duration > 0:
        cur_x = pad + int((elapsed / total_duration) * tbar_w)
        cv2.line(canvas, (cur_x, tl_y0 - 1), (cur_x, tl_y1 + 1), (255, 255, 255), 2)
    cv2.rectangle(canvas, (pad, tl_y0), (w - pad, tl_y1), (110, 110, 110), 1)
    tlbl = f"  T+{elapsed:.1f}s  [{current_state}]"
    text_y = tl_y1 - 7
    cv2.putText(canvas, tlbl, (pad + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
    cv2.putText(canvas, tlbl, (pad + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 210, 255), 1)

    # ── Tiny target-progress bar (left quarter, below timeline) ────────────
    tg_h  = 8
    tg_y0 = tl_y1 + 3
    tg_y1 = tg_y0 + tg_h
    tg_x1 = w // 4
    cv2.rectangle(canvas, (pad, tg_y0), (tg_x1, tg_y1), (25, 25, 25), -1)
    if total_targets > 0:
        fill = int((tg_x1 - pad) * n_hits / total_targets)
        cv2.rectangle(canvas, (pad, tg_y0), (pad + fill, tg_y1), (45, 185, 45), -1)
    cv2.rectangle(canvas, (pad, tg_y0), (tg_x1, tg_y1), (90, 90, 90), 1)
    lbl = f"  {n_hits}/{total_targets if total_targets else '?'}"
    cv2.putText(canvas, lbl, (pad + 4, tg_y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2)
    cv2.putText(canvas, lbl, (pad + 4, tg_y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 255, 180), 1)


def _draw_manifest_overlays(left, right, record, first_ts, survey_flash_start_ts=None,
                            fire_start_times=None, lk_pt_left=None, lk_pt_right=None):
    elapsed = 0.0
    if record.get("timestamp_monotonic") is not None and first_ts is not None:
        elapsed = float(record["timestamp_monotonic"]) - float(first_ts)

    state = record.get("state_name") or "?"
    target_id = record.get("current_target_id")
    active_id = int(target_id) if target_id is not None else None
    hits = record.get("hit_targets_so_far") or []
    n_hits = len(hits)

    lines = [f"T+{elapsed:.2f}s  state={state}"]
    if target_id is not None:
        lines.append(f"target={target_id}  fired={n_hits}")
    _draw_text_lines(left, lines)
    _draw_text_lines(right, lines)

    ts = record.get("timestamp_monotonic")
    survey_flash = (
        ts is not None
        and survey_flash_start_ts is not None
        and 0.0 <= (float(ts) - float(survey_flash_start_ts)) <= SURVEY_CV_FLASH_SEC
        and bool(record.get("planned_target_list"))
        and state not in ("FINE_ALIGN", "LOCKED", "FIRE", "FIRED")
    )
    if survey_flash:
        crop_overlay = _get_survey_crop_overlay_rects(left.shape[1], left.shape[0])
        mode = crop_overlay.get("mode")
        lrect = crop_overlay.get("left_rect")
        rrect = crop_overlay.get("right_rect")
        reason = crop_overlay.get("reason")
        if lrect is not None and rrect is not None:
            lx0, ly0, lx1, ly1 = lrect
            rx0, ry0, rx1, ry1 = rrect
            cv2.rectangle(left, (lx0, ly0), (lx1, ly1), (250, 180, 0), 2, cv2.LINE_AA)
            cv2.rectangle(right, (rx0, ry0), (rx1, ry1), (250, 180, 0), 2, cv2.LINE_AA)
            _put_text(left, f"survey crop: {mode}", (12, 84), scale=0.52, color=(250, 180, 0), thickness=1)
            _put_text(right, f"survey crop: {mode}", (12, 84), scale=0.52, color=(250, 180, 0), thickness=1)
            if reason and mode == "centered":
                _put_text(left, f"fallback: {reason}", (12, 104), scale=0.45, color=(180, 200, 255), thickness=1)

        hit_ids = {h.get("id") for h in hits}
        for planned in record.get("planned_target_list") or []:
            pid = planned.get("id")
            if pid in hit_ids:
                continue
            label = f"survey T{pid}"
            _draw_cv_box(left, planned.get("left_px"), label, color=(0, 220, 255),
                         half=30, confidence=planned.get("confidence"))
            _draw_cv_box(right, planned.get("right_px"), label, color=(0, 220, 255),
                         half=30, confidence=planned.get("confidence"))
        _put_text(left, "SURVEY CV: boxes + box-center points", (12, left.shape[0] - 20),
                  scale=0.58, color=(0, 220, 255), thickness=1)
        _put_text(right, "SURVEY CV: boxes + box-center points", (12, right.shape[0] - 20),
                  scale=0.58, color=(0, 220, 255), thickness=1)

    # Fine-align: show the active survey-reference point in each camera.
    if state == "FINE_ALIGN":
        planned = _planned_target_for_id(record, active_id)
        left_ref = _float_pt((planned or {}).get("left_px"))
        right_ref = _float_pt((planned or {}).get("right_px"))

        for frame, ref_pt in ((left, left_ref), (right, right_ref)):
            if ref_pt is not None:
                cx = int(round(float(ref_pt[0])))
                cy = int(round(float(ref_pt[1])))
                cx = max(0, min(frame.shape[1] - 1, cx))
                cy = max(0, min(frame.shape[0] - 1, cy))
                cv2.drawMarker(frame, (cx, cy), (0, 165, 255), cv2.MARKER_TILTED_CROSS, 13, 1, cv2.LINE_AA)
                _put_text(frame, "survey ref", _label_origin_near(frame, (cx, cy), dx=10, dy=-10), scale=0.42, color=(0, 165, 255))
            if _CROP_X0 > 0 or _CROP_Y0 > 0:
                cv2.rectangle(frame, (_CROP_X0, _CROP_Y0), (_CROP_X1, _CROP_Y1), (255, 200, 0), 2)

    # Fire / locked: compact reticle at the recorded local hit point.
    elif state in ("FIRE", "FIRED", "LOCKED"):
        hit = _hit_for_target(hits, active_id)
        tgt_str = f"T{active_id}" if active_id is not None else "?"
        remaining_s = None
        if state == "LOCKED":
            label = f"LOCKED {tgt_str}"
            color = (60, 230, 60)
        elif state == "FIRE":
            remaining_s = _fire_remaining_s(record, fire_start_times or {})
            countdown = f"{remaining_s:.2f}s" if remaining_s is not None else "--"
            label = f"FIRING: {countdown}"
            color = (0, 80, 255)
        else:
            label = f"FIRED {tgt_str}"
            color = (0, 0, 235)
        left_pt  = lk_pt_left  or (hit.get("left_px")  if hit else None)
        right_pt = lk_pt_right or (hit.get("right_px") if hit else None)
        _draw_strike_marker(left,  left_pt,  label, color=color, fallback_center=True)
        _draw_strike_marker(right, right_pt, label, color=color, fallback_center=True)
        if state == "FIRE":
            _draw_fire_countdown(left, remaining_s, active_id)
            _draw_fire_countdown(right, remaining_s, active_id)

    # Travel: just a label, no pixel markers.
    elif state in ("TRAVEL", "POST_TRAVEL", "TARGET"):
        if not survey_flash:
            tgt_str = f"target {active_id}" if active_id is not None else "target"
            for frame in (left, right):
                h = frame.shape[0]
                cv2.putText(frame, f"Moving to {tgt_str}...",
                            (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frame, f"Moving to {tgt_str}...",
                            (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


# =============================================================================
# YOLO overlay
# =============================================================================

def _draw_yolo(core, frame, imgsz, conf):
    boxes, masks = core._get_filtered_results(frame, conf_override=conf, imgsz=imgsz)
    points = [
        (
            int((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
            int((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
        )
        for b in boxes
    ]
    if boxes and masks and core.qpoint_model is not None:
        try:
            qpoints = core._run_qpoints_batch(frame, boxes, masks)
            qmap = {box_idx: (gx, gy) for gx, gy, box_idx, _ in qpoints}
            points = [qmap.get(i, points[i]) for i in range(len(points))]
        except Exception as exc:
            print(f"[render] qpoint overlay warning: {exc}")
    return core.draw_detections(frame, boxes=boxes, points=points)


def _open_video_writer(output_path, fps, out_size):
    suffix = output_path.suffix.lower()
    if suffix == ".avi":
        codec_candidates = ["MJPG", "XVID"]
    else:
        # Prefer native H.264 writer first. If unavailable in this OpenCV/FFmpeg
        # build, fall back to mp4v and optionally post-convert.
        codec_candidates = ["avc1", "H264", "X264", "mp4v"]

    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)
        if writer.isOpened():
            return writer, codec
        writer.release()

    raise RuntimeError(
        f"Could not open VideoWriter for {output_path}. Tried codecs: {codec_candidates}"
    )


def _transcode_to_h264_mp4(src_path, dst_path):
    try:
        import imageio_ffmpeg
    except Exception as exc:
        print(f"[render] H.264 post-convert unavailable (imageio-ffmpeg import failed): {exc}")
        return False

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(src_path),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:
        print(f"[render] H.264 post-convert failed to launch ffmpeg: {exc}")
        return False

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        tail = "\n".join(stderr[-8:]) if stderr else "(no stderr)"
        print("[render] H.264 post-convert failed; keeping original mp4v file.")
        print(f"[render] ffmpeg stderr (tail):\n{tail}")
        return False

    return True


# =============================================================================
# Main render loop
# =============================================================================

def render_trial(args):
    run_dir = Path(args.run_dir).resolve()
    all_records = _load_records(run_dir)
    records = all_records
    if args.max_frames:
        records = records[: args.max_frames]
    if not records:
        raise RuntimeError(f"No frame records found in {run_dir}")

    fps = args.fps or _estimate_fps(records)
    output = _resolve_output_path(args.output, run_dir)

    detector = None
    if args.run_yolo:
        from vision.detectors.ai_detector import AIDetector

        detector = AIDetector(yolo_path=args.model, conf=args.conf)
        detector.warmup(imgsz=args.imgsz, iters=2, enabled=True)

    first_ts, total_duration, total_targets, timeline_segments = _preprocess_records(records)
    pd_seed_points = _collect_pd_seed_points(all_records)
    pd_reid_delays = _collect_reid_delays(run_dir, all_records)
    render_plan, render_counts = _build_render_plan(records, pd_reid_delays)
    render_chunks = _chunk_render_plan(render_plan)
    survey_flash_start_ts = _collect_survey_flash_start(all_records)
    fire_start_times = _collect_fire_start_times(all_records)
    reid_overlay_cache = {}
    if detector is not None and args.reid_precompute:
        reid_overlay_cache = _precompute_reid_overlay_cache(
            run_dir,
            all_records,
            pd_seed_points,
            detector,
            imgsz=args.imgsz,
            conf=args.conf,
        )
    pd_flow_state = _new_pd_flow_state()
    print(f"[render] {len(records)} frames  fps={fps:.1f}  duration={total_duration:.1f}s  targets={total_targets}")
    print(
        f"[render] plan: independent={render_counts['independent']} reid={render_counts['reid']} "
        f"lk={render_counts['lk']} chunks={len(render_chunks)} parallel={args.parallel_render}"
    )

    output_suffix = output.suffix.lower()
    raw_output = output
    if output_suffix == ".mp4":
        # Write to a temp file first so we can atomically replace the final output,
        # regardless of whether we use native H.264 or post-convert fallback.
        raw_output = output.with_name(f"{output.stem}.__raw__.mp4")

    writer = None
    writer_codec = None
    writer_codec_upper = None
    out_size = None
    render_t0 = time.perf_counter()
    next_progress_pct = 10

    render_ctx = {
        "first_ts": first_ts,
        "total_duration": total_duration,
        "total_targets": total_targets,
        "timeline_segments": timeline_segments,
        "survey_flash_start_ts": survey_flash_start_ts,
        "fire_start_times": fire_start_times,
        "pd_seed_points": pd_seed_points,
        "pd_reid_delays": pd_reid_delays,
        "pd_flow_overlay": args.pd_flow_overlay,
        "detector": detector,
        "yolo_conf": args.conf,
        "reid_overlay_cache": reid_overlay_cache,
    }

    def _fmt_clock(seconds):
        total = max(0, int(round(float(seconds))))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    try:
        total = len(records)

        def _write_combined(idx0, combined):
            nonlocal writer, writer_codec, writer_codec_upper, out_size, next_progress_pct
            if combined is None:
                print(f"[render] skipping missing frame {idx0}")
                return
            if writer is None:
                out_size = (combined.shape[1], combined.shape[0])
                writer, writer_codec = _open_video_writer(raw_output, fps, out_size)
                writer_codec_upper = str(writer_codec).upper()
                print(f"[render] writer codec={writer_codec} size={out_size[0]}x{out_size[1]} fps={fps:.2f}")
            elif (combined.shape[1], combined.shape[0]) != out_size:
                combined = cv2.resize(combined, out_size)

            writer.write(combined)
            pct_done = int((idx0 * 100) / total)
            if pct_done >= next_progress_pct or idx0 == total:
                elapsed_wall_s = time.perf_counter() - render_t0
                avg_s_per_frame = elapsed_wall_s / max(1, idx0)
                eta_s = avg_s_per_frame * max(0, total - idx0)
                while next_progress_pct <= min(100, pct_done):
                    print(
                        f"[render] {next_progress_pct:3d}%  "
                        f"{idx0}/{total} frames  "
                        f"elapsed={_fmt_clock(elapsed_wall_s)}  "
                        f"eta={_fmt_clock(eta_s)}"
                    )
                    next_progress_pct += 10

        def _render_idx(i0):
            return i0, _compose_rendered_frame(run_dir, records[i0], render_plan[i0], render_ctx, pd_flow_state=None)

        for chunk in render_chunks:
            indices = list(range(chunk["start"], chunk["end"]))
            parallel_ok = (
                args.parallel_render
                and chunk["mode"] in ("independent", "reid")
                and len(indices) >= max(1, int(PARALLEL_MIN_CHUNK_FRAMES))
                and (chunk["mode"] != "reid" or args.reid_precompute)
            )

            if parallel_ok:
                print(f"[render] chunk {chunk['mode']} {chunk['start'] + 1}-{chunk['end']} in parallel")
                with ThreadPoolExecutor(max_workers=max(1, int(args.parallel_workers)), thread_name_prefix="render-frame") as pool:
                    for i0, combined in pool.map(_render_idx, indices):
                        _write_combined(i0 + 1, combined)
            else:
                if chunk["mode"] == "lk":
                    pd_flow_state = _new_pd_flow_state()
                for i0 in indices:
                    combined = _compose_rendered_frame(
                        run_dir,
                        records[i0],
                        render_plan[i0],
                        render_ctx,
                        pd_flow_state=pd_flow_state if chunk["mode"] == "lk" else None,
                    )
                    _write_combined(i0 + 1, combined)
    except KeyboardInterrupt:
        print("\n[render] interrupted by user; finalizing output file...")
        raise
    finally:
        if writer is not None:
            writer.release()

    if writer is None:
        raise RuntimeError("No frames were written (all input frames were missing).")

    if output_suffix == ".mp4":
        native_h264_codecs = {"AVC1", "H264", "X264"}
        if writer_codec_upper in native_h264_codecs:
            if raw_output != output:
                raw_output.replace(output)
            print(f"[render] saved {output} (native H.264 codec={writer_codec})")
        else:
            print("[render] Post-converting MP4 temp file to H.264...")
            ok = _transcode_to_h264_mp4(raw_output, output)
            if ok:
                try:
                    raw_output.unlink(missing_ok=True)
                except Exception:
                    pass
                print(f"[render] saved {output} (H.264 post-convert)")
            else:
                if raw_output != output:
                    raw_output.replace(output)
                print(f"[render] saved {output} (fallback mp4v)")
    else:
        print(f"[render] saved {raw_output}")



# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="Raw trial recording folder containing manifest.jsonl")
    parser.add_argument("--trial", type=int, help="Trial number, e.g. 7 for trial_007_*.")
    parser.add_argument("--output", help="Output MP4/AVI path. Default: run_dir/annotated_trial.mp4")
    parser.add_argument("--fps", type=float, help="Output FPS. Default estimates from manifest timestamps.")
    parser.add_argument("--max-frames", type=int, help="Render only the first N frames.")
    parser.add_argument("--run-yolo", action="store_true", help="Run YOLO offline and draw boxes/labels/keypoints.")
    parser.add_argument("--model", help="YOLO .pt or .engine path/name. Default uses config backend selection.")
    parser.add_argument("--imgsz", type=int, help="YOLO inference imgsz for offline overlays.")
    parser.add_argument("--conf", type=float, help="YOLO confidence for offline overlays.")
    parser.add_argument("--no-pd-flow", action="store_true", help="Disable reconstructed PD box-center LK flow overlay.")
    parser.add_argument("--no-reid-precompute", action="store_true", help="Disable staged RE-ID batch precompute and do YOLO inline during render.")
    parser.add_argument("--no-parallel-render", action="store_true", help="Disable chunked parallel frame rendering.")
    parser.add_argument("--parallel-workers", type=int, help="Worker count for independent/re-ID chunk rendering.")
    args = parser.parse_args()

    if args.run_dir:
        args.run_dir = Path(args.run_dir)
    elif args.trial is not None:
        args.run_dir = _find_trial_dir(args.trial)
    else:
        args.run_dir = _default_run_dir()

    args.output = args.output if args.output is not None else OUTPUT_PATH
    args.fps = args.fps if args.fps is not None else OUTPUT_FPS
    args.max_frames = args.max_frames if args.max_frames is not None else MAX_FRAMES
    args.run_yolo = bool(args.run_yolo or RUN_YOLO)
    args.model = args.model if args.model is not None else YOLO_MODEL
    args.imgsz = args.imgsz if args.imgsz is not None else YOLO_IMGSZ
    args.conf = args.conf if args.conf is not None else YOLO_CONF
    args.pd_flow_overlay = bool(PD_FLOW_OVERLAY and not args.no_pd_flow)
    args.reid_precompute = bool(REID_PRECOMPUTE and not args.no_reid_precompute)
    args.parallel_render = bool(PARALLEL_RENDER and not args.no_parallel_render)
    args.parallel_workers = args.parallel_workers if args.parallel_workers is not None else PARALLEL_FRAME_WORKERS

    render_trial(args)


if __name__ == "__main__":
    main()
