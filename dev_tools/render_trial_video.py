#!/usr/bin/env python3
"""Render an annotated video from a raw trial recording folder."""

import argparse
import json
import sys
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
)


# =============================================================================
# Quick Settings
# =============================================================================

# Put the trial number here, then run:
#   python dev_tools/render_trial_video.py
#
# Example: TRIAL_NUMBER = 7 matches trial_recordings/trial_007_*
TRIAL_NUMBER = 27

# Optional fallback if you prefer a path instead of a number.
TRIAL_RUN_DIR = None

# Output can be a folder or a full .mp4/.avi path.
# Default saves into REPO_ROOT/Rendered_Videos/.
OUTPUT_PATH = "Rendered_Videos"

RUN_YOLO = True
YOLO_MODEL = None
YOLO_IMGSZ = 640
YOLO_CONF = AI_CONFIDENCE

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
    """Map target id to the recorded re-ID box center used to seed PD/LK."""
    seeds = {}
    for record in records:
        for hit in record.get("hit_targets_so_far") or []:
            tid = _target_id(hit.get("id"))
            if tid is None or tid in seeds:
                continue
            left_pt = _float_pt(hit.get("left_px"))
            right_pt = _float_pt(hit.get("right_px"))
            if left_pt is not None and right_pt is not None:
                seeds[tid] = {"left": left_pt, "right": right_pt}
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
    for hit in hits or []:
        if _target_id(hit.get("id")) == active_id:
            return hit
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


def _yolo_detections_in_box(core, raw_frame, crop_box, conf):
    """Run YOLO on the crop region; return full-frame detection dicts {center, box, conf}.

    Uses classes=None so every plant class is shown as a candidate, regardless of
    what target_class the detector was configured for.
    """
    bx0, by0, bx1, by1 = crop_box
    crop = raw_frame[by0:by1, bx0:bx1]
    if crop.size == 0:
        return []
    try:
        results = core.yolo(
            crop,
            imgsz=640,
            verbose=False,
            conf=conf,
            retina_masks=False,
            classes=None,
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
        })
    return dets


def _draw_reid_overlay(left, right, raw_left, raw_right, record, active_id, seed,
                       segment_elapsed, reid_delay, detector=None, yolo_conf=None, state=None):
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
        conf = yolo_conf if yolo_conf is not None else YOLO_CONF

        def _detect_side(core, raw_frame, box, seed_pt):
            dets = _yolo_detections_in_box(core, raw_frame, box, conf)
            winner_idx = None
            if dets and seed_pt is not None:
                sp = _as_int_pt(seed_pt)
                if sp is not None:
                    winner_idx = min(
                        range(len(dets)),
                        key=lambda i: (dets[i]["center"][0] - sp[0]) ** 2
                                     + (dets[i]["center"][1] - sp[1]) ** 2,
                    )
            return dets, winner_idx

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
            tag = f"{'winner' if is_winner else 'loser'} {det['conf']:.2f}"
            _put_text(frame, tag, (x1, max(14, y1 - 4)), scale=0.42, color=color)


def _draw_pd_flow_overlay(left, right, raw_left, raw_right, record, seed_points, reid_delays, state,
                          detector=None, yolo_conf=None):
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

    # RE-ID phase + brief flash after it ends.
    if segment_elapsed <= reid_delay + REID_CV_FLASH_SEC:
        _draw_reid_overlay(left, right, raw_left, raw_right, record, active_id, seed,
                           segment_elapsed, reid_delay, detector=detector, yolo_conf=yolo_conf, state=state)
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

    # Fine-align: horizontal Y-reference line only.
    # No vertical line — stereo cameras are symmetric about the laser so the
    # target appears in the right portion of the left cam and left portion of
    # the right cam, never at dead-center X in either individual frame.
    if state == "FINE_ALIGN":
        for frame in (left, right):
            h, w = frame.shape[:2]
            cy = h // 2
            cv2.line(frame, (0, cy), (w, cy), (255, 60, 0), 1, cv2.LINE_AA)
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
    survey_flash_start_ts = _collect_survey_flash_start(all_records)
    fire_start_times = _collect_fire_start_times(all_records)
    pd_flow_state = _new_pd_flow_state()
    print(f"[render] {len(records)} frames  fps={fps:.1f}  duration={total_duration:.1f}s  targets={total_targets}")

    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if output.suffix.lower() == ".avi" else "mp4v"))
    writer = None
    out_size = None

    for idx, record in enumerate(records, start=1):
        left = cv2.imread(str(run_dir / record["left_image_path"]))
        right = cv2.imread(str(run_dir / record["right_image_path"]))
        if left is None or right is None:
            print(f"[render] skipping missing frame {idx}")
            continue

        raw_left = left.copy()
        raw_right = right.copy()

        _draw_manifest_overlays(
            left, right, record, first_ts,
            survey_flash_start_ts=survey_flash_start_ts,
            fire_start_times=fire_start_times,
            lk_pt_left=pd_flow_state.get("pt_left"),
            lk_pt_right=pd_flow_state.get("pt_right"),
        )
        if getattr(args, "pd_flow_overlay", PD_FLOW_OVERLAY):
            _draw_pd_flow_overlay(
                left, right, raw_left, raw_right,
                record, pd_seed_points, pd_reid_delays, pd_flow_state,
                detector=detector, yolo_conf=args.conf,
            )

        if left.shape[0] != right.shape[0]:
            right = cv2.resize(right, (right.shape[1], left.shape[0]))
        video_row = np.hstack([left, right])

        # Prepend the header bar
        header = np.zeros((HEADER_H, video_row.shape[1], 3), dtype=np.uint8)
        combined = np.vstack([header, video_row])

        elapsed = 0.0
        if record.get("timestamp_monotonic") is not None:
            elapsed = float(record["timestamp_monotonic"]) - float(first_ts)
        n_hits = len(record.get("hit_targets_so_far") or [])
        state = record.get("state_name") or "?"

        _draw_progress_header(
            combined, elapsed, total_duration,
            n_hits, total_targets, state, timeline_segments,
        )

        if writer is None:
            out_size = (combined.shape[1], combined.shape[0])
            writer = cv2.VideoWriter(str(output), fourcc, fps, out_size)
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {output}")
        elif (combined.shape[1], combined.shape[0]) != out_size:
            combined = cv2.resize(combined, out_size)

        writer.write(combined)
        if idx % 25 == 0:
            print(f"[render] {idx}/{len(records)} frames")

    if writer is not None:
        writer.release()
    print(f"[render] saved {output}")


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

    render_trial(args)


if __name__ == "__main__":
    main()
