#!/usr/bin/env python3
# Quick start:
# 1) Activate environment from repo root:
#      source .venv/bin/activate
#
# 2) Set trial/stage once (like render script), then run without extra args:
#      TRIAL_NUMBER = 67
#      TRIAL_RUN_DIR = None
#      STAGE = "reid"
#
# 3) Run one stage sweep on the selected trial:
#      python dev_tools/bruteforce_stage_optimizer.py
#
# 4) Sweep stereo matcher gates:
#      python dev_tools/bruteforce_stage_optimizer.py \
#          --stage match
#
# 5) Sweep re-ID and save constellation truth JSON:
#      python dev_tools/bruteforce_stage_optimizer.py \
#          --stage reid \
#          --save-constellation
#
# 6) Override a parameter grid inline (comma-separated values):
#      python dev_tools/bruteforce_stage_optimizer.py \
#          --stage reid \
#          --param burst_count=10,14,18 \
#          --param max_pd_err=120,150,180
#
# 7) Limit combinations for a quick smoke test:
#      python dev_tools/bruteforce_stage_optimizer.py \
#          --max-combos 5
#
# Optional one-off override:
#      python dev_tools/bruteforce_stage_optimizer.py \
#          trial_recordings/trial_067_20260426_100659 \
#          --stage reid
#
# Output:
# - Appends one row per combination to: experiments/metrics/post_optimizer_results.csv
# - Prints the best parameter set at the end, with accuracy-first then time ranking.
# - For --stage reid with --save-constellation, writes constellation truth JSON under
#   experiments/metrics/constellation_truth_<trial>.json.

"""Brute-force post optimizer for stage-isolated timing/accuracy replay.

This script replays recorded stereo frames and sweeps parameter combinations by stage:
- survey: burst/crop/grouping affect stable detections and planned count
- match: stereo match gates affect repaired target set quality
- reid: local burst/crop/match settings affect target reacquisition quality

Scoring priority is accuracy-first, then time:
1) maximize hit/completeness proxies
2) minimize estimated stage time

It also saves constellation truth from planned targets so re-ID candidates can be
compared against the intended global target geometry.
"""

import argparse
import itertools
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    AI_CONFIDENCE,
    DEFAULT_MODEL,
    EXPECTED_WEED_COUNT,
    SURVEY_BURST_COUNT,
    SURVEY_MIN_HITS,
    SURVEY_CLUSTER_RADIUS_PX,
    SURVEY_CROP_HALF_X_PX,
    SURVEY_CROP_HALF_Y_PX,
    SURVEY_YOLO_IMGSZ,
    SURVEY_POINT_MODE,
    FINE_ALIGN_REID_BURST_COUNT,
    FINE_ALIGN_MIN_HITS,
    FINE_ALIGN_CLUSTER_RADIUS_PX,
    FINE_ALIGN_REID_CROP_HALF_PX,
    FINE_ALIGN_REID_YOLO_IMGSZ,
    FINE_ALIGN_REID_MAX_Y_DIFF_PX,
    FINE_ALIGN_REID_MIN_DISPARITY_PX,
    FINE_ALIGN_REID_MAX_DISPARITY_PX,
    FINE_ALIGN_REID_MAX_PD_ERROR_PX,
    SURVEY_POS_X,
    SURVEY_POS_Y,
)
from control.coarse_move import TriangulationCoarseMover  # noqa: E402
from control.fine_align import _enumerate_reid_pairs, _pair_pd_error  # noqa: E402
from main import _repair_survey_matches  # noqa: E402
from vision.detectors.ai_detector import _WeedCVCore, _select_yolo_model_path  # noqa: E402
from vision.matching import match_points  # noqa: E402


OUT_DIR = ROOT / "experiments" / "metrics"
OUT_CSV = OUT_DIR / "post_optimizer_results.csv"


# =============================================================================
# Quick Settings
# =============================================================================

# Put the trial number here, then run:
#   python dev_tools/bruteforce_stage_optimizer.py --stage reid
# Example: TRIAL_NUMBER = 67 matches trial_recordings/trial_067_*
TRIAL_NUMBER = 67

# Optional fallback if you prefer a path instead of a number.
TRIAL_RUN_DIR = None

# Default stage when --stage is not provided.
# Allowed: "survey", "match", "reid", "all"
STAGE = "reid"

# If > 0, caps combos per stage. Set 0 for full grid.
MAX_COMBOS = 0

# Print progress every N combinations.
LIVE_PROGRESS_EVERY = 10

# YOLO runtime knobs for optimizer replay.
YOLO_MODEL_OVERRIDE = None
YOLO_CONF = AI_CONFIDENCE
YOLO_VERBOSE = False

# Per-stage default sweep grids (edit these lists to change what gets swept).
SURVEY_GRID = {
    "burst_count": [max(10, int(SURVEY_BURST_COUNT) - 10), int(SURVEY_BURST_COUNT), int(SURVEY_BURST_COUNT) + 10],
    "min_hits": [1, int(SURVEY_MIN_HITS)],
    "cluster_radius": [9.0, float(SURVEY_CLUSTER_RADIUS_PX), 12.0],
    "crop_half_x": [200, SURVEY_CROP_HALF_X_PX, 300],
    "crop_half_y": [200, SURVEY_CROP_HALF_Y_PX, 300],
    "imgsz": [SURVEY_YOLO_IMGSZ],
    "point_mode": [SURVEY_POINT_MODE],
}

MATCH_GRID = {
    "burst_count": [int(SURVEY_BURST_COUNT)],
    "min_hits": [int(SURVEY_MIN_HITS)],
    "cluster_radius": [float(SURVEY_CLUSTER_RADIUS_PX)],
    "crop_half_x": [SURVEY_CROP_HALF_X_PX],
    "crop_half_y": [SURVEY_CROP_HALF_Y_PX],
    "imgsz": [SURVEY_YOLO_IMGSZ],
    "point_mode": [SURVEY_POINT_MODE],
    "min_disp": [5.0, 10.0],
    "max_disp": [220.0, 250.0, 300.0],
    "max_y_diff": [25.0, 30.0, 35.0],
    "match_radius": [50.0, 60.0, 70.0],
    "min_score": [0.08, 0.1, 0.14],
    "min_box_iou": [0.12, 0.15, 0.2],
    "iou_weight": [0.4, 0.5, 0.6],
}

REID_GRID = {
    "burst_count": [max(8, int(FINE_ALIGN_REID_BURST_COUNT) - 6), int(FINE_ALIGN_REID_BURST_COUNT), int(FINE_ALIGN_REID_BURST_COUNT) + 6],
    "min_hits": [1, int(FINE_ALIGN_MIN_HITS)],
    "cluster_radius": [25.0, float(FINE_ALIGN_CLUSTER_RADIUS_PX), 45.0],
    "crop_half": [60, int(FINE_ALIGN_REID_CROP_HALF_PX), 90],
    "imgsz": [FINE_ALIGN_REID_YOLO_IMGSZ],
    "point_mode": ["box_center"],
    "max_y_diff": [60.0, float(FINE_ALIGN_REID_MAX_Y_DIFF_PX), 100.0],
    "min_disp": [float(FINE_ALIGN_REID_MIN_DISPARITY_PX)],
    "max_disp": [400.0, float(FINE_ALIGN_REID_MAX_DISPARITY_PX), 550.0],
    "max_pd_err": [120.0, float(FINE_ALIGN_REID_MAX_PD_ERROR_PX), 180.0],
    "hit_tol_mm": [10.0, 15.0, 20.0],
}


def _find_trial_dir(trial_number):
    recordings_dir = ROOT / "trial_recordings"
    trial_id = int(trial_number)
    matches = sorted(recordings_dir.glob(f"trial_{trial_id:03d}_*"))
    matches = [p for p in matches if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No recording folder found for trial {trial_id:03d} in {recordings_dir}")
    if len(matches) > 1:
        print(f"[optimizer] Multiple folders matched trial {trial_id:03d}; using latest: {matches[-1]}")
    return matches[-1]


def _default_trial_dir():
    if TRIAL_RUN_DIR:
        return Path(TRIAL_RUN_DIR)
    if TRIAL_NUMBER is not None:
        return _find_trial_dir(TRIAL_NUMBER)
    raise ValueError("Set TRIAL_NUMBER or TRIAL_RUN_DIR at top, or pass trial_dir argument.")


def _default_stage():
    stage = str(STAGE).strip().lower()
    if stage not in ("survey", "match", "reid", "all"):
        raise ValueError("STAGE must be one of: survey, match, reid, all")
    return stage


def _resolve_stage_list(stage):
    return ["survey", "match", "reid"] if stage == "all" else [stage]


def _build_weed_cores():
    model_spec = YOLO_MODEL_OVERRIDE if YOLO_MODEL_OVERRIDE else DEFAULT_MODEL
    yolo_path, backend = _select_yolo_model_path(model_spec)
    print(f"[optimizer] YOLO model={yolo_path} backend={backend} conf={YOLO_CONF}")
    core_l = _WeedCVCore(yolo_path, qpoint_path=None, conf=YOLO_CONF, verbose=YOLO_VERBOSE, yolo_backend=backend)
    core_r = _WeedCVCore(yolo_path, qpoint_path=None, conf=YOLO_CONF, verbose=YOLO_VERBOSE, yolo_backend=backend)
    return core_l, core_r


def _parse_val(raw):
    raw = raw.strip()
    if raw.lower() in ("none", "null"):
        return None
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.startswith("[") and raw.endswith("]"):
        return json.loads(raw)
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_param_specs(specs):
    grid = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(f"Bad --param '{s}', expected name=v1,v2,...")
        k, rhs = s.split("=", 1)
        vals = [_parse_val(v) for v in rhs.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"No values in --param '{s}'")
        grid[k.strip()] = vals
    return grid


def _grid_product(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def _load_manifest(trial_dir):
    path = trial_dir / "manifest.jsonl"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_frame(trial_dir, rel_path):
    img = cv2.imread(str(trial_dir / rel_path), cv2.IMREAD_COLOR)
    return img


def _survey_frame_rows(entries):
    first_travel = None
    for i, e in enumerate(entries):
        if e.get("state_name") == "TRAVEL":
            first_travel = i
            break
    subset = entries if first_travel is None else entries[:first_travel]
    return [e for e in subset if e.get("left_image_path") and e.get("right_image_path")]


def _target_fine_align_segments(entries):
    segs = []
    cur = None
    for e in entries:
        if e.get("state_name") != "FINE_ALIGN":
            if cur is not None:
                segs.append(cur)
                cur = None
            continue

        tid = e.get("current_target_id")
        if tid is None:
            continue
        if cur is None or cur["target_id"] != tid:
            if cur is not None:
                segs.append(cur)
            cur = {
                "target_id": int(tid),
                "current_target_index": e.get("current_target_index"),
                "gantry_xy": e.get("gantry_position_mm"),
                "planned_target_list": e.get("planned_target_list") or [],
                "rows": [e],
            }
        else:
            cur["rows"].append(e)
    if cur is not None:
        segs.append(cur)
    return segs


def _stable_detect(core, frames, min_hits, radius, point_mode, imgsz, crop_half_x=None, crop_half_y=None):
    if not frames:
        return [], {"yolo_time_s": 0.0, "grouping_time_s": 0.0, "qpoint_time_s": 0.0}

    if crop_half_x is not None and crop_half_y is not None:
        h, w = frames[0].shape[:2]
        cx, cy = w // 2, h // 2
        x0 = max(0, int(cx - crop_half_x))
        x1 = min(w, int(cx + crop_half_x))
        y0 = max(0, int(cy - crop_half_y))
        y1 = min(h, int(cy + crop_half_y))
        crops = [f[y0:y1, x0:x1] for f in frames]
        stable = core.return_burst_stable(
            crops,
            min_stable_views=int(min_hits),
            group_radius_px=float(radius),
            classes_override=None,
            point_mode=point_mode,
            imgsz=imgsz,
            heatmap_final=(point_mode != "box_center"),
        )
        shifted = []
        for s in stable:
            shifted.append({
                **s,
                "point": (int(round(s["point"][0] + x0)), int(round(s["point"][1] + y0))),
                "box": (
                    float(s["box"][0] + x0),
                    float(s["box"][1] + y0),
                    float(s["box"][2] + x0),
                    float(s["box"][3] + y0),
                ),
            })
        return shifted, dict(getattr(core, "last_burst_timing", {}))

    stable = core.return_burst_stable(
        frames,
        min_stable_views=int(min_hits),
        group_radius_px=float(radius),
        classes_override=None,
        point_mode=point_mode,
        imgsz=imgsz,
        heatmap_final=(point_mode != "box_center"),
    )
    return stable, dict(getattr(core, "last_burst_timing", {}))


def _class_counts(targets):
    c = Counter()
    for t in targets:
        cls = t.get("left_cls", t.get("right_cls"))
        if cls is not None:
            c[int(cls)] += 1
    return dict(sorted(c.items()))


def _constellation_truth(planned):
    pts = {int(t["id"]): tuple(map(float, t["target_xy_mm"])) for t in planned if t.get("id") is not None}
    out = {}
    for tid, (x, y) in pts.items():
        neighbors = []
        for oid, (ox, oy) in pts.items():
            if oid == tid:
                continue
            dx = ox - x
            dy = oy - y
            neighbors.append((oid, math.hypot(dx, dy), math.atan2(dy, dx)))
        neighbors.sort(key=lambda z: z[1])
        out[tid] = {
            "xy": [x, y],
            "neighbors": [
                {"id": int(nid), "dist_mm": round(float(d), 4), "angle_rad": round(float(a), 6)}
                for nid, d, a in neighbors[:4]
            ],
        }
    return out


def _neighbor_signature_error(candidate_xy, expected_tid, truth, all_planned_xy):
    if expected_tid not in truth:
        return 0.0
    exp = truth[expected_tid]
    cx, cy = candidate_xy
    err = 0.0
    used = 0
    for n in exp["neighbors"]:
        nid = n["id"]
        if nid not in all_planned_xy:
            continue
        nx, ny = all_planned_xy[nid]
        d = math.hypot(nx - cx, ny - cy)
        err += abs(d - n["dist_mm"])
        used += 1
    return err / used if used else 0.0


def _nearest_planned_id(candidate_xy, all_planned_xy):
    cx, cy = candidate_xy
    return min(all_planned_xy.keys(), key=lambda tid: math.hypot(all_planned_xy[tid][0] - cx, all_planned_xy[tid][1] - cy))


def _constellation_best_planned_id(candidate_xy, truth, all_planned_xy):
    cx, cy = candidate_xy
    return min(
        all_planned_xy.keys(),
        key=lambda tid: (_neighbor_signature_error((cx, cy), tid, truth, all_planned_xy), math.hypot(all_planned_xy[tid][0] - cx, all_planned_xy[tid][1] - cy)),
    )


def _save_constellation(trial_dir, truth):
    out_path = OUT_DIR / f"constellation_truth_{trial_dir.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"trial": trial_dir.name, "truth": truth}, f, indent=2)
    return out_path


def evaluate_survey(trial_dir, entries, params, core_l, core_r):
    rows = _survey_frame_rows(entries)
    burst = int(params["burst_count"])
    if len(rows) < burst:
        return None

    left_frames = [_read_frame(trial_dir, r["left_image_path"]) for r in rows[-burst:]]
    right_frames = [_read_frame(trial_dir, r["right_image_path"]) for r in rows[-burst:]]
    if any(f is None for f in left_frames + right_frames):
        return None

    t0 = time.perf_counter()
    stable_l, t_l = _stable_detect(
        core_l,
        left_frames,
        params["min_hits"],
        params["cluster_radius"],
        params["point_mode"],
        params["imgsz"],
        params.get("crop_half_x"),
        params.get("crop_half_y"),
    )
    stable_r, t_r = _stable_detect(
        core_r,
        right_frames,
        params["min_hits"],
        params["cluster_radius"],
        params["point_mode"],
        params["imgsz"],
        params.get("crop_half_x"),
        params.get("crop_half_y"),
    )
    detect_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    matched, _, _ = match_points(stable_l, stable_r, verbose=False)
    repaired = _repair_survey_matches(matched, stable_l, stable_r, expected_count=EXPECTED_WEED_COUNT)
    match_wall = time.perf_counter() - t1

    cm = TriangulationCoarseMover()
    solved = cm.solve_all_from_survey(repaired, SURVEY_POS_X, SURVEY_POS_Y)
    corner = sum(1 for s in solved if s["target_xy_mm"][0] < 40.0 or s["target_xy_mm"][1] < 40.0)

    complete_gap = max(0, EXPECTED_WEED_COUNT - len(repaired))
    score_primary = len(repaired)
    score_time = detect_wall + match_wall

    return {
        "stage": "survey",
        "score_primary": score_primary,
        "score_time_s": round(score_time, 6),
        "score_tuple": (-complete_gap, -score_primary, round(score_time, 6)),
        "repaired_targets": len(repaired),
        "matched_targets": len(matched),
        "stable_left": len(stable_l),
        "stable_right": len(stable_r),
        "target_classes": _class_counts(repaired),
        "corner_targets": corner,
        "survey_detect_wall_s": round(detect_wall, 6),
        "match_wall_s": round(match_wall, 6),
        "survey_yolo_time_s": round(max(float(t_l.get("yolo_time_s", 0.0)), float(t_r.get("yolo_time_s", 0.0))), 6),
        "survey_grouping_time_s": round(max(float(t_l.get("grouping_time_s", 0.0)), float(t_r.get("grouping_time_s", 0.0))), 6),
        "params": params,
    }


def evaluate_match(trial_dir, entries, params, core_l, core_r):
    # Build one stable-detection baseline then sweep only matcher gates.
    base = {
        "burst_count": int(params.get("burst_count", SURVEY_BURST_COUNT)),
        "min_hits": int(params.get("min_hits", SURVEY_MIN_HITS)),
        "cluster_radius": float(params.get("cluster_radius", SURVEY_CLUSTER_RADIUS_PX)),
        "point_mode": params.get("point_mode", SURVEY_POINT_MODE),
        "imgsz": params.get("imgsz", SURVEY_YOLO_IMGSZ),
        "crop_half_x": params.get("crop_half_x", SURVEY_CROP_HALF_X_PX),
        "crop_half_y": params.get("crop_half_y", SURVEY_CROP_HALF_Y_PX),
    }
    survey_res = evaluate_survey(trial_dir, entries, base, core_l, core_r)
    if survey_res is None:
        return None

    rows = _survey_frame_rows(entries)
    burst = base["burst_count"]
    left_frames = [_read_frame(trial_dir, r["left_image_path"]) for r in rows[-burst:]]
    right_frames = [_read_frame(trial_dir, r["right_image_path"]) for r in rows[-burst:]]
    stable_l, _ = _stable_detect(core_l, left_frames, base["min_hits"], base["cluster_radius"], base["point_mode"], base["imgsz"], base["crop_half_x"], base["crop_half_y"])
    stable_r, _ = _stable_detect(core_r, right_frames, base["min_hits"], base["cluster_radius"], base["point_mode"], base["imgsz"], base["crop_half_x"], base["crop_half_y"])

    t0 = time.perf_counter()
    matched, _, _ = match_points(
        stable_l,
        stable_r,
        verbose=False,
        anchor_min_disp=float(params["min_disp"]),
        anchor_max_disp=float(params["max_disp"]),
        anchor_max_y_diff=float(params["max_y_diff"]),
        match_radius=float(params["match_radius"]),
        min_score=float(params["min_score"]),
        min_box_iou=float(params["min_box_iou"]),
        iou_weight=float(params["iou_weight"]),
    )
    repaired = _repair_survey_matches(matched, stable_l, stable_r, expected_count=EXPECTED_WEED_COUNT)
    wall = time.perf_counter() - t0

    cm = TriangulationCoarseMover()
    solved = cm.solve_all_from_survey(repaired, SURVEY_POS_X, SURVEY_POS_Y)
    corner = sum(1 for s in solved if s["target_xy_mm"][0] < 40.0 or s["target_xy_mm"][1] < 40.0)
    complete_gap = max(0, EXPECTED_WEED_COUNT - len(repaired))

    return {
        "stage": "match",
        "score_primary": len(repaired),
        "score_time_s": round(wall, 6),
        "score_tuple": (-complete_gap, -len(repaired), round(wall, 6)),
        "repaired_targets": len(repaired),
        "matched_targets": len(matched),
        "corner_targets": corner,
        "match_wall_s": round(wall, 6),
        "target_classes": _class_counts(repaired),
        "params": params,
    }


def _target_expected_xy(segment):
    tid = int(segment["target_id"])
    for t in segment["planned_target_list"]:
        if int(t.get("id", -1)) == tid:
            return tuple(map(float, t["target_xy_mm"]))
    idx = segment.get("current_target_index")
    if idx is not None:
        for t in segment["planned_target_list"]:
            if int(t.get("id", -1)) == int(idx):
                return tuple(map(float, t["target_xy_mm"]))
    return None


def evaluate_reid(trial_dir, entries, params, core_l, core_r):
    segs = _target_fine_align_segments(entries)
    if not segs:
        return None

    # Use first segment's planned list as global planned constellation truth.
    planned = segs[0]["planned_target_list"]
    truth = _constellation_truth(planned)
    all_planned_xy = {int(t["id"]): tuple(map(float, t["target_xy_mm"])) for t in planned if t.get("id") is not None}

    cm = TriangulationCoarseMover()

    hits = 0
    winner_agree = 0
    chosen_id_correct_nearest = 0
    chosen_id_correct_constellation = 0
    usable_targets = 0
    total_time_s = 0.0
    failures = 0
    sum_pred_dist = 0.0
    sum_truth_dist = 0.0

    for seg in segs:
        expected_xy = _target_expected_xy(seg)
        if expected_xy is None:
            continue
        rows = seg["rows"]
        burst = int(params["burst_count"])
        if len(rows) < burst:
            continue

        left_frames = [_read_frame(trial_dir, r["left_image_path"]) for r in rows[:burst]]
        right_frames = [_read_frame(trial_dir, r["right_image_path"]) for r in rows[:burst]]
        if any(f is None for f in left_frames + right_frames):
            continue

        t0 = time.perf_counter()
        stable_l, tl = _stable_detect(
            core_l,
            left_frames,
            params["min_hits"],
            params["cluster_radius"],
            params["point_mode"],
            params["imgsz"],
            params["crop_half"],
            params["crop_half"],
        )
        stable_r, tr = _stable_detect(
            core_r,
            right_frames,
            params["min_hits"],
            params["cluster_radius"],
            params["point_mode"],
            params["imgsz"],
            params["crop_half"],
            params["crop_half"],
        )
        detect_wall = time.perf_counter() - t0

        settings = {
            "max_y_diff_px": float(params["max_y_diff"]),
            "min_disparity_px": float(params["min_disp"]),
            "max_disparity_px": float(params["max_disp"]),
            "max_pd_error_px": float(params["max_pd_err"]),
        }

        t1 = time.perf_counter()
        pairs = _enumerate_reid_pairs(stable_l, stable_r, settings)
        total_time_s += detect_wall + (time.perf_counter() - t1)

        if not pairs:
            failures += 1
            continue

        gx, gy = seg.get("gantry_xy") or (SURVEY_POS_X, SURVEY_POS_Y)
        candidates = []
        for p in pairs:
            src = {"left_px": p["left_px"], "right_px": p["right_px"]}
            solved = cm.solve_target_from_pose(src, ref_x=float(gx), ref_y=float(gy))
            tri_xy = tuple(map(float, solved["target_xy_mm"]))
            tri_dist = math.hypot(tri_xy[0] - expected_xy[0], tri_xy[1] - expected_xy[1])
            pd_err = _pair_pd_error(p["left_px"], p["right_px"])
            neighbor_err = _neighbor_signature_error(tri_xy, int(seg["target_id"]), truth, all_planned_xy)
            candidates.append({
                "pair": p,
                "tri_xy": tri_xy,
                "tri_dist": tri_dist,
                "pd_err": pd_err,
                "neighbor_err": neighbor_err,
            })

        # Constellation truth winner: nearest expected + neighborhood signature consistency.
        truth_idx = min(
            range(len(candidates)),
            key=lambda i: (candidates[i]["tri_dist"] + 0.35 * candidates[i]["neighbor_err"], candidates[i]["pd_err"]),
        )

        # Current-logic-style winner: enforce PD gate, then rank by tri_dist then pd.
        gated = [c for c in candidates if c["pd_err"] <= float(params["max_pd_err"])]
        if not gated:
            failures += 1
            continue
        pred = min(gated, key=lambda c: (c["tri_dist"], c["pd_err"], -float(c["pair"].get("score", 0.0))))
        pred_idx = candidates.index(pred)

        truth_pick = candidates[truth_idx]
        pred_nearest_id = _nearest_planned_id(pred["tri_xy"], all_planned_xy)
        pred_const_id = _constellation_best_planned_id(pred["tri_xy"], truth, all_planned_xy)
        expected_id = int(seg["target_id"])

        usable_targets += 1
        sum_pred_dist += float(pred["tri_dist"])
        sum_truth_dist += float(truth_pick["tri_dist"])
        if pred["tri_dist"] <= float(params["hit_tol_mm"]):
            hits += 1
        if pred_idx == truth_idx:
            winner_agree += 1
        if pred_nearest_id == expected_id:
            chosen_id_correct_nearest += 1
        if pred_const_id == expected_id:
            chosen_id_correct_constellation += 1

    if usable_targets == 0:
        return None

    miss = usable_targets - hits
    return {
        "stage": "reid",
        "score_primary": hits,
        "score_time_s": round(total_time_s, 6),
        "score_tuple": (
            -miss,
            -hits,
            -(winner_agree / usable_targets),
            -(chosen_id_correct_constellation / usable_targets),
            round(total_time_s, 6),
        ),
        "targets_usable": usable_targets,
        "targets_hit_like": hits,
        "winner_agree_count": winner_agree,
        "winner_agree_rate": round(winner_agree / usable_targets, 6),
        "chosen_id_correct_nearest": chosen_id_correct_nearest,
        "chosen_id_correct_nearest_rate": round(chosen_id_correct_nearest / usable_targets, 6),
        "chosen_id_correct_constellation": chosen_id_correct_constellation,
        "chosen_id_correct_constellation_rate": round(chosen_id_correct_constellation / usable_targets, 6),
        "avg_pred_target_error_mm": round(sum_pred_dist / usable_targets, 6),
        "avg_truth_target_error_mm": round(sum_truth_dist / usable_targets, 6),
        "reid_failures": failures,
        "reid_estimated_time_s": round(total_time_s, 6),
        "params": params,
        "truth": truth,
    }


def _default_grid(stage):
    if stage == "survey":
        return {k: list(v) for k, v in SURVEY_GRID.items()}
    if stage == "match":
        return {k: list(v) for k, v in MATCH_GRID.items()}
    if stage == "reid":
        return {k: list(v) for k, v in REID_GRID.items()}
    raise ValueError(f"Unknown stage {stage}")


def _append_csv(row):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()
    fields = sorted(row.keys())
    if OUT_CSV.exists():
        # keep stable existing header union.
        with open(OUT_CSV, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        fields = header[:]
        for k in row.keys():
            if k not in fields:
                fields.append(k)

        # rewrite with union header if grown.
        rows = []
        with open(OUT_CSV, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        if lines:
            old_fields = lines[0].split(",")
            for ln in lines[1:]:
                if not ln.strip():
                    continue
                vals = ln.split(",")
                d = {old_fields[i]: vals[i] if i < len(vals) else "" for i in range(len(old_fields))}
                rows.append(d)
        with open(OUT_CSV, "w", encoding="utf-8") as f:
            f.write(",".join(fields) + "\n")
            for d in rows:
                f.write(",".join(str(d.get(k, "")) for k in fields) + "\n")
            f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")
        return

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage-wise brute-force post optimizer")
    parser.add_argument("trial_dir", nargs="?", type=Path, help="Trial folder with manifest.jsonl and left/right frames")
    parser.add_argument("--stage", choices=["survey", "match", "reid", "all"], default=None)
    parser.add_argument("--param", action="append", default=[], help="Override grid: name=v1,v2,...")
    parser.add_argument("--max-combos", type=int, default=None, help="Cap combinations per stage (0 = all)")
    parser.add_argument("--save-constellation", action="store_true", help="Write global planned constellation truth JSON")
    args = parser.parse_args()

    trial_dir = args.trial_dir.resolve() if args.trial_dir else _default_trial_dir().resolve()
    stage = args.stage if args.stage else _default_stage()
    stage_list = _resolve_stage_list(stage)
    max_combos = MAX_COMBOS if args.max_combos is None else args.max_combos
    entries = _load_manifest(trial_dir)
    core_l, core_r = _build_weed_cores()

    all_results = []

    for current_stage in stage_list:
        grid = _default_grid(current_stage)
        user = _parse_param_specs(args.param)
        for k, v in user.items():
            grid[k] = v

        combos = list(_grid_product(grid))
        if max_combos and max_combos > 0:
            combos = combos[: max_combos]

        print(f"[optimizer] trial={trial_dir.name} stage={current_stage} combos={len(combos)}")

        results = []
        truth_saved = None
        stage_started = time.perf_counter()
        stage_best = None

        for i, p in enumerate(combos, start=1):
            combo_t0 = time.perf_counter()
            if current_stage == "survey":
                r = evaluate_survey(trial_dir, entries, p, core_l, core_r)
            elif current_stage == "match":
                r = evaluate_match(trial_dir, entries, p, core_l, core_r)
            else:
                r = evaluate_reid(trial_dir, entries, p, core_l, core_r)

            if r is None:
                print(f"[optimizer] {current_stage} {i}/{len(combos)} skipped (insufficient data)")
                continue

            if args.save_constellation and current_stage == "reid" and truth_saved is None:
                truth_saved = _save_constellation(trial_dir, r.get("truth") or {})

            row = {
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "trial": trial_dir.name,
                "stage": current_stage,
                "combo_index": i,
                "score_primary": r.get("score_primary"),
                "score_time_s": r.get("score_time_s"),
                "score_tuple": json.dumps(r.get("score_tuple")),
                "params": json.dumps(p, sort_keys=True),
                "repaired_targets": r.get("repaired_targets"),
                "matched_targets": r.get("matched_targets"),
                "stable_left": r.get("stable_left"),
                "stable_right": r.get("stable_right"),
                "target_classes": json.dumps(r.get("target_classes")) if r.get("target_classes") is not None else None,
                "corner_targets": r.get("corner_targets"),
                "survey_detect_wall_s": r.get("survey_detect_wall_s"),
                "match_wall_s": r.get("match_wall_s"),
                "survey_yolo_time_s": r.get("survey_yolo_time_s"),
                "survey_grouping_time_s": r.get("survey_grouping_time_s"),
                "targets_usable": r.get("targets_usable"),
                "targets_hit_like": r.get("targets_hit_like"),
                "winner_agree_count": r.get("winner_agree_count"),
                "winner_agree_rate": r.get("winner_agree_rate"),
                "chosen_id_correct_nearest": r.get("chosen_id_correct_nearest"),
                "chosen_id_correct_nearest_rate": r.get("chosen_id_correct_nearest_rate"),
                "chosen_id_correct_constellation": r.get("chosen_id_correct_constellation"),
                "chosen_id_correct_constellation_rate": r.get("chosen_id_correct_constellation_rate"),
                "avg_pred_target_error_mm": r.get("avg_pred_target_error_mm"),
                "avg_truth_target_error_mm": r.get("avg_truth_target_error_mm"),
                "reid_failures": r.get("reid_failures"),
                "reid_estimated_time_s": r.get("reid_estimated_time_s"),
            }
            _append_csv(row)
            results.append(r)
            all_results.append(r)

            if stage_best is None or r["score_tuple"] < stage_best["score_tuple"]:
                stage_best = r

            elapsed = time.perf_counter() - stage_started
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(combos) - i) / rate if rate > 0 else 0.0
            combo_dt = time.perf_counter() - combo_t0
            should_log = (i == 1) or (i % max(1, int(LIVE_PROGRESS_EVERY)) == 0) or (stage_best is r)
            if should_log:
                if current_stage == "reid":
                    print(
                        f"[live] {current_stage} {i}/{len(combos)} combo={combo_dt:.2f}s "
                        f"hit={r.get('targets_hit_like')}/{r.get('targets_usable')} "
                        f"agree={r.get('winner_agree_count')}/{r.get('targets_usable')} "
                        f"const_id={r.get('chosen_id_correct_constellation')}/{r.get('targets_usable')} "
                        f"score={r['score_tuple']} "
                        f"best={stage_best['score_tuple']} eta={eta/60.0:.1f}m"
                    )
                else:
                    print(
                        f"[live] {current_stage} {i}/{len(combos)} combo={combo_dt:.2f}s "
                        f"repaired={r.get('repaired_targets')} score={r['score_tuple']} "
                        f"best={stage_best['score_tuple']} eta={eta/60.0:.1f}m"
                    )

        if not results:
            print(f"[optimizer] stage={current_stage} no valid results")
            continue

        best = min(results, key=lambda x: x["score_tuple"])
        print("\n=== BEST ===")
        print(json.dumps({
            "stage": current_stage,
            "score_tuple": best["score_tuple"],
            "score_primary": best["score_primary"],
            "score_time_s": best["score_time_s"],
            "params": best["params"],
            "winner_agree_rate": best.get("winner_agree_rate"),
            "chosen_id_correct_constellation_rate": best.get("chosen_id_correct_constellation_rate"),
            "avg_pred_target_error_mm": best.get("avg_pred_target_error_mm"),
            "truth_file": str(truth_saved) if truth_saved else None,
            "csv": str(OUT_CSV),
        }, indent=2, default=str))

    if not all_results:
        print("[optimizer] no valid results across selected stage(s)")


if __name__ == "__main__":
    main()
