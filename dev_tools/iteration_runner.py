#!/usr/bin/env python3
"""Small experiment runner/logger for timing iteration and plotting.

Features:
- Update config variables from CLI (--set NAME=VALUE)
- Optionally run main.py once (--run-main)
- Append a plot-ready row to experiments/metrics/iteration_dataset.csv
  with per-state timings, selected variables, camera settings, and
  simple lighting proxies measured from recorded images.

Examples:
  python dev_tools/iteration_runner.py --set SURVEY_BURST_COUNT=70 --run-main
  python dev_tools/iteration_runner.py --set AI_CONFIDENCE=0.002 --collect-only
  python dev_tools/iteration_runner.py --trial-dir trial_recordings/trial_067_20260426_100659
"""

import argparse
import ast
import csv
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


DATASET_DEFAULT = ROOT / "experiments" / "metrics" / "iteration_dataset.csv"
RUN_JSON_DIR = ROOT / "experiments" / "metrics" / "json"
TRIAL_DIR_ROOT = ROOT / "trial_recordings"
CONFIG_PATH = ROOT / "config.py"
CAMERA_CONFIG_PATH = ROOT / "params" / "hardware" / "camera_config.json"


TRACKED_VARS = [
    "RUN_PIPELINE_CYCLES",
    "AI_CONFIDENCE",
    "SURVEY_BURST_COUNT",
    "SURVEY_MIN_HITS",
    "SURVEY_CLUSTER_RADIUS_PX",
    "SURVEY_CROP_MODE",
    "SURVEY_CROP_HALF_X_PX",
    "SURVEY_CROP_HALF_Y_PX",
    "SURVEY_YOLO_IMGSZ",
    "SURVEY_DETECT_ALL_CLASSES",
    "SURVEY_DETECT_CLASS_IDS",
    "SURVEY_CAN_TARGET_CLASSES",
    "SURVEY_CANT_TARGET_CLASSES",
    "FINE_ALIGN_REID_BURST_COUNT",
    "FINE_ALIGN_REID_CROP_HALF_PX",
    "FINE_ALIGN_MAX_TIME_SEC",
    "FINE_ALIGN_SETTLE_FRAMES",
    "FINE_ALIGN_DEADZONE_PX",
    "FINE_ALIGN_ENABLE_SNAP",
    "FINAL_SNAP_BURST_COUNT",
    "LASER_FIRE_DURATION_SEC",
    "LASER_ARM_DELAY_SEC",
]


def _parse_value(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _python_literal(value):
    return repr(value)


def apply_config_overrides(overrides):
    if not overrides:
        return {}

    text = CONFIG_PATH.read_text(encoding="utf-8")
    applied = {}

    for name, value in overrides.items():
        pattern = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*).*$", re.MULTILINE)
        replacement = rf"\1{_python_literal(value)}"
        new_text, count = pattern.subn(replacement, text)
        if count == 0:
            raise ValueError(f"Variable {name} not found in config.py")
        text = new_text
        applied[name] = value

    CONFIG_PATH.write_text(text, encoding="utf-8")
    return applied


def run_main_once():
    cmd = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "main.py")]
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def _latest_run_json():
    if not RUN_JSON_DIR.exists():
        return None
    files = sorted(RUN_JSON_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_run_payload(run_json_path):
    with open(run_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_trial_dir(explicit_trial_dir=None):
    if explicit_trial_dir:
        p = Path(explicit_trial_dir)
        if not p.is_absolute():
            p = ROOT / p
        return p.resolve()

    if not TRIAL_DIR_ROOT.exists():
        return None

    trial_dirs = [p for p in TRIAL_DIR_ROOT.glob("trial_*") if p.is_dir()]
    if not trial_dirs:
        return None
    trial_dirs.sort(key=lambda p: p.stat().st_mtime)
    return trial_dirs[-1]


def _sample_lighting_stats(trial_dir, max_samples=40):
    stats = {
        "lighting_left_gray_mean": None,
        "lighting_right_gray_mean": None,
        "lighting_left_gray_std": None,
        "lighting_right_gray_std": None,
        "lighting_left_samples": 0,
        "lighting_right_samples": 0,
    }
    if trial_dir is None:
        return stats

    left_dir = trial_dir / "left"
    right_dir = trial_dir / "right"
    if not left_dir.exists() or not right_dir.exists():
        return stats

    left_paths = sorted(left_dir.glob("left_*.jpg"))
    right_paths = sorted(right_dir.glob("right_*.jpg"))
    if not left_paths or not right_paths:
        return stats

    def _pick(paths):
        if len(paths) <= max_samples:
            return paths
        step = max(1, len(paths) // max_samples)
        return paths[::step][:max_samples]

    left_vals = []
    right_vals = []

    for p in _pick(left_paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            left_vals.append(float(img.mean()))

    for p in _pick(right_paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            right_vals.append(float(img.mean()))

    if left_vals:
        stats["lighting_left_gray_mean"] = round(float(statistics.mean(left_vals)), 3)
        stats["lighting_left_gray_std"] = round(float(statistics.pstdev(left_vals)), 3)
        stats["lighting_left_samples"] = len(left_vals)

    if right_vals:
        stats["lighting_right_gray_mean"] = round(float(statistics.mean(right_vals)), 3)
        stats["lighting_right_gray_std"] = round(float(statistics.pstdev(right_vals)), 3)
        stats["lighting_right_samples"] = len(right_vals)

    return stats


def _camera_settings_row():
    out = {
        "cam_left_auto_exposure": None,
        "cam_left_exposure": None,
        "cam_left_gain": None,
        "cam_left_auto_wb": None,
        "cam_left_white_balance": None,
        "cam_right_auto_exposure": None,
        "cam_right_exposure": None,
        "cam_right_gain": None,
        "cam_right_auto_wb": None,
        "cam_right_white_balance": None,
    }
    if not CAMERA_CONFIG_PATH.exists():
        return out

    with open(CAMERA_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    left = cfg.get("left") or {}
    right = cfg.get("right") or {}

    out.update({
        "cam_left_auto_exposure": left.get("auto_exposure"),
        "cam_left_exposure": left.get("exposure"),
        "cam_left_gain": left.get("gain"),
        "cam_left_auto_wb": left.get("auto_wb"),
        "cam_left_white_balance": left.get("white_balance"),
        "cam_right_auto_exposure": right.get("auto_exposure"),
        "cam_right_exposure": right.get("exposure"),
        "cam_right_gain": right.get("gain"),
        "cam_right_auto_wb": right.get("auto_wb"),
        "cam_right_white_balance": right.get("white_balance"),
    })
    return out


def _tracked_var_row():
    row = {}
    for name in TRACKED_VARS:
        row[f"var_{name}"] = getattr(config, name, None)
    return row


def _append_dataset_row(dataset_path, row):
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    existing_fields = []
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                existing_fields = next(reader)
            except StopIteration:
                existing_fields = []

    fields = list(existing_fields)
    for k in row.keys():
        if k not in fields:
            fields.append(k)

    rows = []
    if dataset_path.exists() and existing_fields:
        with open(dataset_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    rows.append({k: row.get(k) for k in fields})

    with open(dataset_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Iterative variable runner + timing dataset logger")
    parser.add_argument("--set", action="append", default=[], help="Override config value: NAME=VALUE (repeatable)")
    parser.add_argument("--run-main", action="store_true", help="Run main.py once after applying --set values")
    parser.add_argument("--collect-only", action="store_true", help="Only collect latest run into dataset")
    parser.add_argument("--trial-dir", type=str, default=None, help="Explicit trial dir for lighting stats")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DEFAULT), help="CSV path for plot-ready dataset")
    args = parser.parse_args()

    overrides = {}
    for entry in args.set:
        if "=" not in entry:
            raise ValueError(f"Bad --set format: {entry}; expected NAME=VALUE")
        name, raw = entry.split("=", 1)
        overrides[name.strip()] = _parse_value(raw.strip())

    applied = {}
    if overrides:
        applied = apply_config_overrides(overrides)
        print(f"[iteration] applied overrides: {applied}")

    if args.run_main and not args.collect_only:
        print("[iteration] running main.py once...")
        rc = run_main_once()
        print(f"[iteration] main.py exit code: {rc}")

    run_json = _latest_run_json()
    if run_json is None:
        raise RuntimeError("No run JSON found under experiments/metrics/json")

    payload = _load_run_payload(run_json)
    run = payload.get("run") or {}
    run_id = run.get("run_id", run_json.stem)

    trial_dir = _resolve_trial_dir(args.trial_dir)
    lighting = _sample_lighting_stats(trial_dir)

    row = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "run_status": run.get("run_status"),
        "trial_id": run.get("trial_id"),
        "trial_type": run.get("trial_type"),
        "layout_type": run.get("layout_type"),
        "notes": run.get("notes"),
        "trial_dir": str(trial_dir) if trial_dir else None,
        "applied_overrides": json.dumps(applied, sort_keys=True),
        "total_run_time_s": run.get("total_run_time_s"),
        "survey_time_s": run.get("survey_time_s"),
        "survey_camera_read_time_s": run.get("survey_camera_read_time_s"),
        "survey_yolo_time_s": run.get("survey_yolo_time_s"),
        "survey_grouping_time_s": run.get("survey_grouping_time_s"),
        "detection_time_s": run.get("detection_time_s"),
        "stereo_matching_time_s": run.get("stereo_matching_time_s"),
        "triangulation_time_s": run.get("triangulation_time_s"),
        "planning_time_s": run.get("planning_time_s"),
        "total_travel_time_s": run.get("total_travel_time_s"),
        "total_pd_time_s": run.get("total_pd_time_s"),
        "total_fine_align_reid_yolo_time_s": run.get("total_fine_align_reid_yolo_time_s"),
        "total_fine_align_reid_time_s": run.get("total_fine_align_reid_time_s"),
        "total_fine_align_pd_lk_time_s": run.get("total_fine_align_pd_lk_time_s"),
        "total_final_snap_time_s": run.get("total_final_snap_time_s"),
        "total_fire_time_s": run.get("total_fire_time_s"),
        "recording_frame_save_time_s": run.get("recording_frame_save_time_s"),
        "num_targets_planned": run.get("num_targets_planned"),
        "num_targets_attempted": run.get("num_targets_attempted"),
        "num_targets_fired": run.get("num_targets_fired"),
        "model_load_time_s": run.get("model_load_time_s"),
        "warmup_time_s": run.get("warmup_time_s"),
        "weeds_per_min": run.get("weeds_per_min"),
        "area_rate_m2_per_min": run.get("area_rate_m2_per_min"),
        **_tracked_var_row(),
        **_camera_settings_row(),
        **lighting,
    }

    dataset_path = Path(args.dataset)
    _append_dataset_row(dataset_path, row)
    print(f"[iteration] appended dataset row for run_id={run_id} -> {dataset_path}")


if __name__ == "__main__":
    main()
