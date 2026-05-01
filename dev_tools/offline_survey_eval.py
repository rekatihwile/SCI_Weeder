#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    EXPECTED_WEED_COUNT,
    SURVEY_POS_X,
    SURVEY_POS_Y,
    SURVEY_TARGET_CLASSES,
)
from control.coarse_move import TriangulationCoarseMover  # noqa: E402
from main import _repair_survey_matches  # noqa: E402
from vision.detectors.ai_detector import AIDetector  # noqa: E402
from vision.matching import match_points  # noqa: E402


def _parse_int_list(value):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _load_frames(folder, prefix):
    files = sorted(folder.glob(f"{prefix}_*.jpg"))
    frames = []
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def _load_manifest_entries(trial_dir):
    path = trial_dir / "manifest.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _select_frame_ids(entries, mode):
    if not entries:
        return None

    first_travel_idx = None
    for i, e in enumerate(entries):
        if e.get("state_name") == "TRAVEL":
            first_travel_idx = i
            break

    if mode == "first":
        return [int(e["frame_index"]) for e in entries if "frame_index" in e]

    if mode == "pre_travel_tail":
        subset = entries if first_travel_idx is None else entries[:first_travel_idx]
        return [int(e["frame_index"]) for e in subset if "frame_index" in e]

    if mode == "survey_tail":
        subset = entries if first_travel_idx is None else entries[:first_travel_idx]
        subset = [
            e for e in subset
            if e.get("state_name") in ("SURVEY", "DETECT", "MATCH")
        ]
        return [int(e["frame_index"]) for e in subset if "frame_index" in e]

    raise ValueError(f"Unknown frame selection mode: {mode}")


def _class_counts(detections):
    cls_values = [int(d.get("cls", -1)) for d in detections if isinstance(d, dict)]
    return dict(sorted(Counter(cls_values).items()))


def _target_class_counts(targets):
    vals = []
    for t in targets:
        cls_id = t.get("left_cls", t.get("right_cls"))
        if cls_id is not None:
            vals.append(int(cls_id))
    return dict(sorted(Counter(vals).items()))


def _repair_mode_counts(targets):
    modes = [t.get("survey_repair") for t in targets if t.get("survey_repair")]
    return dict(sorted(Counter(modes).items()))


def _corners_count(solved_targets):
    count = 0
    for t in solved_targets:
        x, y = t["target_xy_mm"]
        if x < 40.0 or y < 40.0:
            count += 1
    return count


def evaluate_combo(detector, coarse_mover, left_frames, right_frames, burst_count, min_hits, cluster_radius, point_mode):
    burst_left = left_frames[-burst_count:]
    burst_right = right_frames[-burst_count:]

    stable_left = detector.cv_left.return_burst_stable(
        burst_left,
        min_stable_views=min_hits,
        group_radius_px=cluster_radius,
        classes_override=SURVEY_TARGET_CLASSES,
        point_mode=point_mode,
        heatmap_final=(point_mode != "box_center"),
    )
    stable_right = detector.cv_right.return_burst_stable(
        burst_right,
        min_stable_views=min_hits,
        group_radius_px=cluster_radius,
        classes_override=SURVEY_TARGET_CLASSES,
        point_mode=point_mode,
        heatmap_final=(point_mode != "box_center"),
    )

    matched, unmatched_left, unmatched_right = match_points(
        stable_left,
        stable_right,
        verbose=False,
    )

    repaired = _repair_survey_matches(
        matched,
        stable_left,
        stable_right,
        expected_count=EXPECTED_WEED_COUNT,
    )

    solved = coarse_mover.solve_all_from_survey(repaired, SURVEY_POS_X, SURVEY_POS_Y)

    return {
        "burst_count": burst_count,
        "min_hits": min_hits,
        "cluster_radius": cluster_radius,
        "stable_left": len(stable_left),
        "stable_right": len(stable_right),
        "matched": len(matched),
        "repaired": len(repaired),
        "unmatched_left": len(unmatched_left),
        "unmatched_right": len(unmatched_right),
        "stable_left_classes": _class_counts(stable_left),
        "stable_right_classes": _class_counts(stable_right),
        "target_classes": _target_class_counts(repaired),
        "repair_modes": _repair_mode_counts(repaired),
        "corner_targets": _corners_count(solved),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline survey/matching evaluator from recorded trial frames")
    parser.add_argument("trial_dir", type=Path, help="Trial folder containing left/ and right/ frame images")
    parser.add_argument("--burst-counts", type=str, default="50,65,75", help="Comma-separated burst counts")
    parser.add_argument("--min-hits", type=str, default="1", help="Comma-separated min_hits values")
    parser.add_argument("--cluster-radii", type=str, default="10,12", help="Comma-separated cluster radius px values")
    parser.add_argument("--point-mode", type=str, default="box_center", choices=["box_center", "qpoint"])
    parser.add_argument(
        "--frame-selection",
        type=str,
        default="pre_travel_tail",
        choices=["first", "pre_travel_tail", "survey_tail"],
        help="How to select frames from manifest before choosing the burst tail",
    )
    args = parser.parse_args()

    trial_dir = args.trial_dir.resolve()
    left_dir = trial_dir / "left"
    right_dir = trial_dir / "right"
    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"Expected left/right folders under {trial_dir}")

    left_frames_all = _load_frames(left_dir, "left")
    right_frames_all = _load_frames(right_dir, "right")

    frame_ids = _select_frame_ids(_load_manifest_entries(trial_dir), args.frame_selection)
    if frame_ids is None:
        left_frames = left_frames_all
        right_frames = right_frames_all
    else:
        left_frames = []
        right_frames = []
        for frame_id in frame_ids:
            idx = frame_id - 1
            if 0 <= idx < len(left_frames_all) and 0 <= idx < len(right_frames_all):
                left_frames.append(left_frames_all[idx])
                right_frames.append(right_frames_all[idx])

    n = min(len(left_frames), len(right_frames))
    left_frames = left_frames[:n]
    right_frames = right_frames[:n]
    if n == 0:
        raise RuntimeError("No readable left/right jpg frames found")

    burst_counts = sorted(set(_parse_int_list(args.burst_counts)))
    min_hits_values = sorted(set(_parse_int_list(args.min_hits)))
    radii = sorted(set(float(v) for v in _parse_int_list(args.cluster_radii)))

    max_burst = max(burst_counts)
    if n < max_burst:
        raise RuntimeError(
            f"Not enough frames for max burst {max_burst}: only {n} available in {trial_dir.name}"
        )

    print(
        f"[offline] trial={trial_dir.name} frames={n} "
        f"selection={args.frame_selection} point_mode={args.point_mode}"
    )

    detector = AIDetector()
    coarse_mover = TriangulationCoarseMover()

    results = []
    for burst_count in burst_counts:
        for min_hits in min_hits_values:
            for radius in radii:
                print(
                    f"\n[offline] eval burst={burst_count} min_hits={min_hits} radius={radius:.1f}",
                    flush=True,
                )
                row = evaluate_combo(
                    detector,
                    coarse_mover,
                    left_frames,
                    right_frames,
                    burst_count,
                    min_hits,
                    radius,
                    args.point_mode,
                )
                results.append(row)
                print(
                    "[offline] "
                    f"stable L/R={row['stable_left']}/{row['stable_right']} "
                    f"matched={row['matched']} repaired={row['repaired']} "
                    f"classes={row['target_classes']} repair_modes={row['repair_modes']} "
                    f"corner_targets={row['corner_targets']}"
                )

    print("\n=== OFFLINE SUMMARY ===")
    for row in results:
        print(
            f"burst={row['burst_count']:>3} min_hits={row['min_hits']:>2} radius={row['cluster_radius']:>4.1f} "
            f"L/R={row['stable_left']:>2}/{row['stable_right']:>2} "
            f"matched={row['matched']:>2} repaired={row['repaired']:>2} "
            f"classes={row['target_classes']} corner={row['corner_targets']}"
        )


if __name__ == "__main__":
    main()
