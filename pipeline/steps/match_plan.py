"""Stereo matching and planning step helpers."""

import json
from pathlib import Path

import numpy as np

from planning.target_planner import plan_targets
from vision.matching import match_points


def normalize_match(match):
    if isinstance(match, dict):
        return match

    if isinstance(match, (list, tuple)) and len(match) >= 2:
        left_px = tuple(match[0])
        right_px = tuple(match[1])
        out = {
            "left_px": left_px,
            "right_px": right_px,
            "score": 1.0,
        }
        if len(match) >= 3:
            try:
                out["score"] = float(match[2])
            except (TypeError, ValueError):
                pass
        return out

    raise ValueError(f"Unknown match format: {type(match)} {match}")


def normalize_matches(matches):
    return [normalize_match(match) for match in matches]


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _xy_tuple(start_xy):
    if isinstance(start_xy, dict):
        return float(start_xy["x"]), float(start_xy["y"])
    return float(start_xy[0]), float(start_xy[1])


def run_match_and_plan(left_detections, right_detections, coarse_mover, start_xy, output_path=None):
    matched_targets, unmatched_left, unmatched_right = match_points(
        left_detections,
        right_detections,
        verbose=True,
    )
    matched_targets = normalize_matches(matched_targets)

    ref_x, ref_y = _xy_tuple(start_xy)
    coarse_mover.fit_epipolar(matched_targets)
    solved_targets = coarse_mover.solve_all_from_pose(matched_targets, ref_x, ref_y)
    planned_targets = plan_targets(solved_targets, start_xy=(ref_x, ref_y))

    if output_path is not None:
        path = Path(output_path)
        path.write_text(json.dumps(_json_safe({
            "matched_targets": matched_targets,
            "unmatched_left": unmatched_left,
            "unmatched_right": unmatched_right,
            "solved_targets": solved_targets,
            "planned_targets": planned_targets,
        }), indent=2))

    return matched_targets, solved_targets, planned_targets
