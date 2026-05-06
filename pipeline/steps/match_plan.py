"""Stereo matching and planning step helpers."""

import json
from pathlib import Path

import numpy as np

import config as runtime_config
from experiments.grid_filter import (
    apply_trial_filter,
    assign_grid_metadata,
    make_grid_from_config,
    print_filter_debug,
    summarize_filter_run,
)
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


def _is_target_workspace_safe(target):
    xy = target.get("target_xy_mm")
    if xy is None:
        return False
    return (
        runtime_config.WORKSPACE_X_MIN <= float(xy[0]) <= runtime_config.WORKSPACE_X_MAX
        and runtime_config.WORKSPACE_Y_MIN <= float(xy[1]) <= runtime_config.WORKSPACE_Y_MAX
    )


def run_match_and_plan(
    left_detections,
    right_detections,
    coarse_mover,
    start_xy,
    output_path=None,
    precomputed_matches=None,
):
    if precomputed_matches is None:
        matched_targets, unmatched_left, unmatched_right = match_points(
            left_detections,
            right_detections,
            verbose=True,
        )
    else:
        matched_targets = precomputed_matches
        unmatched_left = []
        unmatched_right = []
        print(f"[MATCH] Reusing {len(matched_targets)} accepted survey match(es).")
    matched_targets = normalize_matches(matched_targets)

    ref_x, ref_y = _xy_tuple(start_xy)
    coarse_mover.fit_epipolar(matched_targets)
    solved_targets = coarse_mover.solve_all_from_pose(matched_targets, ref_x, ref_y)
    for target_id, target in enumerate(solved_targets, start=1):
        target["target_id"] = target_id

    filter_info = {
        "selected_targets": solved_targets,
        "rejected_targets": [],
        "occupied_cell_ids": [],
        "selected_cell_ids": [],
        "rejected_cell_ids": [],
        "warnings": [],
    }
    grid_summary = {}
    if getattr(runtime_config, "EXPERIMENT_GRID_ENABLED", True):
        grid = make_grid_from_config(runtime_config)
        assign_grid_metadata(solved_targets, grid)
        config_values = {
            "experiment_grid_enabled": bool(getattr(runtime_config, "EXPERIMENT_GRID_ENABLED", True)),
            "trial_filter_enabled": bool(getattr(runtime_config, "TRIAL_FILTER_ENABLED", False)),
            "trial_filter_mode": getattr(runtime_config, "TRIAL_FILTER_MODE", "none"),
            "random_seed": getattr(runtime_config, "RANDOM_SEED", None),
            "requested_active_cell_count": getattr(runtime_config, "REQUESTED_ACTIVE_CELL_COUNT", 0),
        }
        planned_input, filter_info = apply_trial_filter(
            solved_targets,
            enabled=getattr(runtime_config, "TRIAL_FILTER_ENABLED", False),
            mode=getattr(runtime_config, "TRIAL_FILTER_MODE", "none"),
            requested_active_cell_count=getattr(runtime_config, "REQUESTED_ACTIVE_CELL_COUNT", 0),
            active_cell_ids=getattr(runtime_config, "ACTIVE_CELL_IDS", []),
            active_radius_cells=getattr(runtime_config, "ACTIVE_RADIUS_CELLS", 0),
            active_ring_index=getattr(runtime_config, "ACTIVE_RING_INDEX", 0),
            random_seed=getattr(runtime_config, "RANDOM_SEED", None),
            filter_only_weed_classes=getattr(runtime_config, "FILTER_ONLY_WEED_CLASSES", True),
            eligible_target_fn=_is_target_workspace_safe,
        )
        grid_summary = summarize_filter_run(grid, solved_targets, filter_info, config_values)
        if getattr(runtime_config, "TRIAL_FILTER_ENABLED", False) or getattr(runtime_config, "DRY_RUN_GRID_FILTER", False):
            print_filter_debug(
                getattr(runtime_config, "TRIAL_FILTER_MODE", "none"),
                grid,
                solved_targets,
                filter_info,
                getattr(runtime_config, "REQUESTED_ACTIVE_CELL_COUNT", 0),
            )
    else:
        planned_input = solved_targets
        for target in solved_targets:
            target["was_selected_by_trial_filter"] = True
            target["selection_reason"] = "grid_disabled"
        grid_summary = {
            "experiment_grid_enabled": False,
            "trial_filter_enabled": False,
            "trial_filter_mode": "none",
            "random_seed": getattr(runtime_config, "RANDOM_SEED", None),
            "surveyed_target_count": len(solved_targets),
            "selected_target_count": len(solved_targets),
            "rejected_target_count": 0,
        }

    planned_targets = plan_targets(planned_input, start_xy=(ref_x, ref_y))

    if output_path is not None:
        path = Path(output_path)
        path.write_text(json.dumps(_json_safe({
            "matched_targets": matched_targets,
            "unmatched_left": unmatched_left,
            "unmatched_right": unmatched_right,
            "solved_targets": solved_targets,
            "planned_targets": planned_targets,
            "grid_filter": filter_info,
            "grid_summary": grid_summary,
        }), indent=2))

    coarse_mover.last_grid_filter_info = filter_info
    coarse_mover.last_grid_summary = grid_summary
    coarse_mover.last_solved_targets = solved_targets
    return matched_targets, solved_targets, planned_targets
