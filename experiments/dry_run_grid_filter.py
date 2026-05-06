import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as runtime_config
from experiments.grid_filter import (
    apply_trial_filter,
    assign_grid_metadata,
    make_grid_from_config,
    print_filter_debug,
    summarize_filter_run,
)


def _load_targets(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("solved_targets") or data.get("planned_targets") or data.get("targets") or []
    targets = []
    for idx, item in enumerate(data, start=1):
        target = dict(item)
        if "target_xy_mm" not in target:
            if "coarse_triangulated_mm" in target:
                target["target_xy_mm"] = tuple(target["coarse_triangulated_mm"][:2])
            else:
                target["target_xy_mm"] = (target.get("x_target_mm"), target.get("y_target_mm"))
        target.setdefault("target_id", item.get("target_id", item.get("id", idx)))
        target.setdefault("source_target", item.get("source_target", {}))
        targets.append(target)
    return targets


def main():
    parser = argparse.ArgumentParser(description="Offline grid filter dry run from saved target JSON.")
    parser.add_argument("targets_json", help="JSON containing targets, solved_targets, or planned_targets")
    parser.add_argument("--out", default=None, help="output manifest JSON path")
    args = parser.parse_args()

    targets = _load_targets(args.targets_json)
    grid = make_grid_from_config(runtime_config)
    assign_grid_metadata(targets, grid)
    for target in targets:
        xy = target.get("target_xy_mm") or (None, None)
        src = target.get("source_target", {})
        target.setdefault("detection_id", target.get("target_id"))
        target.setdefault("class_name", src.get("class_name"))
        target.setdefault("class_id", src.get("left_cls", src.get("right_cls")))
        target.setdefault("confidence", src.get("left_conf", src.get("right_conf", src.get("conf"))))
        target.setdefault("x_target_mm", xy[0])
        target.setdefault("y_target_mm", xy[1])
        target.setdefault("z_target_mm", target.get("z_target_mm"))
    selected, filter_info = apply_trial_filter(
        targets,
        enabled=getattr(runtime_config, "TRIAL_FILTER_ENABLED", False),
        mode=getattr(runtime_config, "TRIAL_FILTER_MODE", "none"),
        requested_active_cell_count=getattr(runtime_config, "REQUESTED_ACTIVE_CELL_COUNT", 0),
        active_cell_ids=getattr(runtime_config, "ACTIVE_CELL_IDS", []),
        active_radius_cells=getattr(runtime_config, "ACTIVE_RADIUS_CELLS", 0),
        active_ring_index=getattr(runtime_config, "ACTIVE_RING_INDEX", 0),
        random_seed=getattr(runtime_config, "RANDOM_SEED", None),
        filter_only_weed_classes=getattr(runtime_config, "FILTER_ONLY_WEED_CLASSES", True),
    )
    summary = summarize_filter_run(grid, targets, filter_info, {
        "experiment_grid_enabled": True,
        "trial_filter_enabled": getattr(runtime_config, "TRIAL_FILTER_ENABLED", False),
        "trial_filter_mode": getattr(runtime_config, "TRIAL_FILTER_MODE", "none"),
        "random_seed": getattr(runtime_config, "RANDOM_SEED", None),
        "requested_active_cell_count": getattr(runtime_config, "REQUESTED_ACTIVE_CELL_COUNT", 0),
    })
    print_filter_debug(
        getattr(runtime_config, "TRIAL_FILTER_MODE", "none"),
        grid,
        targets,
        filter_info,
        getattr(runtime_config, "REQUESTED_ACTIVE_CELL_COUNT", 0),
    )

    manifest = {
        "trial_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        **summary,
        "targets": targets,
    }
    out = Path(args.out) if args.out else ROOT / "experiments" / "metrics" / "grid_filter_dry_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[GridFilter] dry-run manifest saved: {out}")
    print(f"[GridFilter] selected target count: {len(selected)}")


if __name__ == "__main__":
    main()
