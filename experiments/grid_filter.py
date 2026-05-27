import math
import random


FILTER_REASONS = {
    "disabled": "filter_disabled",
    "none": "filter_mode_none",
    "inside_selected": "inside_selected_cell",
    "outside_selected": "outside_selected_cell",
    "inside_radius": "inside_radius",
    "outside_radius": "outside_radius",
    "inside_ring": "inside_ring",
    "outside_ring": "outside_ring",
}


class WorkspaceGrid:
    def __init__(
        self,
        rows=5,
        cols=7,
        x_min_mm=5.0,
        x_max_mm=420.0,
        y_min_mm=5.0,
        y_max_mm=420.0,
        survey_origin_x_mm=200.0,
        survey_origin_y_mm=150.0,
    ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.x_min_mm = float(x_min_mm)
        self.x_max_mm = float(x_max_mm)
        self.y_min_mm = float(y_min_mm)
        self.y_max_mm = float(y_max_mm)
        self.survey_origin_x_mm = float(survey_origin_x_mm)
        self.survey_origin_y_mm = float(survey_origin_y_mm)
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Grid rows and columns must be positive.")
        if self.x_max_mm <= self.x_min_mm or self.y_max_mm <= self.y_min_mm:
            raise ValueError("Grid max bounds must be greater than min bounds.")
        self.cell_width_mm = (self.x_max_mm - self.x_min_mm) / self.cols
        self.cell_height_mm = (self.y_max_mm - self.y_min_mm) / self.rows
        self.survey_row, self.survey_col = self.row_col_for_xy(
            self.survey_origin_x_mm,
            self.survey_origin_y_mm,
        )
        self.survey_cell_id = self.cell_id(self.survey_row, self.survey_col)

    def row_col_for_xy(self, x_mm, y_mm):
        col = math.floor((float(x_mm) - self.x_min_mm) / self.cell_width_mm)
        row = math.floor((float(y_mm) - self.y_min_mm) / self.cell_height_mm)
        row = max(0, min(self.rows - 1, int(row)))
        col = max(0, min(self.cols - 1, int(col)))
        return row, col

    def cell_id(self, row, col):
        return f"r{int(row)}_c{int(col)}"

    def cell_center(self, row, col):
        return (
            self.x_min_mm + (int(col) + 0.5) * self.cell_width_mm,
            self.y_min_mm + (int(row) + 0.5) * self.cell_height_mm,
        )

    def metadata_for_xy(self, x_mm, y_mm):
        x = float(x_mm)
        y = float(y_mm)
        row, col = self.row_col_for_xy(x, y)
        center_x, center_y = self.cell_center(row, col)
        drow = row - self.survey_row
        dcol = col - self.survey_col
        radius = math.hypot(x - self.survey_origin_x_mm, y - self.survey_origin_y_mm)
        angle = math.degrees(math.atan2(
            y - self.survey_origin_y_mm,
            x - self.survey_origin_x_mm,
        ))

        if drow == 0 and dcol == 0:
            axis_label = "off_axis"
        elif drow == 0:
            axis_label = "horizontal"
        elif dcol == 0:
            axis_label = "vertical"
        elif abs(drow) == abs(dcol):
            axis_label = "diagonal"
        else:
            axis_label = "off_axis"

        if drow == 0 or dcol == 0:
            quadrant_label = "center_axis"
        elif drow > 0 and dcol > 0:
            quadrant_label = "upper_right"
        elif drow > 0 and dcol < 0:
            quadrant_label = "upper_left"
        elif drow < 0 and dcol > 0:
            quadrant_label = "lower_right"
        else:
            quadrant_label = "lower_left"

        return {
            "cell_id": self.cell_id(row, col),
            "cell_row": row,
            "cell_col": col,
            "cell_center_x_mm": round(center_x, 3),
            "cell_center_y_mm": round(center_y, 3),
            "distance_from_cell_center_mm": round(math.hypot(x - center_x, y - center_y), 3),
            "radius_from_survey_mm": round(radius, 3),
            "angle_from_survey_deg": round(angle, 3),
            "ring_index": max(abs(drow), abs(dcol)),
            "axis_label": axis_label,
            "quadrant_label": quadrant_label,
        }

    def run_metadata(self):
        return {
            "grid_rows": self.rows,
            "grid_cols": self.cols,
            "grid_x_min_mm": self.x_min_mm,
            "grid_x_max_mm": self.x_max_mm,
            "grid_y_min_mm": self.y_min_mm,
            "grid_y_max_mm": self.y_max_mm,
            "cell_width_mm": round(self.cell_width_mm, 3),
            "cell_height_mm": round(self.cell_height_mm, 3),
            "survey_origin_x_mm": self.survey_origin_x_mm,
            "survey_origin_y_mm": self.survey_origin_y_mm,
            "survey_origin_cell_id": self.survey_cell_id,
            "survey_origin_cell_row": self.survey_row,
            "survey_origin_cell_col": self.survey_col,
        }


def make_grid_from_config(config_module):
    return WorkspaceGrid(
        rows=getattr(config_module, "GRID_ROWS", 5),
        cols=getattr(config_module, "GRID_COLS", 7),
        x_min_mm=getattr(config_module, "GRID_X_MIN_MM", 0.0),
        x_max_mm=getattr(config_module, "GRID_X_MAX_MM", 420.0),
        y_min_mm=getattr(config_module, "GRID_Y_MIN_MM", 0.0),
        y_max_mm=getattr(config_module, "GRID_Y_MAX_MM", 420.0),
        survey_origin_x_mm=getattr(config_module, "SURVEY_ORIGIN_X_MM", 200.0),
        survey_origin_y_mm=getattr(config_module, "SURVEY_ORIGIN_Y_MM", 150.0),
    )


def assign_grid_metadata(targets, grid):
    for idx, target in enumerate(targets or [], start=1):
        xy = target.get("target_xy_mm") or (target.get("x_target_mm"), target.get("y_target_mm"))
        if xy is None or xy[0] is None or xy[1] is None:
            continue
        target.setdefault("target_id", idx)
        target.update(grid.metadata_for_xy(xy[0], xy[1]))
        target.setdefault("was_selected_by_trial_filter", True)
        target.setdefault("selection_reason", FILTER_REASONS["disabled"])
    return targets


def _weed_targets(targets, filter_only_weed_classes):
    if not filter_only_weed_classes:
        return list(targets or [])
    return [
        t for t in (targets or [])
        if not t.get("source_target") or t.get("source_target", {}).get("left_cls") is not None
    ]


def apply_trial_filter(
    targets,
    enabled=False,
    mode="none",
    requested_active_cell_count=0,
    active_cell_ids=None,
    active_radius_cells=0,
    active_ring_index=0,
    random_seed=None,
    filter_only_weed_classes=True,
    eligible_target_fn=None,
):
    targets = list(targets or [])
    active_cell_ids = list(active_cell_ids or [])
    mode = str(mode or "none").strip().lower()
    warnings = []

    if not enabled:
        for target in targets:
            target["was_selected_by_trial_filter"] = True
            target["selection_reason"] = FILTER_REASONS["disabled"]
        return targets, {
            "selected_targets": targets,
            "rejected_targets": [],
            "occupied_cell_ids": sorted({t.get("cell_id") for t in targets if t.get("cell_id")}),
            "selected_cell_ids": sorted({t.get("cell_id") for t in targets if t.get("cell_id")}),
            "rejected_cell_ids": [],
            "warnings": warnings,
        }

    eligible = _weed_targets(targets, filter_only_weed_classes)
    if eligible_target_fn is not None:
        eligible = [t for t in eligible if eligible_target_fn(t)]
    occupied = sorted({t.get("cell_id") for t in eligible if t.get("cell_id")})
    selected_cells = set(occupied)
    filter_attempts = 1
    selection_strategy = None

    if mode == "none":
        for target in targets:
            target["was_selected_by_trial_filter"] = True
            target["selection_reason"] = FILTER_REASONS["none"]
    elif mode == "random_cells":
        count = max(0, int(requested_active_cell_count or 0))
        rng = random.Random(random_seed)
        target_goal = min(count, len(eligible))
        by_cell = _targets_by_cell(eligible)
        attempts = 0
        max_attempts = 200

        # Requested count is a target-count goal first. If the request is at or
        # above the available matched/eligible targets, keep every eligible target
        # even when multiple targets share a cell. This makes high-count trials
        # fail gracefully because stereo matching, not the grid, is the limiter.
        if count >= len(eligible):
            selected_cells = set(occupied)
            selection_strategy = "all_eligible_targets"
            if count > len(eligible):
                warnings.append(
                    f"requested {count} target(s) but only {len(eligible)} eligible matched target(s) exist"
                )
        elif count >= len(occupied):
            selected_cells = set(occupied)
            selection_strategy = "all_occupied_cells"
            if count > len(occupied):
                warnings.append(
                    f"requested {count} target(s); all {len(occupied)} occupied eligible cell(s) were selected, covering {len(eligible)} eligible matched target(s)"
                )
        else:
            selection_strategy = "random_cells_target_goal"
            best_cells = set()
            best_target_count = -1
            while attempts < max_attempts:
                attempts += 1
                candidate = set(rng.sample(occupied, count))
                candidate_count = _selected_count_for_cells(by_cell, candidate)
                if candidate_count > best_target_count:
                    best_cells = candidate
                    best_target_count = candidate_count
                if candidate_count >= target_goal:
                    selected_cells = candidate
                    break
            else:
                selected_cells = _top_up_cells(rng, by_cell, best_cells, target_goal)
                warnings.append(
                    f"random cell sample could not cover {target_goal} target(s) in {max_attempts} attempt(s); topped up selected cells to prioritize target count"
                )
        _mark_by_cells(targets, selected_cells, eligible)
        if count < len(eligible):
            selected_now = [t for t in targets if t.get("was_selected_by_trial_filter")]
            if len(selected_now) > count:
                # Drop extras from multi-target cells first (keep one random per cell)
                by_cell_sel = {}
                for t in selected_now:
                    by_cell_sel.setdefault(t.get("cell_id"), []).append(t)
                to_drop = []
                for cell_targets in by_cell_sel.values():
                    if len(cell_targets) > 1:
                        keep = rng.choice(cell_targets)
                        to_drop.extend(t for t in cell_targets if t is not keep)
                # If still over count, randomly drop from remaining selected
                still_over = len(selected_now) - len(to_drop) - count
                if still_over > 0:
                    drop_ids_so_far = {id(t) for t in to_drop}
                    remaining = [t for t in selected_now if id(t) not in drop_ids_so_far]
                    to_drop.extend(rng.sample(remaining, still_over))
                drop_ids = {id(t) for t in to_drop}
                for t in targets:
                    if id(t) in drop_ids:
                        t["was_selected_by_trial_filter"] = False
                        t["selection_reason"] = "trimmed_to_target_count"
        filter_attempts = attempts if count < len(occupied) else 1
    elif mode == "custom_cells":
        selected_cells = set(active_cell_ids)
        _mark_by_cells(targets, selected_cells, eligible)
        filter_attempts = 1
    elif mode == "radius_cells":
        radius = int(active_radius_cells or 0)
        for target in targets:
            selected = int(target.get("ring_index", 0)) <= radius
            target["was_selected_by_trial_filter"] = selected
            target["selection_reason"] = (
                FILTER_REASONS["inside_radius"] if selected else FILTER_REASONS["outside_radius"]
            )
        selected_cells = {t.get("cell_id") for t in targets if t.get("was_selected_by_trial_filter")}
        filter_attempts = 1
    elif mode == "ring":
        ring = int(active_ring_index or 0)
        for target in targets:
            selected = int(target.get("ring_index", 0)) == ring
            target["was_selected_by_trial_filter"] = selected
            target["selection_reason"] = (
                FILTER_REASONS["inside_ring"] if selected else FILTER_REASONS["outside_ring"]
            )
        selected_cells = {t.get("cell_id") for t in targets if t.get("was_selected_by_trial_filter")}
        filter_attempts = 1
    else:
        warnings.append(f"unknown trial filter mode {mode!r}; keeping all targets")
        for target in targets:
            target["was_selected_by_trial_filter"] = True
            target["selection_reason"] = "unknown_mode_keep_all"
        selected_cells = set(occupied)
        filter_attempts = 1

    selected = [t for t in targets if t.get("was_selected_by_trial_filter")]
    rejected = [t for t in targets if not t.get("was_selected_by_trial_filter")]
    return selected, {
        "selected_targets": selected,
        "rejected_targets": rejected,
        "occupied_cell_ids": occupied,
        "selected_cell_ids": sorted(c for c in selected_cells if c),
        "rejected_cell_ids": sorted({t.get("cell_id") for t in rejected if t.get("cell_id")}),
        "eligible_target_count": len(eligible),
        "requested_target_goal": min(int(requested_active_cell_count or 0), len(eligible)) if mode == "random_cells" else None,
        "random_selection_attempts": filter_attempts,
        "selection_strategy": selection_strategy,
        "selected_cell_target_counts": _cell_target_counts(eligible, selected_cells),
        "selected_cell_target_ids": _cell_target_ids(eligible, selected_cells),
        "warnings": warnings,
    }


def _mark_by_cells(targets, selected_cells, eligible_targets=None):
    eligible_ids = None
    if eligible_targets is not None:
        eligible_ids = {id(t) for t in eligible_targets}
    for target in targets:
        selected = target.get("cell_id") in selected_cells
        ineligible = eligible_ids is not None and id(target) not in eligible_ids
        if ineligible:
            selected = False
        target["was_selected_by_trial_filter"] = selected
        if selected:
            target["selection_reason"] = FILTER_REASONS["inside_selected"]
        elif ineligible:
            target["selection_reason"] = "not_eligible_for_trial"
        else:
            target["selection_reason"] = FILTER_REASONS["outside_selected"]


def _targets_by_cell(targets):
    by_cell = {}
    for target in targets:
        cell_id = target.get("cell_id")
        if cell_id:
            by_cell.setdefault(cell_id, []).append(target)
    return by_cell


def _selected_count_for_cells(by_cell, selected_cells):
    return sum(len(by_cell.get(cell_id, [])) for cell_id in selected_cells)


def _top_up_cells(rng, by_cell, selected_cells, target_goal):
    selected = set(selected_cells)
    cells = list(by_cell.keys())
    rng.shuffle(cells)
    cells.sort(key=lambda c: len(by_cell.get(c, [])), reverse=True)
    for cell_id in cells:
        if _selected_count_for_cells(by_cell, selected) >= target_goal:
            break
        selected.add(cell_id)
    return selected


def _cell_target_counts(targets, selected_cells):
    by_cell = _targets_by_cell(targets)
    return {cell_id: len(by_cell.get(cell_id, [])) for cell_id in sorted(selected_cells) if cell_id}


def _cell_target_ids(targets, selected_cells):
    by_cell = _targets_by_cell(targets)
    return {
        cell_id: [t.get("target_id") for t in by_cell.get(cell_id, [])]
        for cell_id in sorted(selected_cells)
        if cell_id
    }


def summarize_filter_run(grid, all_targets, filter_info, config_values):
    selected = filter_info.get("selected_targets", [])
    rejected = filter_info.get("rejected_targets", [])
    radii = [float(t["radius_from_survey_mm"]) for t in selected if t.get("radius_from_survey_mm") is not None]
    center_dists = [
        float(t["distance_from_cell_center_mm"])
        for t in selected
        if t.get("distance_from_cell_center_mm") is not None
    ]
    points = [
        tuple(t["target_xy_mm"])
        for t in selected
        if t.get("target_xy_mm") is not None
    ]
    summary = {
        **grid.run_metadata(),
        **config_values,
        "surveyed_target_count": len(all_targets or []),
        "selected_target_count": len(selected),
        "rejected_target_count": len(rejected),
        "occupied_cell_count": len(filter_info.get("occupied_cell_ids", [])),
        "occupied_cell_ids": filter_info.get("occupied_cell_ids", []),
        "selected_cell_ids": filter_info.get("selected_cell_ids", []),
        "rejected_cell_ids": filter_info.get("rejected_cell_ids", []),
        "eligible_target_count": filter_info.get("eligible_target_count"),
        "requested_target_goal": filter_info.get("requested_target_goal"),
        "random_selection_attempts": filter_info.get("random_selection_attempts"),
        "selected_cell_target_counts": filter_info.get("selected_cell_target_counts"),
        "selected_cell_target_ids": filter_info.get("selected_cell_target_ids"),
        "mean_selected_radius_mm": _mean(radii),
        "max_selected_radius_mm": round(max(radii), 3) if radii else None,
        "mean_selected_distance_from_cell_center_mm": _mean(center_dists),
        "selected_spread_mm": _spread(points),
        "selected_convex_hull_area_mm2": _convex_hull_area(points),
    }
    if filter_info.get("warnings"):
        summary["trial_filter_warnings"] = filter_info["warnings"]
    return summary


def print_filter_debug(mode, grid, all_targets, filter_info, requested_active_cell_count):
    selected = filter_info.get("selected_targets", [])
    rejected = filter_info.get("rejected_targets", [])
    print("\n=== GRID FILTER ===")
    print(f"  Mode                       {mode}")
    print(f"  Grid                       {grid.cols}x{grid.rows}")
    print(f"  Occupied eligible cells    {filter_info.get('occupied_cell_ids', [])}")
    print(f"  Requested target goal      {requested_active_cell_count}")
    if filter_info.get("requested_target_goal") is not None:
        print(f"  Target goal after limits    {filter_info.get('requested_target_goal')}")
    print(f"  Eligible matched targets   {filter_info.get('eligible_target_count', len(all_targets or []))}")
    if filter_info.get("selection_strategy"):
        print(f"  Selection strategy         {filter_info.get('selection_strategy')}")
    if filter_info.get("selection_strategy") == "random_cells_target_goal":
        print(f"  Random attempts            {filter_info.get('random_selection_attempts', 1)}")
    print(f"  Selected cells             {filter_info.get('selected_cell_ids', [])}")
    if filter_info.get("selected_cell_target_counts"):
        print("  Selected cell target ids")
        for cell_id in filter_info.get("selected_cell_ids", []):
            ids = filter_info.get("selected_cell_target_ids", {}).get(cell_id, [])
            count = filter_info.get("selected_cell_target_counts", {}).get(cell_id, 0)
            print(f"    {cell_id:<8} count={count:<2} targets={ids}")
    print(f"  Targets before filter      {len(all_targets or [])}")
    print(f"  Targets after filter       {len(selected)}")
    print(f"  Rejected targets           {len(rejected)}")
    for warning in filter_info.get("warnings", []):
        print(f"  WARNING                    {warning}")
    print("\n  target_id | x_mm    | y_mm    | cell_id | selected | reason")
    print("  ----------+---------+---------+---------+----------+------------------------")
    for target in all_targets or []:
        xy = target.get("target_xy_mm") or (None, None)
        print(
            f"  {str(target.get('target_id')):<9} | "
            f"{_fmt(xy[0]):>7} | {_fmt(xy[1]):>7} | "
            f"{str(target.get('cell_id')):<7} | "
            f"{str(bool(target.get('was_selected_by_trial_filter'))):<8} | "
            f"{target.get('selection_reason')}"
        )


def _fmt(value):
    if value is None:
        return "None"
    return f"{float(value):.1f}"


def _mean(values):
    return round(sum(values) / len(values), 3) if values else None


def _spread(points):
    if len(points) < 2:
        return 0.0 if points else None
    max_dist = 0.0
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            max_dist = max(max_dist, math.hypot(a[0] - b[0], a[1] - b[1]))
    return round(max_dist, 3)


def _convex_hull_area(points):
    points = sorted(set((float(x), float(y)) for x, y in points))
    if len(points) < 3:
        return 0.0 if points else None

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for i, p in enumerate(hull):
        q = hull[(i + 1) % len(hull)]
        area += p[0] * q[1] - q[0] * p[1]
    return round(abs(area) / 2.0, 3)
