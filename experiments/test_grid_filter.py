import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.grid_filter import WorkspaceGrid, apply_trial_filter, assign_grid_metadata


def _targets(points):
    return [
        {
            "target_id": i,
            "target_xy_mm": xy,
            "source_target": {"left_cls": 1},
        }
        for i, xy in enumerate(points, start=1)
    ]


def test_cell_assignment():
    grid = WorkspaceGrid(rows=5, cols=7, x_min_mm=0, x_max_mm=700, y_min_mm=0, y_max_mm=500,
                         survey_origin_x_mm=350, survey_origin_y_mm=250)
    assert grid.metadata_for_xy(0, 0)["cell_id"] == "r0_c0"
    assert grid.metadata_for_xy(99.9, 99.9)["cell_id"] == "r0_c0"
    assert grid.metadata_for_xy(100, 100)["cell_id"] == "r1_c1"
    assert grid.metadata_for_xy(699.9, 499.9)["cell_id"] == "r4_c6"
    assert grid.metadata_for_xy(700, 500)["cell_id"] == "r4_c6"
    assert grid.metadata_for_xy(-5, -1)["cell_id"] == "r0_c0"


def test_random_reproducible():
    grid = WorkspaceGrid(rows=5, cols=7, x_min_mm=0, x_max_mm=700, y_min_mm=0, y_max_mm=500,
                         survey_origin_x_mm=350, survey_origin_y_mm=250)
    targets_a = assign_grid_metadata(_targets([(10, 10), (110, 10), (210, 10), (310, 10)]), grid)
    targets_b = assign_grid_metadata(_targets([(10, 10), (110, 10), (210, 10), (310, 10)]), grid)
    _, info_a = apply_trial_filter(
        targets_a, enabled=True, mode="random_cells", requested_active_cell_count=2, random_seed=42
    )
    _, info_b = apply_trial_filter(
        targets_b, enabled=True, mode="random_cells", requested_active_cell_count=2, random_seed=42
    )
    assert info_a["selected_cell_ids"] == info_b["selected_cell_ids"]
    assert len(info_a["selected_cell_ids"]) == 2


def test_random_prioritizes_requested_eligible_targets():
    grid = WorkspaceGrid(rows=5, cols=7, x_min_mm=0, x_max_mm=700, y_min_mm=0, y_max_mm=500,
                         survey_origin_x_mm=350, survey_origin_y_mm=250)
    targets = assign_grid_metadata(_targets([(-10, 10), (110, 10), (210, 10)]), grid)
    selected, info = apply_trial_filter(
        targets,
        enabled=True,
        mode="random_cells",
        requested_active_cell_count=2,
        random_seed=7,
        eligible_target_fn=lambda t: t["target_xy_mm"][0] >= 0,
    )
    assert len(selected) == 2
    assert info["requested_target_goal"] == 2
    assert targets[0]["selection_reason"] != "inside_selected_cell"


def test_random_selects_all_available_when_request_is_high():
    grid = WorkspaceGrid(rows=5, cols=7, x_min_mm=0, x_max_mm=700, y_min_mm=0, y_max_mm=500,
                         survey_origin_x_mm=350, survey_origin_y_mm=250)
    targets = assign_grid_metadata(_targets([(10, 10), (20, 20), (110, 10)]), grid)
    selected, info = apply_trial_filter(
        targets,
        enabled=True,
        mode="random_cells",
        requested_active_cell_count=15,
        random_seed=99,
    )
    assert len(selected) == 3
    assert info["requested_target_goal"] == 3
    assert info["selection_strategy"] == "all_eligible_targets"
    assert all(t["was_selected_by_trial_filter"] for t in targets)


def test_custom_radius_ring():
    grid = WorkspaceGrid(rows=5, cols=7, x_min_mm=0, x_max_mm=700, y_min_mm=0, y_max_mm=500,
                         survey_origin_x_mm=350, survey_origin_y_mm=250)
    targets = assign_grid_metadata(_targets([(350, 250), (450, 250), (650, 450)]), grid)

    selected, _ = apply_trial_filter(targets, enabled=True, mode="custom_cells", active_cell_ids=["r2_c3"])
    assert [t["target_id"] for t in selected] == [1]
    assert targets[1]["selection_reason"] == "outside_selected_cell"

    targets = assign_grid_metadata(_targets([(350, 250), (450, 250), (650, 450)]), grid)
    selected, _ = apply_trial_filter(targets, enabled=True, mode="radius_cells", active_radius_cells=1)
    assert [t["target_id"] for t in selected] == [1, 2]
    assert targets[2]["selection_reason"] == "outside_radius"

    targets = assign_grid_metadata(_targets([(350, 250), (450, 250), (650, 450)]), grid)
    selected, _ = apply_trial_filter(targets, enabled=True, mode="ring", active_ring_index=1)
    assert [t["target_id"] for t in selected] == [2]
    assert targets[0]["selection_reason"] == "outside_ring"


def main():
    test_cell_assignment()
    test_random_reproducible()
    test_random_prioritizes_requested_eligible_targets()
    test_random_selects_all_available_when_request_is_high()
    test_custom_radius_ring()
    print("grid_filter tests passed")


if __name__ == "__main__":
    main()
