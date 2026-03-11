import math


def _get_xy(target):
    if "target_xy_mm" in target:
        return target["target_xy_mm"]
    if "x_mm" in target and "y_mm" in target:
        return target["x_mm"], target["y_mm"]
    raise KeyError("Target needs either target_xy_mm or x_mm/y_mm.")


def plan_targets(targets, start_xy=None):
    print("\n=== PLAN ===")

    if not targets:
        print("No matched targets to plan.")
        return []

    remaining = list(targets)
    planned = []

    if start_xy is None:
        current_x, current_y = _get_xy(remaining[0])
    else:
        current_x, current_y = start_xy

    while remaining:
        next_target = min(
            remaining,
            key=lambda t: math.hypot(
                _get_xy(t)[0] - current_x,
                _get_xy(t)[1] - current_y,
            ),
        )
        planned.append(next_target)
        remaining.remove(next_target)
        current_x, current_y = _get_xy(next_target)

    print(f"Planned {len(planned)} targets.")
    return planned