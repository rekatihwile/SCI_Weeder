from ui.terminal import print_match_summary


def _pair_score(y_diff, disp_mag, min_disp, max_disp, max_y_diff):
    y_score = max(0.0, 1.0 - y_diff / max(1, max_y_diff))

    d_center = 0.5 * (min_disp + max_disp)
    d_half = max(1.0, 0.5 * (max_disp - min_disp))
    d_score = max(0.0, 1.0 - abs(disp_mag - d_center) / d_half)

    return 0.65 * y_score + 0.35 * d_score


def match_points_stereo(
    left_points,
    right_points,
    max_y_diff=18,
    min_disp=15,
    max_disp=260,
    min_score=0.55,
):
    """
    Less aggressive stereo matcher for manual clicking.

    It does NOT force one disparity sign.
    It matches based on:
    - small vertical difference
    - plausible disparity magnitude
    - one-to-one assignment

    Returns:
        matched_targets, unmatched_left, unmatched_right
    """
    left_points = [tuple(map(int, p)) for p in left_points]
    right_points = [tuple(map(int, p)) for p in right_points]

    matched_targets = []
    unmatched_left = []
    used_right = set()

    # Sort by y then x so similar rows stay grouped
    left_sorted = sorted(left_points, key=lambda p: (p[1], p[0]))
    right_sorted = sorted(right_points, key=lambda p: (p[1], p[0]))

    for lp in left_sorted:
        xl, yl = lp

        best_idx = None
        best_cost = None
        best_score = 0.0

        for j, rp in enumerate(right_sorted):
            if j in used_right:
                continue

            xr, yr = rp
            y_diff = abs(yr - yl)
            disp_mag = abs(xr - xl)

            if y_diff > max_y_diff:
                continue
            if not (min_disp <= disp_mag <= max_disp):
                continue

            # lower cost is better
            cost = y_diff + 0.03 * abs(disp_mag - 0.5 * (min_disp + max_disp))
            score = _pair_score(y_diff, disp_mag, min_disp, max_disp, max_y_diff)

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_idx = j
                best_score = score

        if best_idx is not None and best_score >= min_score:
            used_right.add(best_idx)
            matched_targets.append(
                {
                    "left_px": lp,
                    "right_px": right_sorted[best_idx],
                    "score": float(best_score),
                }
            )
        else:
            unmatched_left.append(lp)

    unmatched_right = [
        rp for j, rp in enumerate(right_sorted)
        if j not in used_right
    ]

    return matched_targets, unmatched_left, unmatched_right


def match_points(left_points, right_points, verbose=False):
    print("\n=== MATCH ===")

    matched_targets, unmatched_left, unmatched_right = match_points_stereo(
        left_points,
        right_points,
        max_y_diff=100,
        min_disp=10,
        max_disp=200,
        min_score=0.45,
    )

    if verbose:
        print_match_summary(matched_targets, unmatched_left, unmatched_right)

    return matched_targets, unmatched_left, unmatched_right