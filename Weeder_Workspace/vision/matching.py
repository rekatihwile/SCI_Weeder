from ui.terminal import print_match_summary
import numpy as np


def _pair_score(y_diff, disp_mag, min_disp, max_disp, max_y_diff, match_dist, max_match_dist):
    y_score = max(0.0, 1.0 - y_diff / max(1.0, max_y_diff))

    d_center = 0.5 * (min_disp + max_disp)
    d_half = max(1.0, 0.5 * (max_disp - min_disp))
    d_score = max(0.0, 1.0 - abs(disp_mag - d_center) / d_half)

    m_score = max(0.0, 1.0 - match_dist / max(1.0, max_match_dist))

    return 0.35 * y_score + 0.25 * d_score + 0.40 * m_score


def match_points_constellation(
    left_points,
    right_points,
    anchor_min_disp=10,
    anchor_max_disp=220,
    anchor_max_y_diff=100,
    match_radius=28.0,
    min_score=0.35,
):
    """
    Constellation-based stereo matcher inspired by Bronny's match_and_optimize().

    Idea:
    - Try each plausible (left anchor, right anchor) pair.
    - Compute the implied stereo offset dx, dy.
    - Shift the entire left constellation by that offset.
    - Greedily assign nearest right points within a radius.
    - Keep the hypothesis with the most consistent matches.

    Returns:
        matched_targets, unmatched_left, unmatched_right
    """
    left_points = [tuple(map(int, p)) for p in left_points]
    right_points = [tuple(map(int, p)) for p in right_points]

    if not left_points or not right_points:
        return [], left_points, right_points

    pts_l = np.array(left_points, dtype=np.float32)
    pts_r = np.array(right_points, dtype=np.float32)

    best_cfg = {}
    best_meta = {}
    max_matches = -1
    best_total_cost = float("inf")

    for i, a_l in enumerate(pts_l):
        xl, yl = a_l

        for j, a_r in enumerate(pts_r):
            xr, yr = a_r

            dx = xr - xl
            dy = yr - yl

            disp_mag = abs(dx)
            y_diff = abs(dy)

            if not (anchor_min_disp <= disp_mag <= anchor_max_disp):
                continue
            if y_diff > anchor_max_y_diff:
                continue

            shifted_l = pts_l + np.array([dx, dy], dtype=np.float32)

            cfg = {}
            meta = {}
            used_right = set()
            total_cost = 0.0

            for l_idx, p_s in enumerate(shifted_l):
                dists = np.linalg.norm(pts_r - p_s, axis=1)

                order = np.argsort(dists)
                chosen_r = None
                chosen_dist = None

                for r_idx in order:
                    if r_idx in used_right:
                        continue
                    if dists[r_idx] <= match_radius:
                        chosen_r = int(r_idx)
                        chosen_dist = float(dists[r_idx])
                        break

                if chosen_r is None:
                    continue

                lp = tuple(map(int, pts_l[l_idx]))
                rp = tuple(map(int, pts_r[chosen_r]))

                y_pair_diff = abs(rp[1] - lp[1])
                disp_pair_mag = abs(rp[0] - lp[0])

                score = _pair_score(
                    y_pair_diff,
                    disp_pair_mag,
                    anchor_min_disp,
                    anchor_max_disp,
                    anchor_max_y_diff,
                    chosen_dist,
                    match_radius,
                )

                if score < min_score:
                    continue

                cfg[l_idx] = chosen_r
                meta[l_idx] = {
                    "score": float(score),
                    "match_dist": chosen_dist,
                }
                used_right.add(chosen_r)
                total_cost += chosen_dist

            num_matches = len(cfg)

            if num_matches > max_matches or (num_matches == max_matches and total_cost < best_total_cost):
                max_matches = num_matches
                best_total_cost = total_cost
                best_cfg = cfg
                best_meta = meta

    matched_targets = []
    matched_left_idx = set()
    matched_right_idx = set()

    for l_idx, r_idx in best_cfg.items():
        lp = tuple(map(int, pts_l[l_idx]))
        rp = tuple(map(int, pts_r[r_idx]))

        matched_targets.append({
            "left_px": lp,
            "right_px": rp,
            "score": float(best_meta[l_idx]["score"]),
        })

        matched_left_idx.add(l_idx)
        matched_right_idx.add(r_idx)

    unmatched_left = [
        tuple(map(int, pts_l[i]))
        for i in range(len(pts_l))
        if i not in matched_left_idx
    ]

    unmatched_right = [
        tuple(map(int, pts_r[i]))
        for i in range(len(pts_r))
        if i not in matched_right_idx
    ]

    return matched_targets, unmatched_left, unmatched_right


def match_points(left_points, right_points, verbose=False):
    print("\n=== MATCH ===")

    matched_targets, unmatched_left, unmatched_right = match_points_constellation(
        left_points,
        right_points,
        anchor_min_disp=5,
        anchor_max_disp=250,
        anchor_max_y_diff=150,
        match_radius=30.0,
        min_score=0.35,
    )

    if verbose:
        print_match_summary(matched_targets, unmatched_left, unmatched_right)

    return matched_targets, unmatched_left, unmatched_right