from ui.terminal import print_match_summary
import numpy as np

from config import (
    STEREO_MATCH_MIN_DISPARITY_PX,
    STEREO_MATCH_MAX_DISPARITY_PX,
    STEREO_MATCH_MAX_Y_DIFF_PX,
    STEREO_MATCH_RADIUS_PX,
    STEREO_MATCH_MIN_SCORE,
    STEREO_MATCH_MIN_BOX_IOU,
    STEREO_MATCH_IOU_WEIGHT,
)


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------

def _box_iou(b1, b2):
    """Standard IoU between two (x1, y1, x2, y2) boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


def _box_center(b):
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def _shift_box(b, dx, dy):
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


# ---------------------------------------------------------------------------
# Point-only fallback scorer (used when no boxes are available)
# ---------------------------------------------------------------------------

def _pair_score(y_diff, disp_mag, min_disp, max_disp, max_y_diff, match_dist, max_match_dist):
    y_score = max(0.0, 1.0 - y_diff / max(1.0, max_y_diff))

    d_center = 0.5 * (min_disp + max_disp)
    d_half   = max(1.0, 0.5 * (max_disp - min_disp))
    d_score  = max(0.0, 1.0 - abs(disp_mag - d_center) / d_half)

    m_score  = max(0.0, 1.0 - match_dist / max(1.0, max_match_dist))

    return 0.35 * y_score + 0.25 * d_score + 0.40 * m_score


# ---------------------------------------------------------------------------
# Box-IoU constellation matcher
# ---------------------------------------------------------------------------

def match_points_constellation(
    left_detections,
    right_detections,
    anchor_min_disp=STEREO_MATCH_MIN_DISPARITY_PX,
    anchor_max_disp=STEREO_MATCH_MAX_DISPARITY_PX,
    anchor_max_y_diff=STEREO_MATCH_MAX_Y_DIFF_PX,
    match_radius=STEREO_MATCH_RADIUS_PX,
    min_score=STEREO_MATCH_MIN_SCORE,
    # box matching params (only used when boxes are present)
    min_box_iou=STEREO_MATCH_MIN_BOX_IOU,
    iou_weight=STEREO_MATCH_IOU_WEIGHT,
):
    """
    Constellation-based stereo matcher.

    Accepts two list formats (auto-detected):

    A) Rich format (from AI detector):
        [{"point": (x, y), "box": (x1, y1, x2, y2), "views": int}, ...]

    B) Legacy format (plain points, e.g. from ManualDetectorLocal):
        [(x, y), ...]

    When boxes are present, the anchor hypothesis shifts the entire left box
    constellation and scores matches by box IoU rather than pixel distance,
    giving a much stronger stereo consistency signal.

    When boxes are absent, behaviour falls back to the original point-radius
    constellation approach.
    """
    # ---- normalise inputs ----
    def _normalise(dets):
        if not dets:
            return [], []
        if isinstance(dets[0], dict):
            pts  = [d["point"] for d in dets]
            boxes = [d["box"]  for d in dets]
        else:
            pts   = [tuple(map(int, p)) for p in dets]
            boxes = None
        return pts, boxes

    left_pts,  left_boxes  = _normalise(left_detections)
    right_pts, right_boxes = _normalise(right_detections)

    have_boxes = (left_boxes is not None) and (right_boxes is not None)

    if not left_pts or not right_pts:
        return [], list(left_pts), list(right_pts)

    pts_l = np.array(left_pts,  dtype=np.float32)
    pts_r = np.array(right_pts, dtype=np.float32)

    best_cfg        = {}
    best_meta       = {}
    max_matches     = -1
    best_total_cost = float("inf")

    # ---- try every plausible anchor pair ----
    for i in range(len(pts_l)):
        xl, yl = pts_l[i]

        for j in range(len(pts_r)):
            xr, yr = pts_r[j]

            dx = xr - xl
            dy = yr - yl

            disp_mag = abs(dx)
            y_diff   = abs(dy)

            if not (anchor_min_disp <= disp_mag <= anchor_max_disp):
                continue
            if y_diff > anchor_max_y_diff:
                continue

            # shift the entire left constellation by this anchor offset
            shifted_pts_l = pts_l + np.array([dx, dy], dtype=np.float32)

            cfg       = {}
            meta      = {}
            used_right = set()
            total_cost = 0.0

            for l_idx, p_s in enumerate(shifted_pts_l):

                if have_boxes:
                    # ---- BOX IoU path ----
                    shifted_box_l = _shift_box(left_boxes[l_idx], dx, dy)

                    best_r_idx  = None
                    best_iou    = 0.0

                    for r_idx in range(len(pts_r)):
                        if r_idx in used_right:
                            continue
                        iou = _box_iou(shifted_box_l, right_boxes[r_idx])
                        if iou > best_iou:
                            best_iou    = iou
                            best_r_idx  = r_idx

                    if best_r_idx is None or best_iou < min_box_iou:
                        continue

                    # additional geometric sanity on the unshifted points
                    lp = tuple(map(int, pts_l[l_idx]))
                    rp = tuple(map(int, pts_r[best_r_idx]))
                    y_pair_diff  = abs(rp[1] - lp[1])
                    disp_pair_x  = rp[0] - lp[0]

                    if not (anchor_min_disp <= abs(disp_pair_x) <= anchor_max_disp):
                        continue
                    if y_pair_diff > anchor_max_y_diff:
                        continue

                    # score: blend IoU with y-alignment
                    y_score   = max(0.0, 1.0 - y_pair_diff / max(1.0, anchor_max_y_diff))
                    score     = iou_weight * best_iou + (1.0 - iou_weight) * y_score

                    if score < min_score:
                        continue

                    cfg[l_idx]  = best_r_idx
                    meta[l_idx] = {"score": float(score), "iou": float(best_iou)}
                    used_right.add(best_r_idx)
                    total_cost += (1.0 - best_iou)   # lower cost = better IoU

                else:
                    # ---- POINT RADIUS fallback path ----
                    dists = np.linalg.norm(pts_r - p_s, axis=1)
                    order = np.argsort(dists)

                    chosen_r    = None
                    chosen_dist = None

                    for r_idx in order:
                        if r_idx in used_right:
                            continue
                        if dists[r_idx] <= match_radius:
                            chosen_r    = int(r_idx)
                            chosen_dist = float(dists[r_idx])
                            break

                    if chosen_r is None:
                        continue

                    lp = tuple(map(int, pts_l[l_idx]))
                    rp = tuple(map(int, pts_r[chosen_r]))
                    y_pair_diff  = abs(rp[1] - lp[1])
                    disp_pair_mag = abs(rp[0] - lp[0])

                    score = _pair_score(
                        y_pair_diff, disp_pair_mag,
                        anchor_min_disp, anchor_max_disp, anchor_max_y_diff,
                        chosen_dist, match_radius,
                    )

                    if score < min_score:
                        continue

                    cfg[l_idx]  = chosen_r
                    meta[l_idx] = {"score": float(score), "match_dist": chosen_dist}
                    used_right.add(chosen_r)
                    total_cost += chosen_dist

            num_matches = len(cfg)

            if num_matches > max_matches or (
                num_matches == max_matches and total_cost < best_total_cost
            ):
                max_matches     = num_matches
                best_total_cost = total_cost
                best_cfg        = cfg
                best_meta       = meta

    # ---- assemble output ----
    matched_targets    = []
    matched_left_idx   = set()
    matched_right_idx  = set()

    for l_idx, r_idx in best_cfg.items():
        lp = tuple(map(int, pts_l[l_idx]))
        rp = tuple(map(int, pts_r[r_idx]))

        entry = {
            "left_px":  lp,
            "right_px": rp,
            "score":    float(best_meta[l_idx]["score"]),
            "y_diff_px": float(abs(rp[1] - lp[1])),
            "disp_px":   float(abs(rp[0] - lp[0])),
            "dx_px":     float(rp[0] - lp[0]),
            "dy_px":     float(rp[1] - lp[1]),
        }
        if have_boxes:
            entry["left_box"]  = left_boxes[l_idx]
            entry["right_box"] = right_boxes[r_idx]
            entry["box_iou"]   = float(best_meta[l_idx].get("iou", 0.0))
            # Thread class + confidence through so downstream code can filter by plant type
            ld = left_detections[l_idx]
            rd = right_detections[r_idx]
            if isinstance(ld, dict):
                entry["left_cls"]  = ld.get("cls")
                entry["left_conf"] = ld.get("conf")
            if isinstance(rd, dict):
                entry["right_cls"]  = rd.get("cls")
                entry["right_conf"] = rd.get("conf")

        matched_targets.append(entry)
        matched_left_idx.add(l_idx)
        matched_right_idx.add(r_idx)

    unmatched_left  = [tuple(map(int, pts_l[i])) for i in range(len(pts_l))  if i not in matched_left_idx]
    unmatched_right = [tuple(map(int, pts_r[i])) for i in range(len(pts_r)) if i not in matched_right_idx]

    return matched_targets, unmatched_left, unmatched_right


# ---------------------------------------------------------------------------
# Public entry point (called from main.py / coarse_move.py)
# ---------------------------------------------------------------------------

def match_points(
    left_detections,
    right_detections,
    verbose=False,
    anchor_min_disp=STEREO_MATCH_MIN_DISPARITY_PX,
    anchor_max_disp=STEREO_MATCH_MAX_DISPARITY_PX,
    anchor_max_y_diff=STEREO_MATCH_MAX_Y_DIFF_PX,
    match_radius=STEREO_MATCH_RADIUS_PX,
    min_score=STEREO_MATCH_MIN_SCORE,
    min_box_iou=STEREO_MATCH_MIN_BOX_IOU,
    iou_weight=STEREO_MATCH_IOU_WEIGHT,
):
    print("\n=== MATCH ===")

    # Log whether we are running in box mode or point-only mode
    have_boxes = (
        left_detections
        and right_detections
        and isinstance(left_detections[0], dict)
        and isinstance(right_detections[0], dict)
    )
    print(f"[MATCH] mode: {'box-IoU constellation' if have_boxes else 'point-radius constellation (no boxes)'}")
    print(
        f"[MATCH] gates: y_diff<={anchor_max_y_diff:.1f}px "
        f"disp={anchor_min_disp:.1f}-{anchor_max_disp:.1f}px "
        f"score>={min_score:.2f}"
    )

    matched_targets, unmatched_left, unmatched_right = match_points_constellation(
        left_detections,
        right_detections,
        anchor_min_disp=anchor_min_disp,
        anchor_max_disp=anchor_max_disp,
        anchor_max_y_diff=anchor_max_y_diff,
        match_radius=match_radius,
        min_score=min_score,
        min_box_iou=min_box_iou,
        iou_weight=iou_weight,
    )

    kept = []
    rejected_bad_y = []
    for match in matched_targets:
        y_diff = float(match.get("y_diff_px", abs(match["left_px"][1] - match["right_px"][1])))
        if y_diff <= anchor_max_y_diff:
            kept.append(match)
        else:
            rejected_bad_y.append(match)

    if rejected_bad_y:
        print(
            f"[MATCH WARNING] rejected {len(rejected_bad_y)} match(es) after final y gate "
            f"y_diff>{anchor_max_y_diff:.1f}px"
        )
        matched_left = {m["left_px"] for m in kept}
        matched_right = {m["right_px"] for m in kept}

        def _as_pt(det):
            return tuple(det["point"]) if isinstance(det, dict) else tuple(det)

        unmatched_left = [_as_pt(d) for d in left_detections if _as_pt(d) not in matched_left]
        unmatched_right = [_as_pt(d) for d in right_detections if _as_pt(d) not in matched_right]
        matched_targets = kept

    if matched_targets:
        y_diffs = [m.get("y_diff_px", abs(m["left_px"][1] - m["right_px"][1])) for m in matched_targets]
        print(f"[MATCH] kept={len(matched_targets)} max_y_diff={max(y_diffs):.1f}px")
        if verbose:
            for i, m in enumerate(matched_targets, start=1):
                print(
                    f"[MATCH] #{i}: L={m['left_px']} R={m['right_px']} "
                    f"y_diff={m.get('y_diff_px', 0.0):.1f}px "
                    f"disp={m.get('disp_px', 0.0):.1f}px "
                    f"score={m.get('score', 0.0):.3f}"
                )
    else:
        print("[MATCH] kept=0")

    if verbose:
        print_match_summary(matched_targets, unmatched_left, unmatched_right)

    return matched_targets, unmatched_left, unmatched_right
