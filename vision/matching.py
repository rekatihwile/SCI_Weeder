from ui.terminal import print_match_summary
import numpy as np
import cv2

from config import (
    STEREO_MATCH_MIN_DISPARITY_PX,
    STEREO_MATCH_MAX_DISPARITY_PX,
    STEREO_MATCH_MAX_Y_DIFF_PX,
    STEREO_MATCH_RADIUS_PX,
    STEREO_MATCH_MIN_SCORE,
    STEREO_MATCH_MIN_BOX_IOU,
    STEREO_MATCH_IOU_WEIGHT,
    RECT_NPZ_PATH,
)


_RECT_CACHE = {}


def _find_npz_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    return None


def _load_inverse_rect_maps():
    """
    Build raw->rect lookup tables for each camera side.

    RECT_NPZ maps are rectified->raw sampling maps used by cv2.remap.
    We build sparse inverse maps for fast point conversion during matching.
    """
    if _RECT_CACHE:
        return _RECT_CACHE

    data = np.load(str(RECT_NPZ_PATH))

    left_x_key = _find_npz_key(data, ["map1L", "left_map_x", "map1_left", "left_map1", "mapLx", "mapxL", "lmapx", "map1x", "mapx1"])
    left_y_key = _find_npz_key(data, ["map2L", "left_map_y", "map2_left", "left_map2", "mapLy", "mapyL", "lmapy", "map1y", "mapy1"])
    right_x_key = _find_npz_key(data, ["map1R", "right_map_x", "map1_right", "right_map1", "mapRx", "mapxR", "rmapx", "map2x", "mapx2"])
    right_y_key = _find_npz_key(data, ["map2R", "right_map_y", "map2_right", "right_map2", "mapRy", "mapyR", "rmapy", "map2y", "mapy2"])

    if None in (left_x_key, left_y_key, right_x_key, right_y_key):
        raise RuntimeError(
            f"Could not find rectification map keys in {RECT_NPZ_PATH}. "
            f"Available keys: {list(data.keys())}"
        )

    def _as_float_xy_maps(map1, map2):
        m1 = np.asarray(map1)
        m2 = np.asarray(map2)

        # OpenCV can store remap data either as split float maps (HxW + HxW)
        # or as fixed-point map1(HxWx2)+map2(HxW). Convert to split float maps.
        if m1.ndim == 3 and m1.shape[2] == 2:
            try:
                mx, my = cv2.convertMaps(m1, m2, cv2.CV_32FC1)
                return np.asarray(mx, dtype=np.float32), np.asarray(my, dtype=np.float32)
            except cv2.error:
                return np.asarray(m1[..., 0], dtype=np.float32), np.asarray(m1[..., 1], dtype=np.float32)

        if m1.ndim == 2 and m2.ndim == 2:
            return np.asarray(m1, dtype=np.float32), np.asarray(m2, dtype=np.float32)

        raise RuntimeError(
            f"Unsupported rectification map shapes: map1={m1.shape}, map2={m2.shape}"
        )

    left_map_x, left_map_y = _as_float_xy_maps(data[left_x_key], data[left_y_key])
    right_map_x, right_map_y = _as_float_xy_maps(data[right_x_key], data[right_y_key])

    def _invert_map(map_x, map_y):
        h, w = map_x.shape[:2]
        inv_x = np.full((h, w), np.nan, dtype=np.float32)
        inv_y = np.full((h, w), np.nan, dtype=np.float32)

        src_x = np.rint(map_x).astype(np.int32)
        src_y = np.rint(map_y).astype(np.int32)

        rect_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        rect_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

        valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
        inv_x[src_y[valid], src_x[valid]] = rect_x[valid]
        inv_y[src_y[valid], src_x[valid]] = rect_y[valid]
        return inv_x, inv_y

    _RECT_CACHE["left_map_x"] = left_map_x
    _RECT_CACHE["left_map_y"] = left_map_y
    _RECT_CACHE["right_map_x"] = right_map_x
    _RECT_CACHE["right_map_y"] = right_map_y
    _RECT_CACHE["left_inv_x"], _RECT_CACHE["left_inv_y"] = _invert_map(left_map_x, left_map_y)
    _RECT_CACHE["right_inv_x"], _RECT_CACHE["right_inv_y"] = _invert_map(right_map_x, right_map_y)
    return _RECT_CACHE


def _raw_point_to_rectified(pt, side):
    """Map a raw pixel coordinate to rectified pixel coordinate for left/right side."""
    maps = _load_inverse_rect_maps()

    if side == "left":
        inv_x, inv_y = maps["left_inv_x"], maps["left_inv_y"]
        map_x, map_y = maps["left_map_x"], maps["left_map_y"]
    else:
        inv_x, inv_y = maps["right_inv_x"], maps["right_inv_y"]
        map_x, map_y = maps["right_map_x"], maps["right_map_y"]

    h, w = inv_x.shape[:2]
    x = int(np.clip(round(float(pt[0])), 0, w - 1))
    y = int(np.clip(round(float(pt[1])), 0, h - 1))

    rx = inv_x[y, x]
    ry = inv_y[y, x]
    if not np.isnan(rx) and not np.isnan(ry):
        return (float(rx), float(ry))

    # Fallback for sparse inverse holes: nearest source sample in rect map domain.
    dist2 = (map_x - float(pt[0])) ** 2 + (map_y - float(pt[1])) ** 2
    min_idx = int(np.argmin(dist2))
    rr, rc = np.unravel_index(min_idx, dist2.shape)
    return (float(rc), float(rr))


def _raw_box_to_rectified(box, side):
    """Approximate rectified AABB by mapping raw box corners to rectified pixels."""
    x1, y1, x2, y2 = box
    corners = [
        _raw_point_to_rectified((x1, y1), side),
        _raw_point_to_rectified((x2, y1), side),
        _raw_point_to_rectified((x1, y2), side),
        _raw_point_to_rectified((x2, y2), side),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


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
    def _normalise(dets, side):
        if not dets:
            return [], [], [], []
        if isinstance(dets[0], dict):
            raw_pts = [d["point"] for d in dets]
            raw_boxes = [d["box"] for d in dets]

            # Prefer explicit rectified fields if present; otherwise map raw->rectified.
            has_rect_pts = all(("point_rectified" in d or "rectified_point" in d or "point_rect" in d) for d in dets)
            if has_rect_pts:
                pts = [
                    d.get("point_rectified", d.get("rectified_point", d.get("point_rect")))
                    for d in dets
                ]
            else:
                pts = [_raw_point_to_rectified(p, side) for p in raw_pts]

            has_rect_boxes = all(("box_rectified" in d or "rectified_box" in d or "box_rect" in d) for d in dets)
            if has_rect_boxes:
                boxes = [
                    d.get("box_rectified", d.get("rectified_box", d.get("box_rect")))
                    for d in dets
                ]
            else:
                boxes = [_raw_box_to_rectified(b, side) for b in raw_boxes]
        else:
            raw_pts = [tuple(map(int, p)) for p in dets]
            pts = [_raw_point_to_rectified(p, side) for p in raw_pts]
            raw_boxes = None
            boxes = None
        return pts, boxes, raw_pts, raw_boxes

    left_pts, left_boxes, left_raw_pts, left_raw_boxes = _normalise(left_detections, "left")
    right_pts, right_boxes, right_raw_pts, right_raw_boxes = _normalise(right_detections, "right")

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
        lp_rect = tuple(map(float, pts_l[l_idx]))
        rp_rect = tuple(map(float, pts_r[r_idx]))

        # Preserve existing downstream contract: planner/triangulation consume raw pixels.
        lp = tuple(map(int, left_raw_pts[l_idx]))
        rp = tuple(map(int, right_raw_pts[r_idx]))

        entry = {
            "left_px":  lp,
            "right_px": rp,
            "score":    float(best_meta[l_idx]["score"]),
            # Scoring diagnostics are in rectified coordinates only.
            "left_px_rect":  (float(lp_rect[0]), float(lp_rect[1])),
            "right_px_rect": (float(rp_rect[0]), float(rp_rect[1])),
            "y_diff_px": float(abs(rp_rect[1] - lp_rect[1])),
            "disp_px":   float(abs(rp_rect[0] - lp_rect[0])),
            "dx_px":     float(rp_rect[0] - lp_rect[0]),
            "dy_px":     float(rp_rect[1] - lp_rect[1]),
        }
        if have_boxes:
            entry["left_box"]  = left_raw_boxes[l_idx] if left_raw_boxes is not None else left_boxes[l_idx]
            entry["right_box"] = right_raw_boxes[r_idx] if right_raw_boxes is not None else right_boxes[r_idx]
            entry["left_box_rect"] = left_boxes[l_idx]
            entry["right_box_rect"] = right_boxes[r_idx]
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

    unmatched_left = [tuple(map(int, left_raw_pts[i])) for i in range(len(pts_l)) if i not in matched_left_idx]
    unmatched_right = [tuple(map(int, right_raw_pts[i])) for i in range(len(pts_r)) if i not in matched_right_idx]

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
