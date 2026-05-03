import math
import time

import cv2
import numpy as np

from config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    RECT_NPZ_PATH,
    FINE_ALIGN_REID_EPIPOLAR_TOL_MULT,
    FINE_ALIGN_REID_MAX_TRI_DIST_MM,
    FINE_ALIGN_REID_MAX_PD_ERROR_PX,
)
from vision.matching import match_points


_rect_maps_cache = {}


def _find_npz_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    return None


def _load_rect_maps():
    if _rect_maps_cache:
        return _rect_maps_cache

    data = np.load(str(RECT_NPZ_PATH))

    left_x_key = _find_npz_key(data, [
        "map1L", "left_map_x", "map1_left", "left_map1", "mapLx", "mapxL", "lmapx",
        "map1x", "mapx1",
    ])
    left_y_key = _find_npz_key(data, [
        "map2L", "left_map_y", "map2_left", "left_map2", "mapLy", "mapyL", "lmapy",
        "map1y", "mapy1",
    ])
    right_x_key = _find_npz_key(data, [
        "map1R", "right_map_x", "map1_right", "right_map1", "mapRx", "mapxR", "rmapx",
        "map2x", "mapx2",
    ])
    right_y_key = _find_npz_key(data, [
        "map2R", "right_map_y", "map2_right", "right_map2", "mapRy", "mapyR", "rmapy",
        "map2y", "mapy2",
    ])

    if None in (left_x_key, left_y_key, right_x_key, right_y_key):
        raise RuntimeError(
            f"Could not find rectification map keys in {RECT_NPZ_PATH}. "
            f"Available keys: {list(data.keys())}"
        )

    _rect_maps_cache["left_map_x"] = data[left_x_key]
    _rect_maps_cache["left_map_y"] = data[left_y_key]
    _rect_maps_cache["right_map_x"] = data[right_x_key]
    _rect_maps_cache["right_map_y"] = data[right_y_key]
    return _rect_maps_cache


def _rectify_pair(left_frame, right_frame):
    maps = _load_rect_maps()
    left_rect = cv2.remap(
        left_frame,
        maps["left_map_x"],
        maps["left_map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rect = cv2.remap(
        right_frame,
        maps["right_map_x"],
        maps["right_map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left_rect, right_rect


def _normalize_class_filter(class_filter):
    if class_filter is None:
        return None
    if isinstance(class_filter, (list, tuple, set)):
        return [int(v) for v in class_filter]
    return [int(class_filter)]


def _translate_to_full(stable_list, ox, oy):
    out = []
    for det in stable_list:
        px, py = det["point"]
        box = det.get("box")
        full = {
            "point": [float(px + ox), float(py + oy)],
            "cls": int(det.get("cls", -1)),
            "conf": float(det.get("conf", 0.0)),
            "views": int(det.get("views", 1)),
        }
        if box is not None:
            full["box"] = [
                float(box[0] + ox),
                float(box[1] + oy),
                float(box[2] + ox),
                float(box[3] + oy),
            ]
        out.append(full)
    return out


def _draw_overlay(full_img, crop, detections, matches, chosen, side):
    ov = full_img.copy()
    x0, y0, x1, y1 = crop
    cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 255, 255), 2)

    for i, det in enumerate(detections):
        px, py = int(round(det["point"][0])), int(round(det["point"][1]))
        cv2.circle(ov, (px, py), 5, (0, 180, 255), -1)
        cv2.putText(
            ov,
            f"D{i} c{det.get('cls', '?')} {det.get('conf', 0.0):.2f}",
            (px + 6, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 255),
            1,
        )

    key = "left_px" if side == "left" else "right_px"
    for match in matches:
        pt = match.get(key)
        if pt is None:
            continue
        cv2.circle(ov, (int(round(pt[0])), int(round(pt[1]))), 9, (255, 255, 0), 2)

    if chosen is not None and chosen.get(key) is not None:
        pt = chosen[key]
        cv2.circle(ov, (int(round(pt[0])), int(round(pt[1]))), 14, (0, 255, 0), 3)

    cv2.drawMarker(
        ov,
        (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )
    return ov


def _point_inside_crop(pt, crop, margin=8.0):
    x0, y0, x1, y1 = crop
    x, y = float(pt[0]), float(pt[1])
    return (x0 + margin) <= x < (x1 - margin) and (y0 + margin) <= y < (y1 - margin)


def _survey_geo_score(candidate_tri_xy, all_tri_xys, planned_xy, coarse_mover, tol_mm=30.0):
    planned_targets = getattr(coarse_mover, "all_planned_targets", None)
    if not planned_targets or len(planned_targets) < 2:
        return 0.5

    cx, cy = map(float, candidate_tri_xy)
    sx, sy = map(float, planned_xy)
    matched = 0
    total = 0

    for tgt in planned_targets:
        pkx, pky = tgt["target_xy_mm"]
        d_plan = math.hypot(float(pkx) - sx, float(pky) - sy)
        if d_plan < 5.0 or d_plan > 150.0:
            continue
        total += 1
        exp_dx = float(pkx) - sx
        exp_dy = float(pky) - sy
        for ox, oy in all_tri_xys:
            if abs(float(ox) - cx) < 1.0 and abs(float(oy) - cy) < 1.0:
                continue
            if math.hypot((float(ox) - cx) - exp_dx, (float(oy) - cy) - exp_dy) < tol_mm:
                matched += 1
                break

    return (matched / total) if total > 0 else 0.5


def _apply_geometry_filters(
    base_matches,
    crop,
    coarse_mover,
    planned_target,
    actual_hits,
    ref_xy,
    max_pd_error_px,
    epipolar_tol_mult,
    max_tri_dist_mm,
):
    solved_all = coarse_mover.solve_all_from_pose(base_matches, ref_x=float(ref_xy[0]), ref_y=float(ref_xy[1]))
    planned_xy = tuple(map(float, planned_target["target_xy_mm"]))
    all_tri_xys = [
        tuple(map(float, ts["target_xy_mm"]))
        for ts in solved_all
        if ts is not None and "target_xy_mm" in ts
    ]

    ep_slope = getattr(coarse_mover, "epipolar_slope", None)
    ep_tol_raw = getattr(coarse_mover, "epipolar_slope_tol", 0.1)
    ep_tol = max(float(ep_tol_raw) * float(epipolar_tol_mult), 0.15)

    rejects = {"crop": 0, "duplicate": 0, "pd": 0, "epipolar": 0, "max_tri_dist": 0}
    ranked = []
    for solved in solved_all:
        src = solved["source_target"]
        lx, ly = float(src["left_px"][0]), float(src["left_px"][1])
        rx, ry = float(src["right_px"][0]), float(src["right_px"][1])

        if not (_point_inside_crop((lx, ly), crop) and _point_inside_crop((rx, ry), crop)):
            rejects["crop"] += 1
            continue

        tri_xy = tuple(map(float, solved["target_xy_mm"]))
        if coarse_mover.is_duplicate_of_actual(tri_xy, actual_hits or [], tol_mm=15.0):
            rejects["duplicate"] += 1
            continue
        if any(
            "reid_tri_xy_mm" in hit
            and float(np.hypot(
                tri_xy[0] - float(hit["reid_tri_xy_mm"][0]),
                tri_xy[1] - float(hit["reid_tri_xy_mm"][1]),
            )) <= 10.0
            for hit in (actual_hits or [])
        ):
            rejects["duplicate"] += 1
            continue

        pd_err = math.hypot(((lx + rx) / 2.0) - (FRAME_WIDTH / 2.0), ((ly + ry) / 2.0) - (FRAME_HEIGHT / 2.0))
        if pd_err > float(max_pd_error_px):
            rejects["pd"] += 1
            continue

        if ep_slope is not None:
            dx_pair = rx - lx
            if abs(dx_pair) > 5.0:
                pair_slope = (ry - ly) / dx_pair
                if abs(pair_slope - float(ep_slope)) > ep_tol:
                    rejects["epipolar"] += 1
                    continue

        tri_dist_mm = float(np.hypot(tri_xy[0] - planned_xy[0], tri_xy[1] - planned_xy[1]))
        if tri_dist_mm > float(max_tri_dist_mm):
            rejects["max_tri_dist"] += 1
            continue

        geo_score = _survey_geo_score(tri_xy, all_tri_xys, planned_xy, coarse_mover)
        ranked.append((solved, pd_err, tri_dist_mm, geo_score))

    ranked.sort(key=lambda item: (-item[3], item[2], item[1], -float(item[0]["source_target"].get("score", 0.0))))

    candidates = []
    for solved, pd_err, tri_dist_mm, geo_score in ranked:
        src = solved["source_target"]
        entry = {
            "left_px": [float(src["left_px"][0]), float(src["left_px"][1])],
            "right_px": [float(src["right_px"][0]), float(src["right_px"][1])],
            "left_cls": int(src.get("left_cls")) if src.get("left_cls") is not None else None,
            "right_cls": int(src.get("right_cls")) if src.get("right_cls") is not None else None,
            "left_conf": float(src.get("left_conf", 0.0)),
            "right_conf": float(src.get("right_conf", 0.0)),
            "score": float(src.get("score", 0.0)),
            "y_diff_px": float(abs(src["left_px"][1] - src["right_px"][1])),
            "disp_px": float(src["left_px"][0] - src["right_px"][0]),
            "pd_err_px": float(pd_err),
            "tri_xy_mm": [float(tri_xy) for tri_xy in solved["target_xy_mm"]],
            "tri_dist_mm": float(tri_dist_mm),
            "geo_score": float(geo_score),
        }
        if "left_box" in src:
            entry["left_box"] = [float(v) for v in src["left_box"]]
        if "right_box" in src:
            entry["right_box"] = [float(v) for v in src["right_box"]]
        candidates.append(entry)

    return candidates, rejects


def run_fine_align_reid(
    cameras,
    detector,
    target=None,
    crop_w=384,
    crop_h=384,
    burst_count=5,
    min_hits=1,
    cluster_radius_px=None,
    point_mode="box_center",
    class_filter=None,
    conf_override=None,
    imgsz=None,
    use_rectified=True,
    y_gate_px=5,
    min_disp_px=10,
    max_disp_px=500,
    max_pd_error_px=FINE_ALIGN_REID_MAX_PD_ERROR_PX,
    coarse_mover=None,
    planned_target=None,
    actual_hits=None,
    gantry=None,
    ref_xy=None,
    epipolar_tol_mult=FINE_ALIGN_REID_EPIPOLAR_TOL_MULT,
    max_tri_dist_mm=FINE_ALIGN_REID_MAX_TRI_DIST_MM,
    return_debug=True,
):
    t_total = time.perf_counter()
    timing = {}

    result = {
        "ok": False,
        "target": target,
        "frame_mode": "rectified" if use_rectified else "raw",
        "left_detections": [],
        "right_detections": [],
        "matches": [],
        "chosen": None,
        "crop": {},
        "timing": timing,
        "debug_frames": None,
        "filter_mode": "basic",
        "filter_rejects": None,
        "error": None,
    }

    try:
        class_list = _normalize_class_filter(class_filter)
        crop_w = max(32, min(FRAME_WIDTH, int(crop_w)))
        crop_h = max(32, min(FRAME_HEIGHT, int(crop_h)))
        burst_count = max(1, int(burst_count))
        min_hits = max(1, int(min_hits))

        cx = FRAME_WIDTH // 2
        cy = FRAME_HEIGHT // 2
        x0 = max(0, cx - crop_w // 2)
        y0 = max(0, cy - crop_h // 2)
        x1 = min(FRAME_WIDTH, x0 + crop_w)
        y1 = min(FRAME_HEIGHT, y0 + crop_h)
        crop = (x0, y0, x1, y1)

        result["crop"] = {
            "left": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "right": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        }

        left_frames = []
        right_frames = []
        t_read = time.perf_counter()
        attempts = 0
        max_attempts = burst_count * 3
        while len(left_frames) < burst_count and attempts < max_attempts:
            fL, fR = cameras.read_pair()
            attempts += 1
            if fL is None or fR is None:
                continue
            if use_rectified:
                fL, fR = _rectify_pair(fL, fR)
            left_frames.append(fL)
            right_frames.append(fR)
        timing["read_burst_s"] = round(time.perf_counter() - t_read, 6)

        if not left_frames:
            raise RuntimeError("No stereo frames captured for Re-ID burst.")

        left_crops = [f[y0:y1, x0:x1] for f in left_frames]
        right_crops = [f[y0:y1, x0:x1] for f in right_frames]
        timing["crop_s"] = round(0.0, 6)

        old_left_conf = detector.cv_left.conf
        old_right_conf = detector.cv_right.conf
        if conf_override is not None:
            detector.cv_left.conf = float(conf_override)
            detector.cv_right.conf = float(conf_override)

        try:
            t_left = time.perf_counter()
            left_stable_crop = detector.cv_left.return_burst_stable(
                left_crops,
                min_stable_views=min_hits,
                group_radius_px=cluster_radius_px,
                classes_override=class_list,
                debug_label="[FINE REID LEFT]",
                imgsz=imgsz,
                heatmap_final=(point_mode != "box_center"),
                point_mode=point_mode,
            )
            timing["yolo_left_s"] = round(time.perf_counter() - t_left, 6)

            t_right = time.perf_counter()
            right_stable_crop = detector.cv_right.return_burst_stable(
                right_crops,
                min_stable_views=min_hits,
                group_radius_px=cluster_radius_px,
                classes_override=class_list,
                debug_label="[FINE REID RIGHT]",
                imgsz=imgsz,
                heatmap_final=(point_mode != "box_center"),
                point_mode=point_mode,
            )
            timing["yolo_right_s"] = round(time.perf_counter() - t_right, 6)
        finally:
            detector.cv_left.conf = old_left_conf
            detector.cv_right.conf = old_right_conf

        if conf_override is not None:
            min_conf = float(conf_override)
            left_stable_crop = [d for d in left_stable_crop if float(d.get("conf", 0.0)) >= min_conf]
            right_stable_crop = [d for d in right_stable_crop if float(d.get("conf", 0.0)) >= min_conf]

        left_detections = _translate_to_full(left_stable_crop, x0, y0)
        right_detections = _translate_to_full(right_stable_crop, x0, y0)

        # Re-ID detections come from rectified frames when use_rectified=True.
        # Thread explicit rectified fields so matcher uses these directly and
        # avoids applying raw->rectified remapping a second time.
        if use_rectified:
            for det in left_detections:
                det["point_rectified"] = tuple(det["point"])
                if "box" in det:
                    det["box_rectified"] = tuple(det["box"])
            for det in right_detections:
                det["point_rectified"] = tuple(det["point"])
                if "box" in det:
                    det["box_rectified"] = tuple(det["box"])

        result["left_detections"] = left_detections
        result["right_detections"] = right_detections

        t_match = time.perf_counter()
        matched_targets, _, _ = match_points(
            left_detections,
            right_detections,
            verbose=False,
            anchor_min_disp=float(min_disp_px),
            anchor_max_disp=float(max_disp_px),
            anchor_max_y_diff=float(y_gate_px),
        )
        timing["match_s"] = round(time.perf_counter() - t_match, 6)

        base_matches = []
        for m in matched_targets:
            lx, ly = float(m["left_px"][0]), float(m["left_px"][1])
            rx, ry = float(m["right_px"][0]), float(m["right_px"][1])

            left_cls = m.get("left_cls")
            right_cls = m.get("right_cls")
            if class_list is not None:
                if left_cls not in class_list or right_cls not in class_list:
                    continue

            y_diff = abs(ly - ry)
            disp = lx - rx
            if y_diff > float(y_gate_px):
                continue
            if not (float(min_disp_px) <= disp <= float(max_disp_px)):
                continue

            base_match = {
                "left_px": [lx, ly],
                "right_px": [rx, ry],
                "left_cls": int(left_cls) if left_cls is not None else None,
                "right_cls": int(right_cls) if right_cls is not None else None,
                "left_conf": float(m.get("left_conf", 0.0)),
                "right_conf": float(m.get("right_conf", 0.0)),
                "score": float(m.get("score", 0.0)),
            }
            if "left_box" in m:
                base_match["left_box"] = [float(v) for v in m["left_box"]]
            if "right_box" in m:
                base_match["right_box"] = [float(v) for v in m["right_box"]]
            base_matches.append(base_match)

            avg_x = (lx + rx) / 2.0
            avg_y = (ly + ry) / 2.0
            pd_err = math.hypot(avg_x - FRAME_WIDTH / 2.0, avg_y - FRAME_HEIGHT / 2.0)

            candidate = {
                "left_px": [lx, ly],
                "right_px": [rx, ry],
                "left_cls": int(left_cls) if left_cls is not None else None,
                "right_cls": int(right_cls) if right_cls is not None else None,
                "left_conf": float(m.get("left_conf", 0.0)),
                "right_conf": float(m.get("right_conf", 0.0)),
                "y_diff_px": float(y_diff),
                "disp_px": float(disp),
                "pd_err_px": float(pd_err),
                "score": float(m.get("score", 0.0)),
            }
            if "left_box" in m:
                candidate["left_box"] = [float(v) for v in m["left_box"]]
            if "right_box" in m:
                candidate["right_box"] = [float(v) for v in m["right_box"]]
        candidates = []
        geometry_ctx_ok = (
            coarse_mover is not None
            and planned_target is not None
            and (gantry is not None or ref_xy is not None)
        )
        if geometry_ctx_ok:
            if ref_xy is None:
                gantry.sync_estimate_to_machine()
                ref_xy = gantry.get_estimated_xy()
            candidates, rejects = _apply_geometry_filters(
                base_matches=base_matches,
                crop=crop,
                coarse_mover=coarse_mover,
                planned_target=planned_target,
                actual_hits=actual_hits,
                ref_xy=ref_xy,
                max_pd_error_px=max_pd_error_px,
                epipolar_tol_mult=epipolar_tol_mult,
                max_tri_dist_mm=max_tri_dist_mm,
            )
            result["filter_mode"] = "geometry"
            result["filter_rejects"] = rejects
        else:
            candidates = []
            for m in base_matches:
                lx, ly = float(m["left_px"][0]), float(m["left_px"][1])
                rx, ry = float(m["right_px"][0]), float(m["right_px"][1])
                avg_x = (lx + rx) / 2.0
                avg_y = (ly + ry) / 2.0
                pd_err = math.hypot(avg_x - FRAME_WIDTH / 2.0, avg_y - FRAME_HEIGHT / 2.0)
                candidate = {
                    "left_px": [lx, ly],
                    "right_px": [rx, ry],
                    "left_cls": m.get("left_cls"),
                    "right_cls": m.get("right_cls"),
                    "left_conf": float(m.get("left_conf", 0.0)),
                    "right_conf": float(m.get("right_conf", 0.0)),
                    "y_diff_px": float(abs(ly - ry)),
                    "disp_px": float(lx - rx),
                    "pd_err_px": float(pd_err),
                    "score": float(m.get("score", 0.0)),
                }
                if "left_box" in m:
                    candidate["left_box"] = [float(v) for v in m["left_box"]]
                if "right_box" in m:
                    candidate["right_box"] = [float(v) for v in m["right_box"]]
                candidates.append(candidate)
            candidates.sort(key=lambda c: (c["pd_err_px"], c["y_diff_px"], -c["score"]))

        chosen = candidates[0] if candidates else None

        result["matches"] = candidates
        result["chosen"] = chosen

        if return_debug:
            left_full = left_frames[-1].copy()
            right_full = right_frames[-1].copy()
            left_crop_img = left_crops[-1].copy()
            right_crop_img = right_crops[-1].copy()
            left_overlay = _draw_overlay(left_full, crop, left_detections, candidates, chosen, "left")
            right_overlay = _draw_overlay(right_full, crop, right_detections, candidates, chosen, "right")
            result["debug_frames"] = {
                "left_full": left_full,
                "right_full": right_full,
                "left_crop": left_crop_img,
                "right_crop": right_crop_img,
                "left_overlay": left_overlay,
                "right_overlay": right_overlay,
            }

        result["ok"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        timing["total_s"] = round(time.perf_counter() - t_total, 6)

    return result
