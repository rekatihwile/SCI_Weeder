import math
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ui.terminal import print_live_fine_align, end_live_fine_align
from control.fine_align_reid import run_fine_align_reid
from config.survey_params import resolve_burst_count, resolve_point_mode
from config import (
    DETECTOR_MODE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    HAS_DISPLAY,
    FINE_ALIGN_CROP_SCALE,
    FINE_ALIGN_LK_WIN_SIZE,
    FINE_ALIGN_LK_MAX_LEVEL,
    FINE_ALIGN_KP_X,
    FINE_ALIGN_KD_X,
    FINE_ALIGN_KP_Y,
    FINE_ALIGN_KD_Y,
    FINE_ALIGN_STEP_MM,
    FINE_ALIGN_DEADZONE_PX,
    FINE_ALIGN_MAX_JOG_MM,
    FINE_ALIGN_FEED,
    FINE_ALIGN_BURST_COUNT,
    FINE_ALIGN_REID_BURST_COUNT,
    FINE_ALIGN_MIN_HITS,
    FINE_ALIGN_CLUSTER_RADIUS_PX,
    FINE_ALIGN_REID_CROP_HALF_PX,
    FINE_ALIGN_REID_YOLO_IMGSZ,
    FINE_ALIGN_REID_MAX_PD_ERROR_PX,
    FINE_ALIGN_REID_MAX_Y_DIFF_PX,
    FINE_ALIGN_REID_MIN_DISPARITY_PX,
    FINE_ALIGN_REID_MAX_DISPARITY_PX,
    FINE_ALIGN_MAX_TIME_SEC,
    FINE_ALIGN_SETTLE_FRAMES,
    FINE_ALIGN_SNAP_SETTLE_FRAMES,
    FINE_ALIGN_ENABLE_SNAP,
    FINE_ALIGN_SNAP_MODE,
    FINE_ALIGN_SNAP_ON_DEADZONE,
    FINE_ALIGN_SNAP_CROP_HALF_PX,
    FINE_ALIGN_REID_EPIPOLAR_TOL_MULT,
    FINE_ALIGN_REID_MAX_TRI_DIST_MM,
    OVERRIDE_BURST_NUMBER,
    OVERRIDE_BURST_COUNT,
    OVERRIDE_POINT_MODE,
    OVERRIDE_POINT_MODE_VALUE,
    FINE_ALIGN_REID_POINT_MODE,
    FINAL_SNAP_POINT_MODE,
    FINAL_SNAP_BURST_COUNT,
    RECORD_LIVE_OVERLAYS,
    WORKSPACE_X_MIN,
    WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Y_MAX,
)
from vision.matching import match_points

FULL_W = FRAME_WIDTH
FULL_H = FRAME_HEIGHT
FINE_W = int(FRAME_WIDTH * FINE_ALIGN_CROP_SCALE)
FINE_H = int(FRAME_HEIGHT * FINE_ALIGN_CROP_SCALE)
CROP_X0 = (FULL_W - FINE_W) // 2
CROP_Y0 = (FULL_H - FINE_H) // 2
CROP_X1 = CROP_X0 + FINE_W
CROP_Y1 = CROP_Y0 + FINE_H
TARGET_X = FINE_W / 2.0
TARGET_Y = FINE_H / 2.0

LK_PARAMS = dict(
    winSize=(FINE_ALIGN_LK_WIN_SIZE, FINE_ALIGN_LK_WIN_SIZE),
    maxLevel=FINE_ALIGN_LK_MAX_LEVEL,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

Kp_x = FINE_ALIGN_KP_X
Kd_x = FINE_ALIGN_KD_X
Kp_y = FINE_ALIGN_KP_Y
Kd_y = FINE_ALIGN_KD_Y

STEP_MM = FINE_ALIGN_STEP_MM
DEADZONE = FINE_ALIGN_DEADZONE_PX
MAX_JOG = FINE_ALIGN_MAX_JOG_MM
FINE_FEED = FINE_ALIGN_FEED
BURST_COUNT = FINE_ALIGN_REID_BURST_COUNT
MIN_HITS = FINE_ALIGN_MIN_HITS
CLUSTER_RADIUS_PX = FINE_ALIGN_CLUSTER_RADIUS_PX

_REID_HALF = FINE_ALIGN_REID_CROP_HALF_PX
_REID_YOLO_IMGSZ = FINE_ALIGN_REID_YOLO_IMGSZ
_REID_MAX_PD_ERR = FINE_ALIGN_REID_MAX_PD_ERROR_PX
_REID_MAX_Y_DIFF = FINE_ALIGN_REID_MAX_Y_DIFF_PX
_REID_MIN_DISP = FINE_ALIGN_REID_MIN_DISPARITY_PX
_REID_MAX_DISP = FINE_ALIGN_REID_MAX_DISPARITY_PX
_REID_EPIPOLAR_TOL_MULT = FINE_ALIGN_REID_EPIPOLAR_TOL_MULT
_REID_MAX_TRI_DIST_MM = FINE_ALIGN_REID_MAX_TRI_DIST_MM

_WS_MARGIN = 5.0
_FINE_WINDOW = "Fine Align"


def _fine_debug(msg):
    print(f"[FINE DEBUG] {msg}", flush=True)


def _default_reid_settings():
    return {
        "burst_count": resolve_burst_count(BURST_COUNT),
        "min_hits": MIN_HITS,
        "cluster_radius_px": CLUSTER_RADIUS_PX,
        "crop_half_px": _REID_HALF,
        "yolo_imgsz": _REID_YOLO_IMGSZ,
        "max_y_diff_px": _REID_MAX_Y_DIFF,
        "min_disparity_px": _REID_MIN_DISP,
        "max_disparity_px": _REID_MAX_DISP,
        "max_pd_error_px": _REID_MAX_PD_ERR,
        "point_mode": resolve_point_mode(FINE_ALIGN_REID_POINT_MODE),
    }


def _resolve_reid_settings(overrides=None):
    settings = _default_reid_settings()
    if overrides:
        settings.update({k: v for k, v in overrides.items() if v is not None})
    settings["burst_count"] = max(1, int(settings["burst_count"]))
    settings["min_hits"] = max(1, int(settings["min_hits"]))
    settings["crop_half_px"] = max(1, int(settings["crop_half_px"]))
    if isinstance(settings["yolo_imgsz"], str):
        raw_imgsz = settings["yolo_imgsz"].strip().lower()
        settings["yolo_imgsz"] = None if raw_imgsz in ("", "none", "auto", "crop") else int(raw_imgsz)
    if settings["yolo_imgsz"] is not None:
        settings["yolo_imgsz"] = max(32, int(settings["yolo_imgsz"]))
    settings["cluster_radius_px"] = float(settings["cluster_radius_px"])
    settings["max_y_diff_px"] = float(settings["max_y_diff_px"])
    settings["min_disparity_px"] = float(settings["min_disparity_px"])
    settings["max_disparity_px"] = float(settings["max_disparity_px"])
    settings["max_pd_error_px"] = float(settings["max_pd_error_px"])
    settings["point_mode"] = resolve_point_mode(settings.get("point_mode", FINE_ALIGN_REID_POINT_MODE))
    return settings


def get_default_reid_settings():
    return _resolve_reid_settings()


def close_fine_align_window():
    if HAS_DISPLAY:
        try:
            cv2.destroyWindow(_FINE_WINDOW)
        except Exception:
            pass


def _crop_frame(frame):
    return frame[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1].copy()


def _full_to_crop_point(pt):
    return (float(pt[0]) - CROP_X0, float(pt[1]) - CROP_Y0)


def _point_inside_crop(pt, margin=8.0):
    x, y = _full_to_crop_point(pt)
    return margin <= x < (FINE_W - margin) and margin <= y < (FINE_H - margin)


def _compute_errors(left_pt, right_pt):
    xl, yl = left_pt
    xr, yr = right_pt
    err_x = (xl + xr) - (2.0 * TARGET_X)
    err_y = TARGET_Y - ((yl + yr) / 2.0)
    return err_x, err_y


def _pair_pd_error(left_pt, right_pt):
    xl, yl = left_pt
    xr, yr = right_pt
    avg_x = (float(xl) + float(xr)) / 2.0
    avg_y = (float(yl) + float(yr)) / 2.0
    return float(np.hypot(avg_x - FULL_W / 2.0, avg_y - FULL_H / 2.0))


def _clamp_jog(dx, dy, est_x, est_y):
    new_x = np.clip(est_x + dx, WORKSPACE_X_MIN + _WS_MARGIN, WORKSPACE_X_MAX - _WS_MARGIN)
    new_y = np.clip(est_y + dy, WORKSPACE_Y_MIN + _WS_MARGIN, WORKSPACE_Y_MAX - _WS_MARGIN)
    return float(new_x - est_x), float(new_y - est_y)


def _burst_match_ai_points(cameras, detector, expected_cls=None, reid_settings=None, return_debug=False):
    t_total = time.perf_counter()
    settings = _resolve_reid_settings(reid_settings)
    classes_arg = [expected_cls] if expected_cls is not None else None
    _burst_match_ai_points.last_timing = {
        "reid_point_mode": settings["point_mode"],
        "reid_burst_count": int(settings["burst_count"]),
    }

    _fine_debug(
        f"RE-ID burst start: frames={settings['burst_count']} min_hits={settings['min_hits']} "
        f"radius={settings['cluster_radius_px']:.1f}px point_mode={settings['point_mode']} "
        f"expected_cls={expected_cls}"
    )

    left_frames, right_frames = [], []
    attempts = 0
    read_time_total = 0.0
    while len(left_frames) < settings["burst_count"] and attempts < settings["burst_count"] * 3:
        t_read = time.perf_counter()
        fL, fR = cameras.read_pair()
        attempts += 1
        read_dt = time.perf_counter() - t_read
        read_time_total += read_dt
        if fL is None or fR is None:
            _fine_debug(f"RE-ID capture attempt {attempts}: missing after {read_dt:.2f}s")
            time.sleep(0.05)
            continue
        left_frames.append(fL)
        right_frames.append(fR)
        _fine_debug(f"RE-ID captured frame {len(left_frames)}/{settings['burst_count']} in {read_dt:.2f}s")

    if not left_frames:
        _fine_debug(f"RE-ID no frames captured; total={time.perf_counter() - t_total:.2f}s")
        _burst_match_ai_points.last_timing.update({
            "reid_camera_read_time_s": round(read_time_total, 6),
            "reid_total_time_s": round(time.perf_counter() - t_total, 6),
        })
        return ([], None) if return_debug else []

    cx, cy = FULL_W // 2, FULL_H // 2
    crop_half = settings["crop_half_px"]
    # Re-ID uses the same symmetric crop in both cameras.
    lx0 = max(0, cx - crop_half);       lx1 = min(FULL_W, cx + crop_half)
    ly0 = max(0, cy - crop_half);       ly1 = min(FULL_H, cy + crop_half)
    rx0, rx1 = lx0, lx1
    ry0, ry1 = ly0, ly1
    _fine_debug(
        f"RE-ID crop: L x={lx0}:{lx1} y={ly0}:{ly1}  "
        f"R x={rx0}:{rx1} y={ry0}:{ry1}  half={crop_half}px"
    )

    # Store per-camera crop offsets so heatmap refinement can map back to full-frame.
    if detector is not None:
        detector.cv_left._last_reid_rx0  = lx0
        detector.cv_left._last_reid_ry0  = ly0
        detector.cv_right._last_reid_rx0 = rx0
        detector.cv_right._last_reid_ry0 = ry0

    cropped_left  = [f[ly0:ly1, lx0:lx1] for f in left_frames]
    cropped_right = [f[ry0:ry1, rx0:rx1] for f in right_frames]
    yolo_imgsz = settings["yolo_imgsz"]
    lcrop_h, lcrop_w = cropped_left[-1].shape[:2]
    rcrop_h, rcrop_w = cropped_right[-1].shape[:2]
    _fine_debug(f"RE-ID crop frames: L={lcrop_w}x{lcrop_h} R={rcrop_w}x{rcrop_h} YOLO imgsz={yolo_imgsz}")

    def _stable_side(core, frames, label):
        t_side = time.perf_counter()
        stable = core.return_burst_stable(
            frames,
            min_stable_views=settings["min_hits"],
            group_radius_px=settings["cluster_radius_px"],
            classes_override=classes_arg,
            debug_label=label,
            imgsz=yolo_imgsz,
            heatmap_final=(settings["point_mode"] != "box_center"),
            point_mode=settings["point_mode"],
        )
        timing = getattr(core, "last_burst_timing", {})
        return stable, time.perf_counter() - t_side, timing

    t_detect = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="reid-burst") as pool:
        left_future  = pool.submit(_stable_side, detector.cv_left,  cropped_left,  "[FINE DEBUG] RE-ID LEFT")
        right_future = pool.submit(_stable_side, detector.cv_right, cropped_right, "[FINE DEBUG] RE-ID RIGHT")
        stable_left_crop, left_dt, left_timing   = left_future.result()
        stable_right_crop, right_dt, right_timing = right_future.result()
    detect_dt = time.perf_counter() - t_detect
    _fine_debug(
        f"RE-ID stable detection done: left={left_dt:.2f}s right={right_dt:.2f}s "
        f"total={detect_dt:.2f}s"
    )
    _burst_match_ai_points.last_timing.update({
        "reid_camera_read_time_s": round(read_time_total, 6),
        "reid_yolo_time_s": round(max(
            float(left_timing.get("yolo_time_s", 0.0)),
            float(right_timing.get("yolo_time_s", 0.0)),
        ), 6),
        "reid_grouping_time_s": round(max(
            float(left_timing.get("grouping_time_s", 0.0)),
            float(right_timing.get("grouping_time_s", 0.0)),
        ), 6),
        "reid_qpoint_time_s": round(max(
            float(left_timing.get("qpoint_time_s", 0.0)),
            float(right_timing.get("qpoint_time_s", 0.0)),
        ), 6),
        "reid_detection_wall_time_s": round(detect_dt, 6),
    })

    def _to_full_offset(stable_list, ox, oy):
        """Translate crop-space detections to full-frame coordinates."""
        out = []
        for s in stable_list:
            out.append({
                **s,
                "point": (int(round(s["point"][0] + ox)), int(round(s["point"][1] + oy))),
                "box":   (s["box"][0] + ox, s["box"][1] + oy,
                          s["box"][2] + ox, s["box"][3] + oy),
            })
        return out

    stable_left  = _to_full_offset(stable_left_crop,  lx0, ly0)
    stable_right = _to_full_offset(stable_right_crop, rx0, ry0)

    _fine_debug(f"RE-ID stable detections: left={len(stable_left)} right={len(stable_right)}")
    for i, det in enumerate(stable_left, start=1):
        _fine_debug(f"RE-ID stable L{i}: pt={det.get('point')} cls={det.get('cls')} conf={det.get('conf')}")
    for i, det in enumerate(stable_right, start=1):
        _fine_debug(f"RE-ID stable R{i}: pt={det.get('point')} cls={det.get('cls')} conf={det.get('conf')}")

    if cameras._recorder is not None and RECORD_LIVE_OVERLAYS:
        ann_L = detector.cv_left.draw_stable_detections(left_frames[-1], stable_left)
        ann_R = detector.cv_right.draw_stable_detections(right_frames[-1], stable_right)
        cameras._recorder.write_overlay(ann_L, ann_R)

    if not stable_left or not stable_right:
        _fine_debug(
            f"RE-ID no stable stereo candidates: left={len(stable_left)} "
            f"right={len(stable_right)} total={time.perf_counter() - t_total:.2f}s"
        )
        if return_debug:
            return [], {
                "settings": settings,
                "crop": (rx0, ry0, rx1, ry1),
                "left_frames": left_frames,
                "right_frames": right_frames,
                "stable_left": stable_left,
                "stable_right": stable_right,
                "timing": dict(_burst_match_ai_points.last_timing),
            }
        return []

    t_match = time.perf_counter()
    matched_targets, _, _ = match_points(
        stable_left,
        stable_right,
        verbose=False,
        anchor_min_disp=settings["min_disparity_px"],
        anchor_max_disp=settings["max_disparity_px"],
        anchor_max_y_diff=settings["max_y_diff_px"],
    )
    match_dt = time.perf_counter() - t_match

    filtered = []
    reject_y = 0
    reject_disp = 0
    reject_class = 0
    for i, t in enumerate(matched_targets, start=1):
        xl, yl = t["left_px"]
        xr, yr = t["right_px"]
        y_diff = abs(yl - yr)
        disp = abs(xl - xr)
        pd_err = _pair_pd_error((xl, yl), (xr, yr))

        reject_reason = None
        if y_diff > settings["max_y_diff_px"]:
            reject_reason = f"y_diff>{settings['max_y_diff_px']:.1f}"
            reject_y += 1
        elif disp < settings["min_disparity_px"] or disp > settings["max_disparity_px"]:
            reject_reason = f"disp not in {settings['min_disparity_px']:.1f}-{settings['max_disparity_px']:.1f}"
            reject_disp += 1
        elif expected_cls is not None and (
            t.get("left_cls") != expected_cls or t.get("right_cls") != expected_cls
        ):
            reject_reason = f"cls!=expected({expected_cls})"
            reject_class += 1
        else:
            filtered.append(t)

        status = "KEEP" if reject_reason is None else f"REJECT {reject_reason}"
        _fine_debug(
            f"RE-ID raw match {i} {status}: L={t['left_px']} R={t['right_px']} "
            f"y_diff={y_diff:.1f}px disp={disp:.1f}px pd={pd_err:.1f}px "
            f"cls=({t.get('left_cls')},{t.get('right_cls')}) score={t.get('score', 0.0):.3f}"
        )

    _fine_debug(
        f"RE-ID matching.py took {match_dt:.2f}s: raw={len(matched_targets)} kept={len(filtered)} "
        f"reject_y={reject_y} reject_disp={reject_disp} reject_class={reject_class} "
        f"total={time.perf_counter() - t_total:.2f}s"
    )
    _burst_match_ai_points.last_timing = {
        "reid_camera_read_time_s": round(read_time_total, 6),
        "reid_yolo_time_s": round(max(
            float(left_timing.get("yolo_time_s", 0.0)),
            float(right_timing.get("yolo_time_s", 0.0)),
        ), 6),
        "reid_grouping_time_s": round(max(
            float(left_timing.get("grouping_time_s", 0.0)),
            float(right_timing.get("grouping_time_s", 0.0)),
        ), 6),
        "reid_qpoint_time_s": round(max(
            float(left_timing.get("qpoint_time_s", 0.0)),
            float(right_timing.get("qpoint_time_s", 0.0)),
        ), 6),
        "reid_detection_wall_time_s": round(detect_dt, 6),
        "reid_matching_time_s": round(match_dt, 6),
        "reid_total_time_s": round(time.perf_counter() - t_total, 6),
        "reid_point_mode": settings["point_mode"],
        "reid_burst_count": int(settings["burst_count"]),
    }
    if return_debug:
        return filtered, {
            "settings": settings,
            "crop": (rx0, ry0, rx1, ry1),
            "left_frames": left_frames,
            "right_frames": right_frames,
            "stable_left": stable_left,
            "stable_right": stable_right,
            "matched_targets": filtered,
            "timing": dict(_burst_match_ai_points.last_timing),
        }
    return filtered


def _pick_manual_initial_target(cameras, detector):
    left_pt, right_pt = detector.refine_live(cameras)
    if left_pt is None or right_pt is None:
        return None, None, None, None, None, None

    frameL = getattr(detector, "last_displayed_left", None)
    frameR = getattr(detector, "last_displayed_right", None)
    if frameL is None or frameR is None:
        frameL, frameR = cameras.read_pair()
    if frameL is None or frameR is None:
        return None, None, None, None, None, None

    return left_pt, right_pt, frameL.copy(), frameR.copy(), None, None


def _survey_geo_score(candidate_tri_xy, all_tri_xys, current_planned_xy, coarse_mover, tol_mm=30.0):
    """Score candidate as the current target using survey-constellation geometry.

    Computes the fraction of nearby survey targets whose expected relative offset from
    the current target is observed in the re-ID detections.  A candidate that sees its
    survey neighbours in the right relative positions scores high; a candidate that is
    actually a different plant (wrong neighbourhood) scores low.

    Uses relative positions only so systematic calibration error (same for all candidates
    from the same gantry position) cancels out.
    """
    planned_targets = getattr(coarse_mover, "all_planned_targets", None)
    if not planned_targets or len(planned_targets) < 2:
        return 0.5

    cx, cy = candidate_tri_xy
    sx, sy = current_planned_xy
    matched = 0
    total = 0

    for tgt in planned_targets:
        pkx, pky = tgt["target_xy_mm"]
        d_plan = math.hypot(pkx - sx, pky - sy)
        if d_plan < 5.0 or d_plan > 150.0:
            continue
        total += 1
        exp_dx, exp_dy = pkx - sx, pky - sy
        for (ox, oy) in all_tri_xys:
            if abs(ox - cx) < 1.0 and abs(oy - cy) < 1.0:
                continue
            if math.hypot((ox - cx) - exp_dx, (oy - cy) - exp_dy) < tol_mm:
                matched += 1
                break

    return matched / total if total > 0 else 0.5


def _rank_reid_matches(gantry, coarse_mover, planned_target, actual_hits, matched_targets, reid_settings=None):
    settings = _resolve_reid_settings(reid_settings)
    gantry.sync_estimate_to_machine()
    current_x, current_y = gantry.get_estimated_xy()
    solved_all = coarse_mover.solve_all_from_pose(matched_targets, ref_x=current_x, ref_y=current_y)
    target_xy_mm = planned_target["target_xy_mm"]

    # All stereo-solved tri_xys for geometric neighbourhood scoring (includes filtered-out pairs).
    all_tri_xys = [
        tuple(map(float, ts["target_xy_mm"]))
        for ts in solved_all
        if ts is not None and "target_xy_mm" in ts
    ]
    current_planned_xy = tuple(map(float, target_xy_mm))

    # Epipolar tolerance: survey std is tight across many pairs; individual re-ID pairs
    # deviate more due to depth variation and gantry-position effects.
    # Apply multiplier + floor so very tight survey fits don't create too-narrow windows.
    ep_slope = getattr(coarse_mover, "epipolar_slope", None)
    ep_tol_raw = getattr(coarse_mover, "epipolar_slope_tol", 0.1)
    ep_tol = max(ep_tol_raw * _REID_EPIPOLAR_TOL_MULT, 0.15)

    candidates = []
    rejects = {"crop": 0, "duplicate": 0, "pd": 0, "epipolar": 0, "max_tri_dist": 0}
    for t_solved in solved_all:
        src = t_solved["source_target"]
        xl, yl = src["left_px"]
        xr, yr = src["right_px"]

        if not (_point_inside_crop((xl, yl)) and _point_inside_crop((xr, yr))):
            rejects["crop"] += 1
            continue
        if coarse_mover.is_duplicate_of_actual(t_solved["target_xy_mm"], actual_hits, tol_mm=15.0):
            rejects["duplicate"] += 1
            continue
        # Reject if this candidate's tri_xy is within 10mm of any previously locked plant's
        # reid_tri_xy — tri_xy comparisons share the same systematic calibration error so
        # relative distances are reliable even when absolute positions are off by ~30mm.
        if any(
            float(np.hypot(
                t_solved["target_xy_mm"][0] - hit["reid_tri_xy_mm"][0],
                t_solved["target_xy_mm"][1] - hit["reid_tri_xy_mm"][1],
            )) <= 10.0
            for hit in actual_hits
            if "reid_tri_xy_mm" in hit
        ):
            rejects["duplicate"] += 1
            _fine_debug(
                f"RE-ID candidate rejected: tri_xy {t_solved['target_xy_mm']} "
                f"within 10mm of a previously locked plant's reid_tri_xy"
            )
            continue

        pd_err = _pair_pd_error((xl, yl), (xr, yr))
        if pd_err > settings["max_pd_error_px"]:
            rejects["pd"] += 1
            continue

        # Epipolar slope check with 2× survey tolerance.
        if ep_slope is not None:
            dx_pair = float(xr) - float(xl)
            if abs(dx_pair) > 5.0:
                pair_slope = (float(yr) - float(yl)) / dx_pair
                if abs(pair_slope - ep_slope) > ep_tol:
                    rejects["epipolar"] += 1
                    _fine_debug(
                        f"RE-ID epipolar reject: slope={pair_slope:.3f} expected "
                        f"{ep_slope:.3f}±{ep_tol:.3f} (2×{ep_tol_raw:.3f}) "
                        f"L={src['left_px']} R={src['right_px']}"
                    )
                    continue

        tri_dist_mm = float(np.hypot(
            t_solved["target_xy_mm"][0] - target_xy_mm[0],
            t_solved["target_xy_mm"][1] - target_xy_mm[1],
        ))
        # Hard cutoff: good re-ID picks are <10mm from planned; block distant fallbacks
        # to prevent cascade duplicate-rejection failures on subsequent targets.
        if tri_dist_mm > _REID_MAX_TRI_DIST_MM:
            rejects["max_tri_dist"] += 1
            _fine_debug(
                f"RE-ID candidate rejected: tri_dist={tri_dist_mm:.1f}mm > {_REID_MAX_TRI_DIST_MM:.0f}mm limit "
                f"L={src['left_px']} R={src['right_px']}"
            )
            continue

        geo_score = _survey_geo_score(
            tuple(map(float, t_solved["target_xy_mm"])),
            all_tri_xys,
            current_planned_xy,
            coarse_mover,
        )
        candidates.append((t_solved, pd_err, tri_dist_mm, geo_score))
        _fine_debug(
            f"RE-ID candidate: tri_dist={tri_dist_mm:.1f}mm pd_err={pd_err:.1f}px "
            f"geo={geo_score:.2f} L={src['left_px']} R={src['right_px']}"
        )

    # Primary sort: geo_score descending (survey-geometry match); tri_dist ascending as tiebreaker.
    return sorted(candidates, key=lambda x: (-x[3], x[2])), rejects


def _print_reid_ranked(candidates, header="[RE-ID RANKED] candidates after stereo solve + pair PD filter:"):
    print(header)
    for rank, (t_solved, pd_err, tri_dist_mm, geo_score) in enumerate(candidates, start=1):
        src = t_solved["source_target"]
        xy = t_solved["target_xy_mm"]
        print(
            f"  #{rank}: tri_dist={tri_dist_mm:.1f}mm pd_err={pd_err:.1f}px "
            f"geo={geo_score:.2f} tri_xy=({xy[0]:.1f},{xy[1]:.1f}) "
            f"L={src['left_px']} R={src['right_px']}"
        )


def _show_reid_debug_window(debug_data, candidates, target_idx=None, total_targets=None):
    if not HAS_DISPLAY or not debug_data:
        return
    left_frames = debug_data.get("left_frames") or []
    right_frames = debug_data.get("right_frames") or []
    if not left_frames or not right_frames:
        return

    dispL = left_frames[-1].copy()
    dispR = right_frames[-1].copy()
    rx0, ry0, rx1, ry1 = debug_data["crop"]

    cv2.rectangle(dispL, (rx0, ry0), (rx1, ry1), (255, 0, 0), 2)
    cv2.rectangle(dispR, (rx0, ry0), (rx1, ry1), (255, 0, 0), 2)

    for i, det in enumerate(debug_data.get("stable_left", []), start=1):
        x, y = det["point"]
        cv2.circle(dispL, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(dispL, f"L{i}", (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    for i, det in enumerate(debug_data.get("stable_right", []), start=1):
        x, y = det["point"]
        cv2.circle(dispR, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(dispR, f"R{i}", (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    combined = np.hstack([dispL, dispR])
    split_x = dispL.shape[1]
    for rank, (t_solved, pd_err, tri_dist_mm, geo_score) in enumerate(candidates, start=1):
        src = t_solved["source_target"]
        xl, yl = src["left_px"]
        xr, yr = src["right_px"]
        color = (0, 255, 0) if rank == 1 else (0, 255, 255)
        pt_l = (int(round(xl)), int(round(yl)))
        pt_r = (int(round(xr + split_x)), int(round(yr)))
        cv2.line(combined, pt_l, pt_r, color, 2, cv2.LINE_AA)
        label = f"#{rank} tri={tri_dist_mm:.1f} geo={geo_score:.2f} pd={pd_err:.1f}"
        mid = ((pt_l[0] + pt_r[0]) // 2, max(20, min(pt_l[1], pt_r[1]) - 10))
        cv2.putText(combined, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(combined, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    tgt = f"target {target_idx}/{total_targets}" if target_idx is not None else "target ?"
    cv2.putText(combined, f"RE-ID debug: {tgt} | green = chosen | blue boxes = YOLO crop",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    window = "Re-ID Debug"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(combined.shape[1], 1600), min(combined.shape[0], 900))
    cv2.moveWindow(window, 120, 120)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.imshow(window, combined)
    cv2.waitKey(1)


def debug_reid_target(
    gantry,
    cameras,
    detector,
    coarse_mover,
    planned_target,
    actual_hits,
    reid_settings=None,
    show_debug=HAS_DISPLAY,
    target_idx=None,
    total_targets=None,
):
    del show_debug, target_idx, total_targets
    settings = _resolve_reid_settings(reid_settings)
    expected_cls = planned_target.get("source_target", {}).get("left_cls")
    reid_result = run_fine_align_reid(
        cameras=cameras,
        detector=detector,
        target=planned_target,
        crop_w=int(settings["crop_half_px"]) * 2,
        crop_h=int(settings["crop_half_px"]) * 2,
        burst_count=int(settings["burst_count"]),
        min_hits=int(settings["min_hits"]),
        cluster_radius_px=float(settings["cluster_radius_px"]),
        point_mode=settings["point_mode"],
        class_filter=expected_cls,
        conf_override=None,
        imgsz=settings["yolo_imgsz"],
        use_rectified=True,
        y_gate_px=float(settings["max_y_diff_px"]),
        min_disp_px=float(settings["min_disparity_px"]),
        max_disp_px=float(settings["max_disparity_px"]),
        max_pd_error_px=float(settings["max_pd_error_px"]),
        coarse_mover=coarse_mover,
        planned_target=planned_target,
        actual_hits=actual_hits,
        gantry=gantry,
        return_debug=True,
    )

    chosen = reid_result.get("chosen")
    if not reid_result.get("ok") or chosen is None:
        print("[RE-ID DEBUG] No candidate selected by unified Re-ID pipeline.")
        if reid_result.get("error"):
            print(f"[RE-ID DEBUG] error: {reid_result['error']}")
        return None

    left_px = [float(chosen["left_px"][0]), float(chosen["left_px"][1])]
    right_px = [float(chosen["right_px"][0]), float(chosen["right_px"][1])]
    selected_target = {
        "target_xy_mm": [
            float(planned_target["target_xy_mm"][0]),
            float(planned_target["target_xy_mm"][1]),
        ],
        "source_target": {
            "left_px": left_px,
            "right_px": right_px,
            "left_cls": chosen.get("left_cls"),
            "right_cls": chosen.get("right_cls"),
            "left_conf": chosen.get("left_conf"),
            "right_conf": chosen.get("right_conf"),
            "score": float(chosen.get("score", 0.0)),
            "left_box": chosen.get("left_box"),
            "right_box": chosen.get("right_box"),
        },
    }
    print(
        f"[RE-ID DEBUG OK] chosen L={left_px} R={right_px} "
        f"pd_err={float(chosen.get('pd_err_px', 0.0)):.1f}px"
    )
    return selected_target


def _pick_best_ai_target(
    gantry, cameras, detector, coarse_mover,
    planned_target, actual_hits,
    expected_cls=None,
):
    t_pick = time.perf_counter()
    settings = _resolve_reid_settings()
    reid_result = run_fine_align_reid(
        cameras=cameras,
        detector=detector,
        target=planned_target,
        crop_w=int(settings["crop_half_px"]) * 2,
        crop_h=int(settings["crop_half_px"]) * 2,
        burst_count=int(settings["burst_count"]),
        min_hits=int(settings["min_hits"]),
        cluster_radius_px=float(settings["cluster_radius_px"]),
        point_mode=settings["point_mode"],
        class_filter=expected_cls,
        conf_override=None,
        imgsz=settings["yolo_imgsz"],
        use_rectified=True,
        y_gate_px=float(settings["max_y_diff_px"]),
        min_disp_px=float(settings["min_disparity_px"]),
        max_disp_px=float(settings["max_disparity_px"]),
        max_pd_error_px=float(settings["max_pd_error_px"]),
        coarse_mover=coarse_mover,
        planned_target=planned_target,
        actual_hits=actual_hits,
        gantry=gantry,
        return_debug=True,
    )

    chosen = reid_result.get("chosen")
    _pick_best_ai_target.last_reid_debug = {
        "reid_ok": bool(reid_result.get("ok", False)),
        "reid_error": reid_result.get("error"),
        "reid_filter_mode": reid_result.get("filter_mode", "basic"),
        "reid_filter_rejects": dict(reid_result.get("filter_rejects") or {}),
        "reid_left_count": len(reid_result.get("left_detections", [])),
        "reid_right_count": len(reid_result.get("right_detections", [])),
        "reid_match_count": len(reid_result.get("matches", [])),
        "reid_expected_cls": expected_cls,
        "reid_point_mode": settings["point_mode"],
        "reid_burst_count": int(settings["burst_count"]),
        "reid_chosen": bool(chosen is not None),
        "reid_chosen_detail": {
            "pd_err_px": float(chosen.get("pd_err_px", 0.0)) if chosen is not None else None,
            "tri_dist_mm": float(chosen.get("tri_dist_mm", 0.0)) if chosen is not None else None,
            "geo_score": float(chosen.get("geo_score", 0.0)) if chosen is not None else None,
        },
    }

    timing = dict(reid_result.get("timing", {}))
    timing.update({
        "reid_point_mode": settings["point_mode"],
        "reid_burst_count": int(settings["burst_count"]),
        "reid_total_time_s": float(timing.get("total_s", 0.0)),
        "reid_yolo_time_s": float(timing.get("yolo_left_s", 0.0)) + float(timing.get("yolo_right_s", 0.0)),
        "reid_matching_time_s": float(timing.get("match_s", 0.0)),
    })
    _pick_best_ai_target.last_reid_debug.update({
        "reid_timing": timing,
    })

    if not reid_result.get("ok") or chosen is None:
        print("[RE-ID FAIL] No candidate selected by unified Re-ID pipeline.")
        _fine_debug(f"initial target pick failed after {time.perf_counter() - t_pick:.2f}s")
        _fine_debug(f"reid error={reid_result.get('error')}")
        return None, None, None, None, None, None

    debug_frames = reid_result.get("debug_frames") or {}
    frameL = debug_frames.get("left_full")
    frameR = debug_frames.get("right_full")
    if frameL is None or frameR is None:
        frameL, frameR = cameras.read_pair()
    if frameL is None or frameR is None:
        return None, None, None, None, None, None

    left_px = [float(chosen["left_px"][0]), float(chosen["left_px"][1])]
    right_px = [float(chosen["right_px"][0]), float(chosen["right_px"][1])]
    selected_target = {
        "target_xy_mm": [
            float(chosen.get("tri_xy_mm", planned_target["target_xy_mm"])[0]),
            float(chosen.get("tri_xy_mm", planned_target["target_xy_mm"])[1]),
        ],
        "source_target": {
            "left_px": left_px,
            "right_px": right_px,
            "left_cls": chosen.get("left_cls"),
            "right_cls": chosen.get("right_cls"),
            "left_conf": chosen.get("left_conf"),
            "right_conf": chosen.get("right_conf"),
            "score": float(chosen.get("score", 0.0)),
            "left_box": chosen.get("left_box"),
            "right_box": chosen.get("right_box"),
        },
    }

    print(
        f"[RE-ID OK] {len(reid_result.get('matches', []))} candidate(s) — "
        f"picked pd_err={float(chosen.get('pd_err_px', 0.0)):.1f}px"
    )
    _fine_debug(f"initial target pick took {time.perf_counter() - t_pick:.2f}s")
    reid_info = {
        "pd_error_px": round(float(chosen.get("pd_err_px", 0.0)), 1),
        "n_candidates": len(reid_result.get("matches", [])),
        "tri_xy_mm": [round(v, 3) for v in selected_target["target_xy_mm"]],
        "timing": timing,
    }
    return left_px, right_px, frameL, frameR, selected_target, reid_info


def _pick_initial_target(
    gantry, cameras, detector, coarse_mover,
    planned_target, actual_hits, expected_cls=None,
):
    if DETECTOR_MODE == "manual":
        return _pick_manual_initial_target(cameras, detector)
    if DETECTOR_MODE == "ai":
        return _pick_best_ai_target(
            gantry, cameras, detector, coarse_mover,
            planned_target, actual_hits,
            expected_cls=expected_cls,
        )
    raise ValueError(f"Unknown DETECTOR_MODE: {DETECTOR_MODE}")


def _show_fine_align_window(frameL, frameR, left_pt, right_pt,
                            err_x, err_y, dx, dy, target_idx=None, total=None):
    dispL = frameL.copy()
    dispR = frameR.copy()

    xl = int(round(left_pt[0]))
    yl = int(round(left_pt[1]))
    xr = int(round(right_pt[0]))
    yr = int(round(right_pt[1]))

    cv2.circle(dispL, (xl, yl), 5, (0, 0, 255), -1)
    cv2.circle(dispR, (xr, yr), 5, (0, 0, 255), -1)

    for disp in (dispL, dispR):
        cv2.line(disp, (int(TARGET_X), 0), (int(TARGET_X), FINE_H), (255, 0, 0), 1)
        cv2.line(disp, (0, int(TARGET_Y)), (FINE_W, int(TARGET_Y)), (255, 0, 0), 1)

    cv2.putText(dispL, "LEFT", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(dispR, "RIGHT", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    combined = np.hstack([dispL, dispR])
    tgt_str = f"{target_idx}/{total} | " if target_idx is not None else ""
    status = f"{tgt_str}ex={err_x:.1f}px ey={err_y:.1f}px | dx={dx:.3f} dy={dy:.3f}"
    cv2.putText(combined, status, (10, FINE_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow(_FINE_WINDOW, combined)


def fine_align_target(
    gantry,
    cameras,
    detector,
    coarse_mover,
    planned_target,
    actual_hits,
    max_time=FINE_ALIGN_MAX_TIME_SEC,
    settle_frames=FINE_ALIGN_SETTLE_FRAMES,
    show_debug=HAS_DISPLAY,
    target_idx=None,
    total_targets=None,
):
    target_label = f"{target_idx}/{total_targets}" if target_idx is not None else "?"
    snap_point_mode = resolve_point_mode(FINAL_SNAP_POINT_MODE)
    snap_enabled = (
        bool(FINE_ALIGN_ENABLE_SNAP)
        and str(FINE_ALIGN_SNAP_MODE).lower() == "qpoint"
        and snap_point_mode == "qpoint"
    )
    snap_burst_count = resolve_burst_count(FINAL_SNAP_BURST_COUNT)
    fine_align_target.last_timing = {
        "fine_align_reid_yolo_time_s": 0.0,
        "fine_align_reid_total_time_s": 0.0,
        "fine_align_pd_lk_time_s": 0.0,
        "final_snap_time_s": 0.0,
        "fine_align_iterations": 0,
        "fine_align_snap_used": False,
        "fine_align_snap_move_px": None,
        "fine_align_snap_mode": FINE_ALIGN_SNAP_MODE,
        "final_snap_point_mode": snap_point_mode,
    }
    fine_align_target.last_reid_debug = {}
    _fine_debug(
        f"start target {target_label}: max_time={max_time:.1f}s "
        f"settle_frames={settle_frames} deadzone={DEADZONE:.1f}px crop={FINE_W}x{FINE_H} "
        f"snap_enabled={snap_enabled} snap_mode={FINE_ALIGN_SNAP_MODE} "
        f"final_point_mode={snap_point_mode} snap_burst={snap_burst_count}"
    )

    if cameras is not None:
        cameras.clear_overlay()

    effective_show = show_debug and (DETECTOR_MODE != "manual")
    if effective_show:
        cv2.namedWindow(_FINE_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_FINE_WINDOW, FINE_W * 2, FINE_H)
        cv2.moveWindow(_FINE_WINDOW, 80, 80)
        try:
            cv2.setWindowProperty(_FINE_WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    expected_cls = planned_target.get("source_target", {}).get("left_cls")

    left_pt_full, right_pt_full, old_left_full, old_right_full, selected_target, reid_info = _pick_initial_target(
        gantry, cameras, detector, coarse_mover,
        planned_target, actual_hits,
        expected_cls=expected_cls,
    )
    if DETECTOR_MODE == "ai":
        fine_align_target.last_reid_debug = dict(getattr(_pick_best_ai_target, "last_reid_debug", {}) or {})
    reid_timing = reid_info.get("timing") if isinstance(reid_info, dict) else None
    if not reid_timing:
        reid_timing = getattr(_burst_match_ai_points, "last_timing", {})
    if reid_timing:
        fine_align_target.last_timing.update({
            "fine_align_reid_yolo_time_s": float(reid_timing.get("reid_yolo_time_s", 0.0)),
            "fine_align_reid_total_time_s": float(reid_timing.get("reid_total_time_s", 0.0)),
            "fine_align_reid_grouping_time_s": float(reid_timing.get("reid_grouping_time_s", 0.0)),
            "fine_align_reid_matching_time_s": float(reid_timing.get("reid_matching_time_s", 0.0)),
            "fine_align_reid_point_mode": reid_timing.get("reid_point_mode"),
            "fine_align_reid_burst_count": reid_timing.get("reid_burst_count"),
        })

    if left_pt_full is None or right_pt_full is None:
        print("\n[!] FINE ALIGN: Detection/Matching failed.")
        print("[!] Skipping strike. Moving to next target.")
        _fine_debug(f"target {target_label} failed before tracking loop")
        return False, None

    if not (_point_inside_crop(left_pt_full) and _point_inside_crop(right_pt_full)):
        print("[FINE ALIGN FAIL] Target is outside the centre crop. Skipping strike.")
        _fine_debug(f"target {target_label} rejected outside centre crop")
        return False, None

    left_pt = _full_to_crop_point(left_pt_full)
    right_pt = _full_to_crop_point(right_pt_full)
    old_left = _crop_frame(old_left_full)
    old_right = _crop_frame(old_right_full)

    track_pt_L = np.array([[left_pt]], dtype=np.float32)
    track_pt_R = np.array([[right_pt]], dtype=np.float32)

    old_gray_L = cv2.cvtColor(old_left, cv2.COLOR_BGR2GRAY)
    old_gray_R = cv2.cvtColor(old_right, cv2.COLOR_BGR2GRAY)

    gantry.sync_estimate_to_machine()
    est_x, est_y = gantry.get_estimated_xy()

    prev_ex = 0.0
    prev_ey = 0.0
    inside_cnt = 0
    t0 = time.time()
    iterations = 0
    last_err_x = None
    last_err_y = None
    last_dx = 0.0
    last_dy = 0.0

    _snap_done = not snap_enabled
    _active_settle = settle_frames
    snap_move_px = None
    snap_used = False
    t_pd_loop = time.perf_counter()

    def _update_pd_timing():
        snap_time = float(fine_align_target.last_timing.get("final_snap_time_s", 0.0))
        fine_align_target.last_timing.update({
            "fine_align_pd_lk_time_s": round(max(0.0, time.perf_counter() - t_pd_loop - snap_time), 6),
            "fine_align_iterations": int(iterations),
            "fine_align_snap_used": bool(snap_used),
            "fine_align_snap_move_px": snap_move_px,
        })

    while time.time() - t0 < max_time:
        frameL_full, frameR_full = cameras.read_pair()
        if frameL_full is None or frameR_full is None:
            continue
        iterations += 1

        frameL = _crop_frame(frameL_full)
        frameR = _crop_frame(frameR_full)
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        new_pt_L, stL, _ = cv2.calcOpticalFlowPyrLK(old_gray_L, grayL, track_pt_L, None, **LK_PARAMS)
        new_pt_R, stR, _ = cv2.calcOpticalFlowPyrLK(old_gray_R, grayR, track_pt_R, None, **LK_PARAMS)

        if stL is None or stR is None or stL[0][0] == 0 or stR[0][0] == 0:
            gantry.stop()
            end_live_fine_align()
            _update_pd_timing()
            print("[FINE ALIGN FAIL] Lost LK tracking.")
            return False, None

        track_pt_L = new_pt_L
        track_pt_R = new_pt_R

        xl = float(track_pt_L[0, 0, 0])
        yl = float(track_pt_L[0, 0, 1])
        xr = float(track_pt_R[0, 0, 0])
        yr = float(track_pt_R[0, 0, 1])

        if not (0 <= xl < FINE_W and 0 <= yl < FINE_H and 0 <= xr < FINE_W and 0 <= yr < FINE_H):
            gantry.stop()
            end_live_fine_align()
            _update_pd_timing()
            print("[FINE ALIGN FAIL] Tracked point left the centre crop.")
            return False, None

        err_x, err_y = _compute_errors((xl, yl), (xr, yr))
        err_x += -5.0
        err_y += -1.0
        last_err_x = err_x
        last_err_y = err_y

        if cameras is not None:
            cameras.set_recording_context(
                pd_error_x_px=round(err_x, 3),
                pd_error_y_px=round(err_y, 3),
                pd_locked=bool(abs(err_x) <= DEADZONE and abs(err_y) <= DEADZONE),
                pd_settle_count=int(inside_cnt),
            )

        dex = err_x - prev_ex
        dey = err_y - prev_ey
        prev_ex = err_x
        prev_ey = err_y

        dx, dy = 0.0, 0.0
        if abs(err_x) > DEADZONE:
            dx = round((err_x * Kp_x + dex * Kd_x) * STEP_MM, 3)
        if abs(err_y) > DEADZONE:
            dy = round((err_y * Kp_y + dey * Kd_y) * STEP_MM, 3)

        dx = float(np.clip(dx, -MAX_JOG, MAX_JOG))
        dy = float(np.clip(dy, -MAX_JOG, MAX_JOG))

        dx, dy = _clamp_jog(dx, dy, est_x, est_y)
        last_dx = dx
        last_dy = dy

        print_live_fine_align(
            err_x, err_y, dx, dy,
            planned_xy=planned_target["target_xy_mm"],
            throttle_s=0.25,
            settle_count=inside_cnt,
            settle_frames=_active_settle,
            elapsed_s=time.time() - t0,
            max_time_s=max_time,
        )

        if cameras is not None and cameras._recorder is not None and RECORD_LIVE_OVERLAYS:
            tgt_str = f"{target_idx}/{total_targets} | " if target_idx is not None else ""
            _status = f"{tgt_str}ex={err_x:.1f}px ey={err_y:.1f}px"
            recL = frameL_full.copy()
            recR = frameR_full.copy()
            cv2.circle(recL, (int(xl + CROP_X0), int(yl + CROP_Y0)), 8, (0, 0, 255), -1)
            cv2.circle(recR, (int(xr + CROP_X0), int(yr + CROP_Y0)), 8, (0, 0, 255), -1)
            cv2.putText(recL, _status, (12, FULL_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cameras._recorder.write_overlay(recL, recR)

        if abs(err_x) <= DEADZONE and abs(err_y) <= DEADZONE:
            inside_cnt += 1
            gantry.stop()

            if (
                inside_cnt >= _active_settle
                and not _snap_done
                and FINE_ALIGN_SNAP_ON_DEADZONE
                and DETECTOR_MODE == "ai"
                and detector is not None
                and selected_target is not None
            ):
                # Snap: crop 1.2× the initial bbox around the current LK point,
                # run qpoint directly (no YOLO) to get a sub-pixel meristem location,
                # then re-seed LK for a final short settle to cancel any drift.
                t_snap = time.perf_counter()
                _src_box = selected_target.get("source_target", {}).get("left_box")
                if FINE_ALIGN_SNAP_CROP_HALF_PX is not None:
                    _px = _py = max(4, int(FINE_ALIGN_SNAP_CROP_HALF_PX))
                elif _src_box:
                    _bw = max(20, _src_box[2] - _src_box[0])
                    _bh = max(20, _src_box[3] - _src_box[1])
                    _px = max(24, int(_bw * 0.6))
                    _py = max(24, int(_bh * 0.6))
                else:
                    _bw = _bh = 60
                    _px = max(24, int(_bw * 0.6))
                    _py = max(24, int(_bh * 0.6))

                _xl_f, _yl_f = xl + CROP_X0, yl + CROP_Y0
                _xr_f, _yr_f = xr + CROP_X0, yr + CROP_Y0

                _lsx0 = max(0, int(_xl_f) - _px);  _lsx1 = min(FULL_W, int(_xl_f) + _px)
                _lsy0 = max(0, int(_yl_f) - _py);  _lsy1 = min(FULL_H, int(_yl_f) + _py)
                _rsx0 = max(0, int(_xr_f) - _px);  _rsx1 = min(FULL_W, int(_xr_f) + _px)
                _rsy0 = max(0, int(_yr_f) - _py);  _rsy1 = min(FULL_H, int(_yr_f) + _py)

                _snap_cropL = frameL_full[_lsy0:_lsy1, _lsx0:_lsx1]
                _snap_cropR = frameR_full[_rsy0:_rsy1, _rsx0:_rsx1]
                if snap_burst_count > 1:
                    _crops_l = [_snap_cropL]
                    _crops_r = [_snap_cropR]
                    for _ in range(snap_burst_count - 1):
                        _extra_l, _extra_r = cameras.read_pair()
                        if _extra_l is None or _extra_r is None:
                            continue
                        _crops_l.append(_extra_l[_lsy0:_lsy1, _lsx0:_lsx1])
                        _crops_r.append(_extra_r[_rsy0:_rsy1, _rsx0:_rsx1])
                    if len(_crops_l) > 1:
                        _snap_cropL = np.mean(np.stack(_crops_l), axis=0).astype(np.uint8)
                        _snap_cropR = np.mean(np.stack(_crops_r), axis=0).astype(np.uint8)

                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="snap") as _sp:
                    _fl = _sp.submit(detector.cv_left.snap_meristem_on_crop, _snap_cropL)
                    _fr = _sp.submit(detector.cv_right.snap_meristem_on_crop, _snap_cropR)
                    _spt_l = _fl.result()
                    _spt_r = _fr.result()
                snap_dt = time.perf_counter() - t_snap
                fine_align_target.last_timing["final_snap_time_s"] = round(
                    fine_align_target.last_timing.get("final_snap_time_s", 0.0) + snap_dt, 6
                )

                if _spt_l is not None and _spt_r is not None:
                    _new_xl = float(_spt_l[0] + _lsx0) - CROP_X0
                    _new_yl = float(_spt_l[1] + _lsy0) - CROP_Y0
                    _new_xr = float(_spt_r[0] + _rsx0) - CROP_X0
                    _new_yr = float(_spt_r[1] + _rsy0) - CROP_Y0
                    if (0 <= _new_xl < FINE_W and 0 <= _new_yl < FINE_H
                            and 0 <= _new_xr < FINE_W and 0 <= _new_yr < FINE_H):
                        move_l = float(np.hypot(_new_xl - xl, _new_yl - yl))
                        move_r = float(np.hypot(_new_xr - xr, _new_yr - yr))
                        snap_move_px = round((move_l + move_r) / 2.0, 3)
                        snap_used = True
                        _fine_debug(
                            f"snap re-seed: L ({xl:.1f},{yl:.1f})→({_new_xl:.1f},{_new_yl:.1f}) "
                            f"R ({xr:.1f},{yr:.1f})→({_new_xr:.1f},{_new_yr:.1f}) "
                            f"move={snap_move_px:.2f}px time={snap_dt:.3f}s"
                        )
                        print(f"[SNAP] qpoint snap used; mean move={snap_move_px:.2f}px time={snap_dt:.3f}s")
                        track_pt_L = np.array([[_new_xl, _new_yl]], dtype=np.float32).reshape(1, 1, 2)
                        track_pt_R = np.array([[_new_xr, _new_yr]], dtype=np.float32).reshape(1, 1, 2)
                        old_gray_L = grayL
                        old_gray_R = grayR
                        inside_cnt = 0
                        _snap_done = True
                        _active_settle = FINE_ALIGN_SNAP_SETTLE_FRAMES
                        continue
                    else:
                        print(f"[SNAP] qpoint snap skipped; refined point left crop after {snap_dt:.3f}s")
                        _fine_debug("snap point outside crop, using LK position")
                else:
                    print(f"[SNAP] qpoint snap failed; using LK point after {snap_dt:.3f}s")
                    _fine_debug("snap failed (qpoint None), using LK position")
                _snap_done = True  # don't retry snap even if it failed

            if inside_cnt >= _active_settle:
                gantry.wait_for_idle(timeout=3.0)
                end_live_fine_align()
                print(f"Fine align locked: ex={err_x:.2f}px ey={err_y:.2f}px")
                _update_pd_timing()
                if not snap_enabled:
                    print("[SNAP] disabled; firing from LK/PD lock.")
                elif not snap_used:
                    print("[SNAP] not used; firing from LK/PD lock.")

                gantry.sync_estimate_to_machine()
                final_xy = gantry.get_estimated_xy()

                if DETECTOR_MODE == "ai" and selected_target is not None:
                    actual_entry = coarse_mover.append_actual_target(
                        planned_target, selected_target, final_xy,
                        filename="actual_pd_targets.json",
                    )
                else:
                    px, py = planned_target["target_xy_mm"]
                    fx, fy = final_xy
                    actual_entry = {
                        "planned_xy_mm": [float(px), float(py)],
                        "selected_local_xy_mm": [float(px), float(py)],
                        "final_xy_mm": [float(fx), float(fy)],
                        "left_px": [float(left_pt_full[0]), float(left_pt_full[1])],
                        "right_px": [float(right_pt_full[0]), float(right_pt_full[1])],
                        "score": 1.0,
                    }

                if actual_entry is not None and reid_info is not None:
                    actual_entry["reid_protocol"] = reid_info
                if actual_entry is not None:
                    actual_entry["timing"] = dict(fine_align_target.last_timing)

                if cameras is not None:
                    cls_name = "Target"
                    if selected_target is not None and detector is not None:
                        src = selected_target.get("source_target", {})
                        cls_id = src.get("left_cls")
                        if cls_id is not None:
                            cls_name = detector.cv_left.yolo.names.get(int(cls_id), str(cls_id))
                    tgt_str = f"{target_idx}/{total_targets} " if target_idx is not None else ""
                    fx2, fy2 = final_xy
                    cameras.set_recording_status([
                        f"Target {tgt_str}",
                        f"LOCKED IN: {cls_name}",
                        f"Final: ({fx2:.1f}, {fy2:.1f}) mm",
                    ])

                return True, actual_entry
        else:
            inside_cnt = 0
            if dx != 0.0 or dy != 0.0:
                gantry.jog(dx, dy, FINE_FEED)
                est_x += dx
                est_y += dy

        if effective_show:
            _show_fine_align_window(
                frameL, frameR, (xl, yl), (xr, yr),
                err_x, err_y, dx, dy,
                target_idx=target_idx, total=total_targets,
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("r")):
                gantry.stop()
                end_live_fine_align()
                print("[FINE ALIGN FAIL] Cancelled by user.")
                return False, None

        old_gray_L = grayL
        old_gray_R = grayR

    gantry.stop()
    end_live_fine_align("timeout")
    _update_pd_timing()
    elapsed = time.time() - t0
    if last_err_x is None or last_err_y is None:
        print(f"[FINE ALIGN FAIL] Timeout after {elapsed:.1f}s before any usable tracking frame.")
    else:
        print(
            f"[FINE ALIGN FAIL] Timeout after {elapsed:.1f}s: "
            f"iterations={iterations}, last_err=({last_err_x:+.1f}, {last_err_y:+.1f})px, "
            f"last_jog=({last_dx:+.4f}, {last_dy:+.4f})mm, "
            f"settle={inside_cnt}/{settle_frames}."
        )
    return False, None
