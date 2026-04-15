import time
import cv2
import numpy as np
from ui.terminal import print_live_fine_align, end_live_fine_align
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
    FINE_ALIGN_MIN_HITS,
    FINE_ALIGN_CLUSTER_RADIUS_PX,
    FINE_ALIGN_MAX_TIME_SEC,
    FINE_ALIGN_SETTLE_FRAMES,
    FINE_ALIGN_LK_REDETECT_INTERVAL,
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
BURST_COUNT = FINE_ALIGN_BURST_COUNT
MIN_HITS = FINE_ALIGN_MIN_HITS
CLUSTER_RADIUS_PX = FINE_ALIGN_CLUSTER_RADIUS_PX

_REDETECT_INTERVAL = FINE_ALIGN_LK_REDETECT_INTERVAL
_SNAP_RADIUS_PX = 25.0

_WS_MARGIN = 5.0

_FINE_WINDOW = "Fine Align"


def close_fine_align_window():
    if HAS_DISPLAY:
        try:
            cv2.destroyWindow(_FINE_WINDOW)
        except Exception:
            pass


def _constellation_reid_score(
    candidate_left_pt,
    all_current_left_pts,
    target_survey_left_pt,
    all_survey_left_pts,
    match_radius=80.0,
    neighbor_radius=600.0,
):
    if not all_current_left_pts or len(all_survey_left_pts) < 2:
        return 0.5

    sx = candidate_left_pt[0] - target_survey_left_pt[0]
    sy = candidate_left_pt[1] - target_survey_left_pt[1]

    neighbours = [
        p for p in all_survey_left_pts
        if p != target_survey_left_pt
        and np.hypot(p[0] - target_survey_left_pt[0], p[1] - target_survey_left_pt[1]) <= neighbor_radius
    ]
    if not neighbours:
        return 0.5

    hits = 0
    for sp in neighbours:
        pred_x = sp[0] + sx
        pred_y = sp[1] + sy
        for cp in all_current_left_pts:
            if np.hypot(cp[0] - pred_x, cp[1] - pred_y) <= match_radius:
                hits += 1
                break

    return hits / len(neighbours)


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


def _clamp_jog(dx, dy, est_x, est_y):
    new_x = np.clip(est_x + dx, WORKSPACE_X_MIN + _WS_MARGIN, WORKSPACE_X_MAX - _WS_MARGIN)
    new_y = np.clip(est_y + dy, WORKSPACE_Y_MIN + _WS_MARGIN, WORKSPACE_Y_MAX - _WS_MARGIN)
    return float(new_x - est_x), float(new_y - est_y)


def _burst_match_ai_points(cameras, detector):
    left_frames, right_frames = [], []
    for _ in range(BURST_COUNT):
        fL, fR = cameras.read_pair()
        left_frames.append(fL)
        right_frames.append(fR)

    stable_left = detector.cv_left.return_burst_stable(
        left_frames, min_stable_views=MIN_HITS, group_radius_px=CLUSTER_RADIUS_PX
    )
    stable_right = detector.cv_right.return_burst_stable(
        right_frames, min_stable_views=MIN_HITS, group_radius_px=CLUSTER_RADIUS_PX
    )

    if not stable_left or not stable_right:
        return []

    matched_targets, _, _ = match_points(stable_left, stable_right, verbose=False)

    filtered = []
    for t in matched_targets:
        xl, yl = t["left_px"]
        xr, yr = t["right_px"]
        if abs(yl - yr) > 60:
            continue
        disp = abs(xl - xr)
        if disp < 10 or disp > 500:
            continue
        filtered.append(t)

    return filtered


def _pick_manual_initial_target(cameras, detector):
    left_pt, right_pt = detector.refine_live(cameras)
    if left_pt is None or right_pt is None:
        return None, None, None, None, None

    frameL = getattr(detector, "last_displayed_left", None)
    frameR = getattr(detector, "last_displayed_right", None)
    if frameL is None or frameR is None:
        frameL, frameR = cameras.read_pair()
    if frameL is None or frameR is None:
        return None, None, None, None, None

    return left_pt, right_pt, frameL.copy(), frameR.copy(), None


def _pick_best_ai_target(
    gantry, cameras, detector, coarse_mover,
    planned_target, actual_hits,
    survey_targets=None,
):
    matched_targets = _burst_match_ai_points(cameras, detector)

    gantry.sync_estimate_to_machine()
    current_x, current_y = gantry.get_estimated_xy()

    solved_all = (
        coarse_mover.solve_all_from_pose(matched_targets, ref_x=current_x, ref_y=current_y)
        if matched_targets else []
    )

    have_constellation = (
        survey_targets is not None
        and len(survey_targets) > 1
        and "source_target" in planned_target
        and "left_px" in planned_target["source_target"]
    )

    if have_constellation:
        survey_left_pts = [
            t["source_target"]["left_px"]
            for t in survey_targets
            if "source_target" in t and "left_px" in t["source_target"]
        ]
        target_survey_pt = planned_target["source_target"]["left_px"]
    else:
        survey_left_pts = []
        target_survey_pt = None

    frameL_fresh, frameR_fresh = cameras.read_pair()
    if frameL_fresh is None:
        frameL_fresh, frameR_fresh = cameras.read_pair()

    fresh_left_pts = detector.cv_left.detect_points(frameL_fresh) if frameL_fresh is not None else []

    best_solved = None
    best_score = float("inf")
    best_left = None
    best_right = None

    for t_solved in solved_all:
        t_source = t_solved["source_target"]
        left_pt = t_source["left_px"]
        right_pt = t_source["right_px"]

        if not (_point_inside_crop(left_pt) and _point_inside_crop(right_pt)):
            continue
        if coarse_mover.is_duplicate_of_actual(
            t_solved["target_xy_mm"], actual_hits, tol_mm=15.0
        ):
            continue

        xl, yl = left_pt
        xr, yr = right_pt
        err_x = (xl + xr) - (2.0 * (FULL_W / 2.0))
        err_y = (FULL_H / 2.0 - 5) - ((yl + yr) / 2.0)
        img_err = float(np.hypot(err_x, err_y))

        if have_constellation and fresh_left_pts:
            c_score = _constellation_reid_score(
                left_pt, fresh_left_pts, target_survey_pt, survey_left_pts
            )
        else:
            c_score = 0.5

        norm_img_err = img_err / 500.0
        combined_cost = 0.5 * norm_img_err + 0.5 * (1.0 - c_score)

        if combined_cost < best_score:
            best_score = combined_cost
            best_solved = t_solved
            best_left = left_pt
            best_right = right_pt

    frameL, frameR = cameras.read_pair()
    if frameL is None or frameR is None:
        return None, None, None, None, None

    fresh_left_pts2 = detector.cv_left.detect_points(frameL)
    fresh_right_pts2 = detector.cv_right.detect_points(frameR)

    final_left = best_left
    final_right = best_right

    print(f"\n[DEBUG] Ghost Protocol Verification:")

    if fresh_left_pts2 and best_left:
        dists = [np.hypot(p[0] - best_left[0], p[1] - best_left[1]) for p in fresh_left_pts2]
        min_dist = min(dists)
        if min_dist < 80:
            final_left = fresh_left_pts2[int(np.argmin(dists))]
            print(f"  [OK] Left snapped to fresh YOLO point ({min_dist:.1f}px).")
        else:
            print("  [WARN] Left YOLO missed. Trusting burst point.")
    else:
        print("  [WARN] Left YOLO empty. Trusting burst point.")

    if fresh_right_pts2 and best_right:
        dists = [np.hypot(p[0] - best_right[0], p[1] - best_right[1]) for p in fresh_right_pts2]
        min_dist = min(dists)
        if min_dist < 80:
            final_right = fresh_right_pts2[int(np.argmin(dists))]
            print(f"  [OK] Right snapped to fresh YOLO point ({min_dist:.1f}px).")
        else:
            print("  [WARN] Right YOLO missed. Trusting burst point.")
    else:
        print("  [WARN] Right YOLO empty. Trusting burst point.")

    if final_left is None or final_right is None:
        return None, None, None, None, None

    return final_left, final_right, frameL, frameR, best_solved


def _pick_initial_target(
    gantry, cameras, detector, coarse_mover,
    planned_target, actual_hits, survey_targets=None,
):
    if DETECTOR_MODE == "manual":
        return _pick_manual_initial_target(cameras, detector)
    if DETECTOR_MODE == "ai":
        return _pick_best_ai_target(
            gantry, cameras, detector, coarse_mover,
            planned_target, actual_hits, survey_targets=survey_targets,
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

    cv2.namedWindow(_FINE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_FINE_WINDOW, combined.shape[1], combined.shape[0])
    cv2.moveWindow(_FINE_WINDOW, 80, 80)
    try:
        cv2.setWindowProperty(_FINE_WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.imshow(_FINE_WINDOW, combined)


def _final_deadzone_ai_check(detector, frameL_full, frameR_full, xl, yl, xr, yr):
    xl_full = xl + CROP_X0
    yl_full = yl + CROP_Y0
    xr_full = xr + CROP_X0
    yr_full = yr + CROP_Y0

    fresh_L = detector.cv_left.detect_points(frameL_full)
    fresh_R = detector.cv_right.detect_points(frameR_full)

    snapped = False

    if fresh_L:
        dists = [np.hypot(p[0] - xl_full, p[1] - yl_full) for p in fresh_L]
        min_i = int(np.argmin(dists))
        if dists[min_i] < _SNAP_RADIUS_PX:
            xl = fresh_L[min_i][0] - CROP_X0
            yl = fresh_L[min_i][1] - CROP_Y0
            snapped = True

    if fresh_R:
        dists = [np.hypot(p[0] - xr_full, p[1] - yr_full) for p in fresh_R]
        min_i = int(np.argmin(dists))
        if dists[min_i] < _SNAP_RADIUS_PX:
            xr = fresh_R[min_i][0] - CROP_X0
            yr = fresh_R[min_i][1] - CROP_Y0
            snapped = True

    return xl, yl, xr, yr, snapped


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
    survey_targets=None,
    target_idx=None,
    total_targets=None,
):
    left_pt_full, right_pt_full, old_left_full, old_right_full, selected_target = _pick_initial_target(
        gantry, cameras, detector, coarse_mover,
        planned_target, actual_hits, survey_targets=survey_targets,
    )

    if left_pt_full is None or right_pt_full is None:
        print(f"\n[!] FINE ALIGN: Detection/Matching failed.")
        print("[!] Skipping strike. Moving to next target.")
        return False, None

    if not (_point_inside_crop(left_pt_full) and _point_inside_crop(right_pt_full)):
        print("[FINE ALIGN FAIL] Target is outside the centre crop. Skipping strike.")
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
    frame_count = 0
    t0 = time.time()

    effective_show = show_debug and (DETECTOR_MODE != "manual")

    while time.time() - t0 < max_time:
        frameL_full, frameR_full = cameras.read_pair()
        if frameL_full is None or frameR_full is None:
            continue

        frameL = _crop_frame(frameL_full)
        frameR = _crop_frame(frameR_full)
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        new_pt_L, stL, _ = cv2.calcOpticalFlowPyrLK(old_gray_L, grayL, track_pt_L, None, **LK_PARAMS)
        new_pt_R, stR, _ = cv2.calcOpticalFlowPyrLK(old_gray_R, grayR, track_pt_R, None, **LK_PARAMS)

        if stL is None or stR is None or stL[0][0] == 0 or stR[0][0] == 0:
            gantry.stop()
            end_live_fine_align()
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
            print("[FINE ALIGN FAIL] Tracked point left the centre crop.")
            return False, None

        frame_count += 1
        if DETECTOR_MODE == "ai" and frame_count % _REDETECT_INTERVAL == 0:
            fresh_L = detector.cv_left.detect_points(frameL_full)
            fresh_R = detector.cv_right.detect_points(frameR_full)

            xl_full = xl + CROP_X0
            yl_full = yl + CROP_Y0
            xr_full = xr + CROP_X0
            yr_full = yr + CROP_Y0

            if fresh_L:
                dists = [np.hypot(p[0] - xl_full, p[1] - yl_full) for p in fresh_L]
                min_i = int(np.argmin(dists))
                if dists[min_i] < _SNAP_RADIUS_PX:
                    xl = fresh_L[min_i][0] - CROP_X0
                    yl = fresh_L[min_i][1] - CROP_Y0
                    track_pt_L = np.array([[[xl, yl]]], dtype=np.float32)

            if fresh_R:
                dists = [np.hypot(p[0] - xr_full, p[1] - yr_full) for p in fresh_R]
                min_i = int(np.argmin(dists))
                if dists[min_i] < _SNAP_RADIUS_PX:
                    xr = fresh_R[min_i][0] - CROP_X0
                    yr = fresh_R[min_i][1] - CROP_Y0
                    track_pt_R = np.array([[[xr, yr]]], dtype=np.float32)

        err_x, err_y = _compute_errors((xl, yl), (xr, yr))
        err_x += -5.0
        err_y += -1.0

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

        print_live_fine_align(
            err_x, err_y, dx, dy,
            planned_xy=planned_target["target_xy_mm"],
            throttle_s=0.25,
        )

        if abs(err_x) <= DEADZONE and abs(err_y) <= DEADZONE:
            inside_cnt += 1
            gantry.stop()

            if inside_cnt >= settle_frames:
                if DETECTOR_MODE == "ai":
                    xl, yl, xr, yr, snapped = _final_deadzone_ai_check(
                        detector, frameL_full, frameR_full, xl, yl, xr, yr
                    )
                    if snapped:
                        track_pt_L = np.array([[[xl, yl]]], dtype=np.float32)
                        track_pt_R = np.array([[[xr, yr]]], dtype=np.float32)

                    err_x, err_y = _compute_errors((xl, yl), (xr, yr))
                    err_x += -5.0
                    err_y += -1.0

                    if not (abs(err_x) <= DEADZONE and abs(err_y) <= DEADZONE):
                        inside_cnt = 0
                        old_gray_L = grayL
                        old_gray_R = grayR
                        continue

                gantry.wait_for_idle(timeout=3.0)
                end_live_fine_align()
                print(f"Fine align locked: ex={err_x:.2f}px ey={err_y:.2f}px")

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
    end_live_fine_align()
    print("[FINE ALIGN FAIL] Timeout.")
    return False, None