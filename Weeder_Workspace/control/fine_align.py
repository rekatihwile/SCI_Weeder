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


def _safe_destroy_fine_align_window():
    if HAS_DISPLAY:
        try:
            cv2.destroyWindow("Fine Align")
        except Exception:
            pass


def _crop_frame(frame):
    return frame[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1].copy()


def _full_to_crop_point(pt):
    x = float(pt[0]) - CROP_X0
    y = float(pt[1]) - CROP_Y0
    return (x, y)


def _point_inside_crop(pt, margin=8.0):
    x, y = _full_to_crop_point(pt)
    return margin <= x < (FINE_W - margin) and margin <= y < (FINE_H - margin)


def _compute_errors(left_pt, right_pt):
    xl, yl = left_pt
    xr, yr = right_pt
    err_x = (xl + xr) - (2.0 * TARGET_X)
    err_y = TARGET_Y - ((yl + yr) / 2.0)
    return err_x, err_y


def _norm_xy(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _burst_match_ai_points(cameras, detector):
    left_frames = []
    right_frames = []

    for _ in range(BURST_COUNT):
        frameL, frameR = cameras.read_pair()
        left_frames.append(frameL)
        right_frames.append(frameR)

    stable_left = detector.cv_left.return_burst_stable(
        left_frames,
        min_stable_views=MIN_HITS,
        group_radius_px=CLUSTER_RADIUS_PX,
    )
    stable_right = detector.cv_right.return_burst_stable(
        right_frames,
        min_stable_views=MIN_HITS,
        group_radius_px=CLUSTER_RADIUS_PX,
    )

    if not stable_left or not stable_right:
        return []

    matched_targets, _, _ = match_points(stable_left, stable_right, verbose=False)

    filtered = []
    for t in matched_targets:
        xl, yl = t["left_px"]
        xr, yr = t["right_px"]
        y_err = abs(yl - yr)
        disp = abs(xl - xr)

        if y_err > 60:
            continue
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
        print("[FINE ALIGN] Failed to acquire frozen manual frames.")
        return None, None, None, None, None

    return left_pt, right_pt, frameL.copy(), frameR.copy(), None


def _pick_best_ai_target(gantry, cameras, detector, coarse_mover, planned_target, actual_hits):
    matched_targets = _burst_match_ai_points(cameras, detector)

    gantry.sync_estimate_to_machine()
    current_x, current_y = gantry.get_estimated_xy()

    solved_all = (
        coarse_mover.solve_all_from_pose(matched_targets, ref_x=current_x, ref_y=current_y)
        if matched_targets else []
    )

    best_solved = None
    min_err = float("inf")
    best_left = None
    best_right = None

    for t_solved in solved_all:
        t_source = t_solved["source_target"]
        left_pt = t_source["left_px"]
        right_pt = t_source["right_px"]

        if not (_point_inside_crop(left_pt) and _point_inside_crop(right_pt)):
            continue

        if coarse_mover.is_duplicate_of_actual(
            t_solved["target_xy_mm"],
            actual_hits,
            tol_mm=15.0,
        ):
            continue

        xl, yl = left_pt
        xr, yr = right_pt
        err_x = (xl + xr) - (2.0 * (FULL_W / 2.0))
        err_y = (FULL_H / 2.0 - 5) - ((yl + yr) / 2.0)
        image_err = float(np.hypot(err_x, err_y))

        if image_err < min_err:
            min_err = image_err
            best_solved = t_solved
            best_left = left_pt
            best_right = right_pt

    time.sleep(0.15)

    frameL, frameR = cameras.read_pair()
    if frameL is None or frameR is None:
        return None, None, None, None, None

    fresh_left_pts = detector.cv_left.detect_points(frameL)
    fresh_right_pts = detector.cv_right.detect_points(frameR)

    final_left = best_left
    final_right = best_right
    fail_reason = None

    print(f"\n[DEBUG] Ghost Protocol Verification:")

    if fresh_left_pts and best_left:
        dists = [np.hypot(p[0] - best_left[0], p[1] - best_left[1]) for p in fresh_left_pts]
        min_dist = min(dists)
        closest_l = fresh_left_pts[np.argmin(dists)]

        if min_dist < 80:
            final_left = closest_l
            print(f"  [SUCCESS] Left snapped to fresh YOLO point ({min_dist:.1f}px).")
        else:
            final_left = best_left
            print("  [WARNING] Left YOLO missed. Trusting burst point.")
    else:
        final_left = best_left
        print("  [WARNING] Left YOLO empty. Trusting burst point.")

    if fresh_right_pts and best_right:
        dists = [np.hypot(p[0] - best_right[0], p[1] - best_right[1]) for p in fresh_right_pts]
        min_dist = min(dists)
        closest_r = fresh_right_pts[np.argmin(dists)]

        if min_dist < 80:
            final_right = closest_r
            print(f"  [SUCCESS] Right snapped to fresh YOLO point ({min_dist:.1f}px).")
        else:
            final_right = best_right
            print("  [WARNING] Right YOLO missed. Trusting burst point.")
    else:
        final_right = best_right
        print("  [WARNING] Right YOLO empty. Trusting burst point.")

    if fail_reason:
        print(f"[FINE ALIGN] Fallback triggered: {fail_reason}")
        return None, None, None, None, None

    return final_left, final_right, frameL, frameR, best_solved


def _pick_initial_target(gantry, cameras, detector, coarse_mover, planned_target, actual_hits):
    if DETECTOR_MODE == "manual":
        return _pick_manual_initial_target(cameras, detector)
    if DETECTOR_MODE == "ai":
        return _pick_best_ai_target(gantry, cameras, detector, coarse_mover, planned_target, actual_hits)
    raise ValueError(f"Unknown DETECTOR_MODE: {DETECTOR_MODE}")


def _show_combined_fine_align_window(frameL, frameR, left_pt, right_pt, err_x, err_y, dx, dy):
    window_name = "Fine Align"

    dispL = frameL.copy()
    dispR = frameR.copy()

    xl, yl = int(round(left_pt[0])), int(round(left_pt[1]))
    xr, yr = int(round(right_pt[0])), int(round(right_pt[1]))

    cv2.circle(dispL, (xl, yl), 5, (0, 0, 255), -1)
    cv2.circle(dispR, (xr, yr), 5, (0, 0, 255), -1)

    cv2.line(dispL, (int(TARGET_X), 0), (int(TARGET_X), FINE_H), (255, 0, 0), 1)
    cv2.line(dispL, (0, int(TARGET_Y)), (FINE_W, int(TARGET_Y)), (255, 0, 0), 1)
    cv2.line(dispR, (int(TARGET_X), 0), (int(TARGET_X), FINE_H), (255, 0, 0), 1)
    cv2.line(dispR, (0, int(TARGET_Y)), (FINE_W, int(TARGET_Y)), (255, 0, 0), 1)

    cv2.putText(dispL, "LEFT CROP", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(dispR, "RIGHT CROP", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    combined = np.hstack([dispL, dispR])
    status = f"FINE ALIGN | ex={err_x:.1f}px ey={err_y:.1f}px | dx={dx:.3f} dy={dy:.3f}"
    cv2.putText(combined, status, (10, FINE_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, combined.shape[1], combined.shape[0])
    cv2.moveWindow(window_name, 80, 80)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.imshow(window_name, combined)


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
):
    left_pt_full, right_pt_full, old_left_full, old_right_full, selected_target = _pick_initial_target(
        gantry,
        cameras,
        detector,
        coarse_mover,
        planned_target,
        actual_hits,
    )

    if left_pt_full is None or right_pt_full is None:
        print(f"\n[!] FINE ALIGN: Detection/Matching failed for target {planned_target.get('id', '?')}.")
        print("[!] ACTION: Skipping strike. Moving to next target.")
        return False, None

    if not (_point_inside_crop(left_pt_full) and _point_inside_crop(right_pt_full)):
        print("[FINE ALIGN FAIL] Target is outside the center crop. Skipping strike.")
        return False, None

    left_pt = _full_to_crop_point(left_pt_full)
    right_pt = _full_to_crop_point(right_pt_full)
    old_left = _crop_frame(old_left_full)
    old_right = _crop_frame(old_right_full)

    track_pt_L = np.array([[left_pt]], dtype=np.float32)
    track_pt_R = np.array([[right_pt]], dtype=np.float32)

    old_gray_L = cv2.cvtColor(old_left, cv2.COLOR_BGR2GRAY)
    old_gray_R = cv2.cvtColor(old_right, cv2.COLOR_BGR2GRAY)

    prev_ex = 0.0
    prev_ey = 0.0
    inside_count = 0
    t0 = time.time()

    # In manual mode, avoid immediately opening another OpenCV debug window
    # right after the manual click UI. That handoff is exactly where Jetson
    # HighGUI tends to be fragile.
    effective_show_debug = show_debug and (DETECTOR_MODE != "manual")

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
            _safe_destroy_fine_align_window()
            print("[FINE ALIGN FAIL] Lost LK tracking.")
            return False, None

        track_pt_L = new_pt_L
        track_pt_R = new_pt_R

        xl, yl = float(track_pt_L[0, 0, 0]), float(track_pt_L[0, 0, 1])
        xr, yr = float(track_pt_R[0, 0, 0]), float(track_pt_R[0, 0, 1])

        if not (0 <= xl < FINE_W and 0 <= yl < FINE_H and 0 <= xr < FINE_W and 0 <= yr < FINE_H):
            gantry.stop()
            end_live_fine_align()
            _safe_destroy_fine_align_window()
            print("[FINE ALIGN FAIL] Tracked point left the center crop.")
            return False, None

        OFFSET_X_PX = -5.0
        OFFSET_Y_PX = -1.0

        err_x, err_y = _compute_errors((xl, yl), (xr, yr))
        err_x += OFFSET_X_PX
        err_y += OFFSET_Y_PX

        dex = err_x - prev_ex
        dey = err_y - prev_ey
        prev_ex = err_x
        prev_ey = err_y

        dx = 0.0
        dy = 0.0
        if abs(err_x) > DEADZONE:
            dx = round((err_x * Kp_x + dex * Kd_x) * STEP_MM, 3)
        if abs(err_y) > DEADZONE:
            dy = round((err_y * Kp_y + dey * Kd_y) * STEP_MM, 3)

        dx = float(np.clip(dx, -MAX_JOG, MAX_JOG))
        dy = float(np.clip(dy, -MAX_JOG, MAX_JOG))

        print_live_fine_align(
            err_x,
            err_y,
            dx,
            dy,
            planned_xy=planned_target["target_xy_mm"],
            throttle_s=0.25,
        )

        if abs(err_x) <= DEADZONE and abs(err_y) <= DEADZONE:
            inside_count += 1
            gantry.stop()

            if inside_count >= settle_frames:
                end_live_fine_align()
                _safe_destroy_fine_align_window()
                print(f"Fine align locked: ex={err_x:.2f}px ey={err_y:.2f}px")

                gantry.sync_estimate_to_machine()
                final_xy = gantry.get_estimated_xy()

                if DETECTOR_MODE == "ai" and selected_target is not None:
                    actual_entry = coarse_mover.append_actual_target(
                        planned_target,
                        selected_target,
                        final_xy,
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
            inside_count = 0
            if dx != 0.0 or dy != 0.0:
                gantry.jog(dx, dy, FINE_FEED)

        if effective_show_debug:
            _show_combined_fine_align_window(frameL, frameR, (xl, yl), (xr, yr), err_x, err_y, dx, dy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("r"):
                gantry.stop()
                end_live_fine_align()
                _safe_destroy_fine_align_window()
                print("[FINE ALIGN FAIL] Cancelled by user.")
                return False, None

        old_gray_L = grayL
        old_gray_R = grayR

    gantry.stop()
    end_live_fine_align()
    _safe_destroy_fine_align_window()
    print("[FINE ALIGN FAIL] Timeout.")
    return False, None