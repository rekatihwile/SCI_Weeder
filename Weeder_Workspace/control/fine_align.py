import time
import cv2
import numpy as np
from ui.terminal import print_live_fine_align, end_live_fine_align
from config import DETECTOR_MODE
from vision.matching import match_points

W = 640
H = 480
TARGET_Y_L = 240
TARGET_Y_R = 240

LK_PARAMS = dict(
    winSize=(31, 31),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

Kp_x = 10.0
Kd_x = 2.5
Kp_y = 10.0
Kd_y = 2.5

STEP_MM = 0.001
DEADZONE = 4
MAX_JOG = 10.0
FINE_FEED = 5000

BURST_COUNT = 5
MIN_HITS = 3
CLUSTER_RADIUS_PX = 12.0

IMAGE_ERR_W = 1.0
WORLD_ERR_W = 10.0
MAX_LOCAL_WORLD_ERR_MM = 35.0

DUPLICATE_TOL_MM = 40.0
SECOND_CHOICE_TOL_MM = 30.0


def _compute_errors(left_pt, right_pt):
    xl, yl = left_pt
    xr, yr = right_pt

    err_x = (xl - W + xr)
    err_y = -((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2.0
    return err_x, err_y


def _norm_xy(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _cluster_burst_points(frames_points, radius_px=12.0, min_hits=3):
    clusters = []

    for frame_idx, pts in enumerate(frames_points):
        for pt in pts:
            p = (float(pt[0]), float(pt[1]))
            best_i = None
            best_d = radius_px

            for i, c in enumerate(clusters):
                if frame_idx in c["frames"]:
                    continue
                cx = c["sum_x"] / c["count"]
                cy = c["sum_y"] / c["count"]
                d = _norm_xy(p, (cx, cy))
                if d <= best_d:
                    best_d = d
                    best_i = i

            if best_i is None:
                clusters.append({
                    "sum_x": p[0],
                    "sum_y": p[1],
                    "count": 1,
                    "frames": {frame_idx},
                })
            else:
                c = clusters[best_i]
                c["sum_x"] += p[0]
                c["sum_y"] += p[1]
                c["count"] += 1
                c["frames"].add(frame_idx)

    stable = []
    for c in clusters:
        if len(c["frames"]) >= min_hits:
            cx = c["sum_x"] / c["count"]
            cy = c["sum_y"] / c["count"]
            stable.append((int(round(cx)), int(round(cy))))

    return stable


def _burst_match_ai_points(cameras, detector):
    left_frames = []
    right_frames = []
    last_frameL = None
    last_frameR = None

    for _ in range(BURST_COUNT):
        frameL, frameR = cameras.read_pair()
        last_frameL = frameL
        last_frameR = frameR
        left_frames.append(detector.cv_left.detect_points(frameL))
        right_frames.append(detector.cv_right.detect_points(frameR))

    stable_left = _cluster_burst_points(
        left_frames,
        radius_px=CLUSTER_RADIUS_PX,
        min_hits=MIN_HITS,
    )
    stable_right = _cluster_burst_points(
        right_frames,
        radius_px=CLUSTER_RADIUS_PX,
        min_hits=MIN_HITS,
    )

    if not stable_left or not stable_right:
        return [], last_frameL, last_frameR

    matched_targets, _, _ = match_points(
        stable_left,
        stable_right,
        verbose=False,
    )

    Y_TOL_PX = 20
    MIN_DISPARITY_PX = 40
    MAX_DISPARITY_PX = 220

    filtered = []
    for t in matched_targets:
        xl, yl = t["left_px"]
        xr, yr = t["right_px"]

        y_err = abs(yl - yr)
        disp = abs(xl - xr)

        if y_err > Y_TOL_PX:
            continue
        if disp < MIN_DISPARITY_PX or disp > MAX_DISPARITY_PX:
            continue

        filtered.append(t)

    return filtered, last_frameL, last_frameR


def _pick_manual_initial_target(cameras, detector):
    left_pt, right_pt = detector.refine_live(cameras)
    if left_pt is None or right_pt is None:
        return None, None, None, None, None

    frameL, frameR = cameras.read_pair()
    return left_pt, right_pt, frameL, frameR, None


def _candidate_cost(t, planned_xy):
    left_pt = t["source_target"]["left_px"]
    right_pt = t["source_target"]["right_px"]

    ex, ey = _compute_errors(left_pt, right_pt)
    image_err = float(np.hypot(ex, ey))
    world_err = _norm_xy(t["target_xy_mm"], planned_xy)

    return IMAGE_ERR_W * image_err + WORLD_ERR_W * world_err


def _is_near_previous_target(candidate_xy, actual_hits, tol_mm):
    for hit in actual_hits:
        hx, hy = hit["final_xy_mm"]
        if _norm_xy(candidate_xy, (hx, hy)) <= tol_mm:
            return True
    return False


def _pick_best_ai_target(gantry, cameras, detector, coarse_mover, planned_target, actual_hits):
    matched_targets, frameL, frameR = _burst_match_ai_points(cameras, detector)

    if not matched_targets or frameL is None or frameR is None:
        return None, None, None, None, None

    gantry.sync_estimate_to_machine()
    current_x, current_y = gantry.get_estimated_xy()

    solved_local = coarse_mover.solve_all_from_pose(
        matched_targets,
        ref_x=current_x,
        ref_y=current_y,
    )

    planned_xy = planned_target["target_xy_mm"]

    valid_candidates = []
    rejected_far = 0

    for t in solved_local:
        world_err = _norm_xy(t["target_xy_mm"], planned_xy)
        if world_err <= MAX_LOCAL_WORLD_ERR_MM:
            valid_candidates.append(t)
        else:
            rejected_far += 1

    if valid_candidates:
        solved_local = valid_candidates
    else:
        print(
            f"AI fine target warning: all local candidates were farther than "
            f"{MAX_LOCAL_WORLD_ERR_MM:.1f} mm from planned target; falling back to all candidates."
        )

    ranked = sorted(solved_local, key=lambda t: _candidate_cost(t, planned_xy))

    best = None
    used_second_choice = False

    for cand in ranked:
        if not _is_near_previous_target(
            cand["target_xy_mm"],
            actual_hits,
            tol_mm=SECOND_CHOICE_TOL_MM,
        ):
            best = cand
            break

    if best is None:
        best = ranked[0]
    elif ranked and best is not ranked[0]:
        used_second_choice = True

    left_pt = best["source_target"]["left_px"]
    right_pt = best["source_target"]["right_px"]

    ex0, ey0 = _compute_errors(left_pt, right_pt)
    world_err0 = _norm_xy(best["target_xy_mm"], planned_xy)

    print(
        "AI fine target selected: "
        f"L={left_pt} R={right_pt} | "
        f"ex={ex0:.2f}px ey={ey0:.2f}px | "
        f"world_err={world_err0:.2f} mm | "
        f"rejected_far={rejected_far} | "
        f"second_choice={used_second_choice}"
    )

    return left_pt, right_pt, frameL, frameR, best


def _pick_initial_target(gantry, cameras, detector, coarse_mover, planned_target, actual_hits):
    if DETECTOR_MODE == "manual":
        return _pick_manual_initial_target(cameras, detector)

    if DETECTOR_MODE == "ai":
        return _pick_best_ai_target(
            gantry,
            cameras,
            detector,
            coarse_mover,
            planned_target,
            actual_hits,
        )

    raise ValueError(f"Unknown DETECTOR_MODE: {DETECTOR_MODE}")


def fine_align_target(
    gantry,
    cameras,
    detector,
    coarse_mover,
    planned_target,
    actual_hits,
    max_time=6.0,
    settle_frames=3,
    show_debug=True,
):
    left_pt, right_pt, old_left, old_right, selected_target = _pick_initial_target(
        gantry,
        cameras,
        detector,
        coarse_mover,
        planned_target,
        actual_hits,
    )

    if left_pt is None or right_pt is None:
        print("Fine align cancelled or no target found.")
        end_live_fine_align()
        return False, None

    track_pt_L = np.array([[left_pt]], dtype=np.float32)
    track_pt_R = np.array([[right_pt]], dtype=np.float32)

    old_gray_L = cv2.cvtColor(old_left, cv2.COLOR_BGR2GRAY)
    old_gray_R = cv2.cvtColor(old_right, cv2.COLOR_BGR2GRAY)

    prev_ex = 0.0
    prev_ey = 0.0
    inside_count = 0
    t0 = time.time()

    while time.time() - t0 < max_time:
        frameL, frameR = cameras.read_pair()
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        new_pt_L, stL, _ = cv2.calcOpticalFlowPyrLK(
            old_gray_L,
            grayL,
            track_pt_L,
            None,
            **LK_PARAMS,
        )
        new_pt_R, stR, _ = cv2.calcOpticalFlowPyrLK(
            old_gray_R,
            grayR,
            track_pt_R,
            None,
            **LK_PARAMS,
        )

        if stL is None or stR is None or stL[0][0] == 0 or stR[0][0] == 0:
            gantry.stop()
            end_live_fine_align()
            print("Fine align lost LK tracking.")
            return False, None

        track_pt_L = new_pt_L
        track_pt_R = new_pt_R

        xl, yl = float(track_pt_L[0, 0, 0]), float(track_pt_L[0, 0, 1])
        xr, yr = float(track_pt_R[0, 0, 0]), float(track_pt_R[0, 0, 1])

        err_x, err_y = _compute_errors((xl, yl), (xr, yr))

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
                        "left_px": [float(left_pt[0]), float(left_pt[1])],
                        "right_px": [float(right_pt[0]), float(right_pt[1])],
                        "score": 1.0,
                    }
                    coarse_mover.append_actual_target(
                        planned_target,
                        planned_target,
                        final_xy,
                        filename="actual_pd_targets.json",
                    )

                return True, actual_entry
        else:
            inside_count = 0
            if dx != 0.0 or dy != 0.0:
                gantry.jog(dx, dy, FINE_FEED)

        if show_debug:
            dispL = frameL.copy()
            dispR = frameR.copy()

            cv2.circle(dispL, (int(xl), int(yl)), 5, (0, 0, 255), -1)
            cv2.circle(dispR, (int(xr), int(yr)), 5, (0, 0, 255), -1)

            cv2.line(dispL, (W // 2, 0), (W // 2, H), (255, 0, 0), 1)
            cv2.line(dispL, (0, TARGET_Y_L), (W, TARGET_Y_L), (255, 0, 0), 1)
            cv2.line(dispR, (W // 2, 0), (W // 2, H), (255, 0, 0), 1)
            cv2.line(dispR, (0, TARGET_Y_R), (W, TARGET_Y_R), (255, 0, 0), 1)

            status = f"FINE ALIGN | ex={err_x:.1f}px ey={err_y:.1f}px | dx={dx:.3f} dy={dy:.3f}"
            cv2.putText(
                dispL,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Fine Align - Left", dispL)
            cv2.imshow("Fine Align - Right", dispR)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("r"):
                gantry.stop()
                end_live_fine_align()
                print("Fine align cancelled.")
                return False, None

        old_gray_L = grayL
        old_gray_R = grayR

    gantry.stop()
    end_live_fine_align()
    print("Fine align timeout.")
    return False, None