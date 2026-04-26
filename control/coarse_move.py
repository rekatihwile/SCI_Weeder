import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from control.calibration_correction import AffineXYCorrection

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    SURVEY_FRAME_WIDTH,
    SURVEY_FRAME_HEIGHT,
    SURVEY_CONF_SENSITIVITY_DEBUG,
    SURVEY_YOLO_IMGSZ,
    CALIB_NPZ_PATH,
    RECT_NPZ_PATH,
    CALIBRATION_EXPECTS_UNFLIPPED,
    TRI_X_SIGN,
    TRI_Y_SIGN,
    LASER_OFFSET_X_MM,
    LASER_OFFSET_Y_MM,
    TRI_X_GAIN,
    TRI_Y_GAIN,
    SURVEY_CLUSTER_RADIUS_PX,
    SURVEY_CROP_MODE,
    SURVEY_CROP_HALF_X_PX,
    SURVEY_CROP_HALF_Y_PX,
    SURVEY_PROJECT_CROP_MARGIN_PX,
    SURVEY_PROJECT_Z_MIN_MM,
    SURVEY_PROJECT_Z_MAX_MM,
    SURVEY_PROJECT_Z_SAMPLES,
    SURVEY_POS_X,
    SURVEY_POS_Y,
    WORKSPACE_X_MIN,
    WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Y_MAX,
)
from vision.workspace_crop import project_workspace_crop_left_right, crop_frame_with_rect

# ── workspace bounds helpers (exported so main.py can import them) ────────────

_WS_MARGIN_MM = 5.0   # safety margin inside each wall


def is_in_workspace(x, y, margin=_WS_MARGIN_MM):
    """Return True if (x, y) mm is safely inside the workspace."""
    return (
        WORKSPACE_X_MIN + margin <= x <= WORKSPACE_X_MAX - margin
        and WORKSPACE_Y_MIN + margin <= y <= WORKSPACE_Y_MAX - margin
    )


def clamp_to_workspace(x, y, margin=_WS_MARGIN_MM):
    """Clamp (x, y) mm to the safe interior of the workspace."""
    cx = float(np.clip(x, WORKSPACE_X_MIN + margin, WORKSPACE_X_MAX - margin))
    cy = float(np.clip(y, WORKSPACE_Y_MIN + margin, WORKSPACE_Y_MAX - margin))
    return cx, cy


# ── internal helpers ──────────────────────────────────────────────────────────


def _unflip_point_180(u, v, width, height):
    return (width - 1 - u, height - 1 - v)


def _normalize_rectified_calibration_units_to_meters(T, P1, P2):
    T  = np.asarray(T,  dtype=np.float64).reshape(3)
    P1 = np.asarray(P1, dtype=np.float64).copy()
    P2 = np.asarray(P2, dtype=np.float64).copy()

    baseline = float(np.linalg.norm(T))
    if baseline > 1.0:
        scale = 1e-3
        T     *= scale
        P1[:, 3] *= scale
        P2[:, 3] *= scale

    return T, P1, P2


def _triangulate_point_rectified(uL, vL, uR, vR, K1, D1, K2, D2, R1, P1, R2, P2):
    ptsL = np.array([[[uL, vL]]], dtype=np.float64)
    ptsR = np.array([[[uR, vR]]], dtype=np.float64)

    ptsLr = cv2.fisheye.undistortPoints(ptsL, K1, D1, R=R1, P=P1)
    ptsRr = cv2.fisheye.undistortPoints(ptsR, K2, D2, R=R2, P=P2)

    xL, yL = float(ptsLr[0, 0, 0]), float(ptsLr[0, 0, 1])
    xR, yR = float(ptsRr[0, 0, 0]), float(ptsRr[0, 0, 1])

    X_h = cv2.triangulatePoints(
        P1, P2,
        np.array([[xL], [yL]], dtype=np.float64),
        np.array([[xR], [yR]], dtype=np.float64),
    )
    X = (X_h[:3] / X_h[3]).reshape(3)
    return X


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _scale_stable_to_calib(stable_list, sx, sy):
    """Scale burst-stable detections from HD survey space to calibration space."""
    out = []
    for s in stable_list:
        px, py = s["point"]
        x1, y1, x2, y2 = s["box"]
        out.append({
            **s,
            "point": (int(round(px * sx)), int(round(py * sy))),
            "box":   (x1 * sx, y1 * sy, x2 * sx, y2 * sy),
        })
    return out


def _cluster_burst_points(frames_points, radius_px=SURVEY_CLUSTER_RADIUS_PX, min_hits=3):
    clusters = []
    for frame_idx, pts in enumerate(frames_points):
        for pt in pts:
            p = (float(pt[0]), float(pt[1]))
            best_i = None
            best_d = radius_px
            for i, c in enumerate(clusters):
                if frame_idx in c["frames"]:
                    continue
                center = (c["sum_x"] / c["count"], c["sum_y"] / c["count"])
                d = _dist(p, center)
                if d <= best_d:
                    best_d = d
                    best_i = i
            if best_i is None:
                clusters.append({"sum_x": p[0], "sum_y": p[1], "count": 1, "frames": {frame_idx}})
            else:
                c = clusters[best_i]
                c["sum_x"] += p[0]; c["sum_y"] += p[1]
                c["count"] += 1;    c["frames"].add(frame_idx)

    stable = []
    for c in clusters:
        if len(c["frames"]) >= min_hits:
            stable.append((int(round(c["sum_x"] / c["count"])),
                           int(round(c["sum_y"] / c["count"]))))
    return stable


# ── main class ────────────────────────────────────────────────────────────────

class TriangulationCoarseMover:
    def __init__(self):
        from control.pixel_error_model import StereoPixelErrorModel
        from config import USE_PIXEL_ERROR_CORRECTION, PIXEL_ERROR_MODEL_PATH

        self.pixel_err_model = None
        if USE_PIXEL_ERROR_CORRECTION and PIXEL_ERROR_MODEL_PATH.exists():
            self.pixel_err_model = StereoPixelErrorModel(PIXEL_ERROR_MODEL_PATH)

        self.xy_correction = None

        calib = np.load(CALIB_NPZ_PATH)
        rect  = np.load(RECT_NPZ_PATH)

        self.last_survey_frameL = None
        self.last_survey_frameR = None
        self.epipolar_slope = None
        self.epipolar_slope_tol = None
        self.mean_disparity_px = None

        self.K1 = np.asarray(calib["K1"], dtype=np.float64)
        self.D1 = np.asarray(calib["D1"], dtype=np.float64)
        self.K2 = np.asarray(calib["K2"], dtype=np.float64)
        self.D2 = np.asarray(calib["D2"], dtype=np.float64)
        self.T  = np.asarray(calib["T"],  dtype=np.float64).reshape(3)

        self.R1 = np.asarray(rect["R1"], dtype=np.float64)
        self.P1 = np.asarray(rect["P1"], dtype=np.float64)
        self.R2 = np.asarray(rect["R2"], dtype=np.float64)
        self.P2 = np.asarray(rect["P2"], dtype=np.float64)

        self.T, self.P1, self.P2 = _normalize_rectified_calibration_units_to_meters(
            self.T, self.P1, self.P2
        )
        self.T_rect = (self.R1 @ self.T.reshape(3, 1)).reshape(3)

        self.planning_dir = Path(__file__).resolve().parent.parent / "planning"
        self.planning_dir.mkdir(parents=True, exist_ok=True)

    # ── triangulation ────────────────────────────────────────────────────────

    def _solve_geometry(self, target):
        xl, yl = target["left_px"]
        xr, yr = target["right_px"]

        if CALIBRATION_EXPECTS_UNFLIPPED:
            xl, yl = _unflip_point_180(xl, yl, FRAME_WIDTH, FRAME_HEIGHT)
            xr, yr = _unflip_point_180(xr, yr, FRAME_WIDTH, FRAME_HEIGHT)

        X_rect = _triangulate_point_rectified(
            xl, yl, xr, yr,
            self.K1, self.D1, self.K2, self.D2,
            self.R1, self.P1, self.R2, self.P2,
        )
        X_mid    = X_rect - 0.5 * self.T_rect
        offset_m = np.array([LASER_OFFSET_X_MM / 1000.0, LASER_OFFSET_Y_MM / 1000.0, 0.0])
        X_laser  = X_mid - offset_m

        dx_mm = TRI_X_SIGN * TRI_X_GAIN * float(X_laser[0] * 1000.0)
        dy_mm = TRI_Y_SIGN * TRI_Y_GAIN * float(X_laser[1] * 1000.0)
        return X_rect, X_mid, X_laser, dx_mm, dy_mm

    def solve_target_from_pose(self, target, ref_x, ref_y):
        X_rect, X_mid, X_laser, dx_mm, dy_mm = self._solve_geometry(target)
        raw_tx = float(ref_x + dx_mm)
        raw_ty = float(ref_y + dy_mm)

        corr_dx, corr_dy = 0.0, 0.0
        if self.pixel_err_model:
            corr_dx, corr_dy = self.pixel_err_model.predict_error(
                target["left_px"], target["right_px"]
            )

        tx_final = raw_tx + corr_dx
        ty_final = raw_ty + corr_dy

        if self.xy_correction is not None:
            tx_final, ty_final = self.xy_correction.apply(tx_final, ty_final)

        return {
            "source_target":              target,
            "target_xy_mm":               (float(tx_final), float(ty_final)),
            "raw_triangulated_xy_mm":     (raw_tx, raw_ty),
            "pixel_correction_applied_mm": (corr_dx, corr_dy),
        }

    def solve_target_from_survey(self, target, survey_x, survey_y):
        return self.solve_target_from_pose(target, survey_x, survey_y)

    def solve_all_from_pose(self, matched_targets, ref_x, ref_y):
        return [self.solve_target_from_pose(t, ref_x, ref_y) for t in matched_targets]

    def solve_all_from_survey(self, matched_targets, survey_x, survey_y):
        return self.solve_all_from_pose(matched_targets, survey_x, survey_y)

    # ── detection ────────────────────────────────────────────────────────────

    def detect_stable_points(
        self, cameras=None, detector=None, detector_mode=None,
        burst_count=5, min_hits=3, cluster_radius_px=SURVEY_CLUSTER_RADIUS_PX,
        survey_classes=None,  # int | list[int] | None — per-survey class override
        point_mode="box_center",
    ):
        def _survey_debug(msg):
            print(f"[SURVEY DEBUG] {msg}", flush=True)

        self.last_survey_timing = {}

        if detector_mode == "manual" and cameras:
            ptsL, ptsR = detector.detect_live(cameras)
            self.last_survey_frameL = getattr(detector, "last_displayed_left",  None)
            self.last_survey_frameR = getattr(detector, "last_displayed_right", None)
            return ptsL, ptsR

        if detector_mode != "ai":
            raise ValueError(f"Unknown detector mode: {detector_mode}")

        use_hd = (
            SURVEY_FRAME_WIDTH is not None
            and SURVEY_FRAME_HEIGHT is not None
            and (SURVEY_FRAME_WIDTH != FRAME_WIDTH or SURVEY_FRAME_HEIGHT != FRAME_HEIGHT)
        )

        t_survey = time.perf_counter()
        capture_w = SURVEY_FRAME_WIDTH if use_hd else FRAME_WIDTH
        capture_h = SURVEY_FRAME_HEIGHT if use_hd else FRAME_HEIGHT
        _survey_debug(
            f"start burst: frames={burst_count} min_hits={min_hits} "
            f"radius={cluster_radius_px:.1f}px classes={survey_classes} "
            f"capture={capture_w}x{capture_h} point_mode={point_mode}"
        )

        if use_hd:
            t_res = time.perf_counter()
            print(f"[SURVEY] Switching to HD capture: {SURVEY_FRAME_WIDTH}×{SURVEY_FRAME_HEIGHT}")
            cameras.set_resolution(SURVEY_FRAME_WIDTH, SURVEY_FRAME_HEIGHT)
            _survey_debug(f"HD switch took {time.perf_counter() - t_res:.2f}s")

        t_capture = time.perf_counter()
        try:
            left_frames, right_frames = [], []
            attempts = 0
            read_time_total = 0.0
            while len(left_frames) < burst_count and attempts < (burst_count * 3):
                t_read = time.perf_counter()
                fL, fR = cameras.read_pair()
                attempts += 1
                read_dt = time.perf_counter() - t_read
                read_time_total += read_dt
                if fL is None or fR is None:
                    _survey_debug(
                        f"capture attempt {attempts}: missing frame(s) "
                        f"after {read_dt:.2f}s"
                    )
                    time.sleep(0.05)
                    continue
                left_frames.append(fL)
                right_frames.append(fR)
                _survey_debug(
                    f"captured frame {len(left_frames)}/{burst_count} "
                    f"in {read_dt:.2f}s (attempt {attempts})"
                )
        finally:
            if use_hd:
                t_res = time.perf_counter()
                cameras.set_resolution(FRAME_WIDTH, FRAME_HEIGHT)
                print(f"[SURVEY] Restored capture to: {FRAME_WIDTH}×{FRAME_HEIGHT}")
                _survey_debug(f"restore switch took {time.perf_counter() - t_res:.2f}s")

        capture_dt = time.perf_counter() - t_capture
        _survey_debug(
            f"capture done: {len(left_frames)}/{burst_count} frame(s), "
            f"{attempts} attempt(s), {capture_dt:.2f}s"
        )

        if not left_frames:
            _survey_debug(f"no frames captured; total {time.perf_counter() - t_survey:.2f}s")
            return [], []

        # Scale the cluster radius if we captured at HD (groups are physically bigger in HD pixels).
        burst_radius = cluster_radius_px * (SURVEY_FRAME_WIDTH / FRAME_WIDTH) if use_hd else cluster_radius_px

        # Optional survey crop before YOLO.
        # Mode is configurable: projected or centered.
        survey_crop_mode = str(SURVEY_CROP_MODE or "projected").strip().lower()
        if survey_crop_mode not in ("projected", "centered"):
            survey_crop_mode = "projected"

        survey_crop_half_x = SURVEY_CROP_HALF_X_PX
        survey_crop_half_y = SURVEY_CROP_HALF_Y_PX

        def _build_centered_rect():
            if survey_crop_half_x is None or survey_crop_half_y is None:
                return None
            cx, cy = capture_w // 2, capture_h // 2
            x0 = max(0, int(cx - survey_crop_half_x))
            x1 = min(capture_w, int(cx + survey_crop_half_x))
            y0 = max(0, int(cy - survey_crop_half_y))
            y1 = min(capture_h, int(cy + survey_crop_half_y))
            if x1 <= x0 or y1 <= y0:
                return None
            return (x0, y0, x1, y1)

        left_rect = (0, 0, capture_w, capture_h)
        right_rect = (0, 0, capture_w, capture_h)
        projected_crop_ok = False

        if not use_hd and survey_crop_mode == "projected":
            left_rect_proj, right_rect_proj, proj_info = project_workspace_crop_left_right(
                frame_width=capture_w,
                frame_height=capture_h,
                calib_npz_path=CALIB_NPZ_PATH,
                rect_npz_path=RECT_NPZ_PATH,
                workspace_x_min=WORKSPACE_X_MIN,
                workspace_x_max=WORKSPACE_X_MAX,
                workspace_y_min=WORKSPACE_Y_MIN,
                workspace_y_max=WORKSPACE_Y_MAX,
                survey_pos_x=SURVEY_POS_X,
                survey_pos_y=SURVEY_POS_Y,
                tri_sign_x=TRI_X_SIGN,
                tri_sign_y=TRI_Y_SIGN,
                tri_x_gain=TRI_X_GAIN,
                tri_y_gain=TRI_Y_GAIN,
                laser_offset_x_mm=LASER_OFFSET_X_MM,
                laser_offset_y_mm=LASER_OFFSET_Y_MM,
                z_min_mm=SURVEY_PROJECT_Z_MIN_MM,
                z_max_mm=SURVEY_PROJECT_Z_MAX_MM,
                z_samples=SURVEY_PROJECT_Z_SAMPLES,
                margin_px=SURVEY_PROJECT_CROP_MARGIN_PX,
                calibration_expects_unflipped=CALIBRATION_EXPECTS_UNFLIPPED,
            )
            if left_rect_proj is not None and right_rect_proj is not None:
                left_rect = left_rect_proj
                right_rect = right_rect_proj
                projected_crop_ok = True
                print("[SURVEY CROP] mode=projected")
                lx0, ly0, lx1, ly1 = left_rect
                rx0, ry0, rx1, ry1 = right_rect
                print(
                    f"[SURVEY CROP] projected LEFT  x={lx0} y={ly0} "
                    f"size={lx1-lx0}x{ly1-ly0}"
                )
                print(
                    f"[SURVEY CROP] projected RIGHT x={rx0} y={ry0} "
                    f"size={rx1-rx0}x{ry1-ry0}"
                )
            else:
                reason = (proj_info or {}).get("reason", "unknown")
                print(f"[SURVEY CROP] projected fallback -> centered reason={reason}")

        if not use_hd and projected_crop_ok:
            left_frames_yolo = [crop_frame_with_rect(f, left_rect) for f in left_frames]
            right_frames_yolo = [crop_frame_with_rect(f, right_rect) for f in right_frames]
        elif not use_hd:
            centered_rect = _build_centered_rect()
            if centered_rect is not None:
                left_rect = centered_rect
                right_rect = centered_rect
                scx0, scy0, scx1, scy1 = centered_rect
                left_frames_yolo = [crop_frame_with_rect(f, left_rect) for f in left_frames]
                right_frames_yolo = [crop_frame_with_rect(f, right_rect) for f in right_frames]
                print("[SURVEY CROP] mode=centered")
                print(
                    f"[SURVEY CROP] centered LEFT  x={scx0} y={scy0} "
                    f"size={scx1-scx0}x{scy1-scy0}"
                )
                print(
                    f"[SURVEY CROP] centered RIGHT x={scx0} y={scy0} "
                    f"size={scx1-scx0}x{scy1-scy0}"
                )
            else:
                left_frames_yolo = left_frames
                right_frames_yolo = right_frames
                print("[SURVEY CROP] mode=full (centered crop disabled)")
        else:
            left_frames_yolo = left_frames
            right_frames_yolo = right_frames
            if use_hd:
                print("[SURVEY CROP] mode=full (HD survey capture)")
            else:
                print("[SURVEY CROP] mode=full (crop disabled)")

        def _stable_side(core, frames, label):
            t_side = time.perf_counter()
            stable = core.return_burst_stable(
                frames,
                min_stable_views=min_hits,
                group_radius_px=burst_radius,
                classes_override=survey_classes,
                debug_label=label,
                imgsz=SURVEY_YOLO_IMGSZ,
                heatmap_final=(point_mode != "box_center"),
                point_mode=point_mode,
            )
            timing = getattr(core, "last_burst_timing", {})
            return stable, time.perf_counter() - t_side, timing

        t_detect = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="survey-burst") as pool:
            left_future = pool.submit(_stable_side, detector.cv_left, left_frames_yolo, "[SURVEY DEBUG] LEFT")
            right_future = pool.submit(_stable_side, detector.cv_right, right_frames_yolo, "[SURVEY DEBUG] RIGHT")
            stable_left, left_dt, left_timing = left_future.result()
            stable_right, right_dt, right_timing = right_future.result()

        # Translate crop-space points back to full-frame coordinates.
        if not use_hd:
            def _to_full_survey(stable_list, crop_rect):
                x0, y0, _, _ = crop_rect
                out = []
                for s in stable_list:
                    px = s["point"][0] + x0
                    py = s["point"][1] + y0
                    x1 = s["box"][0] + x0
                    y1 = s["box"][1] + y0
                    x2 = s["box"][2] + x0
                    y2 = s["box"][3] + y0
                    out.append({**s, "point": (int(round(px)), int(round(py))), "box": (x1, y1, x2, y2)})
                return out

            stable_left = _to_full_survey(stable_left, left_rect)
            stable_right = _to_full_survey(stable_right, right_rect)

        detect_dt = time.perf_counter() - t_detect
        _survey_debug(
            f"stable detection done: left={left_dt:.2f}s right={right_dt:.2f}s "
            f"total={detect_dt:.2f}s"
        )

        # Confidence sensitivity is useful while tuning, but it costs several
        # extra YOLO passes. Keep the normal survey fast unless explicitly enabled.
        t_conf = time.perf_counter()
        if SURVEY_CONF_SENSITIVITY_DEBUG:
            base_conf = detector.cv_left.conf
            last_L, last_R = left_frames[-1], right_frames[-1]
            print("\n[CONF SENSITIVITY] Single-frame raw detection counts (no IoM, no burst filter):")
            print(f"  {'Conf':>8}  {'Left':>5}  {'Right':>5}  {'Time':>15}")
            for delta_pct in (-10, 0, +10):
                c = base_conf * (1.0 + delta_pct / 100.0)
                t_l = time.perf_counter()
                nL = detector.cv_left.count_at_conf(last_L, c, classes_override=survey_classes)
                left_conf_dt = time.perf_counter() - t_l
                t_r = time.perf_counter()
                nR = detector.cv_right.count_at_conf(last_R, c, classes_override=survey_classes)
                right_conf_dt = time.perf_counter() - t_r
                tag = "  ← current" if delta_pct == 0 else f"  ({delta_pct:+d}%)"
                print(
                    f"  {c:>8.3f}  {nL:>5}  {nR:>5}  "
                    f"L {left_conf_dt:>4.1f}s R {right_conf_dt:>4.1f}s{tag}"
                )
            print(f"  Burst stable: Left={len(stable_left)}, Right={len(stable_right)} "
                  f"(min_hits={min_hits}/{len(left_frames)} frames)\n")
        else:
            print(
                f"[SURVEY] Burst stable: Left={len(stable_left)}, Right={len(stable_right)} "
                f"(min_hits={min_hits}/{len(left_frames)} frames)"
            )
        conf_dt = time.perf_counter() - t_conf
        _survey_debug(f"confidence sensitivity step took {conf_dt:.2f}s")

        t_frame_prep = time.perf_counter()
        self.last_survey_frameL = left_frames[-1]
        self.last_survey_frameR = right_frames[-1]

        
        frame_prep_dt = time.perf_counter() - t_frame_prep
        self.last_survey_timing = {
            "survey_camera_read_time_s": round(read_time_total, 6),
            "survey_capture_time_s": round(capture_dt, 6),
            "survey_yolo_time_s": round(max(
                float(left_timing.get("yolo_time_s", 0.0)),
                float(right_timing.get("yolo_time_s", 0.0)),
            ), 6),
            "survey_grouping_time_s": round(max(
                float(left_timing.get("grouping_time_s", 0.0)),
                float(right_timing.get("grouping_time_s", 0.0)),
            ), 6),
            "survey_qpoint_time_s": round(max(
                float(left_timing.get("qpoint_time_s", 0.0)),
                float(right_timing.get("qpoint_time_s", 0.0)),
            ), 6),
            "survey_detection_wall_time_s": round(detect_dt, 6),
            "survey_frame_prep_time_s": round(frame_prep_dt, 6),
            "survey_point_mode": point_mode,
            "survey_burst_count": int(burst_count),
        }
        _survey_debug(f"debug frame prep took {frame_prep_dt:.2f}s")
        _survey_debug(
            f"survey total {time.perf_counter() - t_survey:.2f}s "
            f"(capture {capture_dt:.2f}s, stable detect {detect_dt:.2f}s, "
            f"confidence {conf_dt:.2f}s, frame prep {frame_prep_dt:.2f}s)"
        )

        return stable_left, stable_right

    # ── planning / logging ────────────────────────────────────────────────────

    def fit_epipolar(self, matched_targets):
        """Fit the stereo epipolar relationship from survey match pairs.
        Stores mean dy/dx slope, tolerance, and mean horizontal disparity
        so re-ID can validate candidate stereo pairs against ground-truth geometry."""
        slopes, disps = [], []
        for t in matched_targets:
            lx, ly = t["left_px"]
            rx, ry = t["right_px"]
            dx = float(rx - lx)
            dy = float(ry - ly)
            if abs(dx) < 5.0:
                continue
            slopes.append(dy / dx)
            disps.append(abs(dx))
        if len(slopes) < 2:
            return
        self.epipolar_slope = float(np.mean(slopes))
        self.epipolar_slope_tol = max(0.05, 2.0 * float(np.std(slopes)))
        self.mean_disparity_px = float(np.mean(disps))
        print(
            f"[EPIPOLAR] slope={self.epipolar_slope:.3f}±{self.epipolar_slope_tol:.3f} "
            f"disp={self.mean_disparity_px:.1f}px from {len(slopes)} pairs"
        )

    def select_best_local_candidate(
        self, solved_local, planned_xy, actual_hits, image_error_fn,
        image_err_w=1.0, world_err_w=0.1,
        max_local_world_err_mm=80.0, duplicate_tol_mm=20.0, duplicate_penalty=1e6,
    ):
        pool = [t for t in solved_local
                if _dist(t["target_xy_mm"], planned_xy) <= max_local_world_err_mm]
        if not pool:
            pool = solved_local

        def cost(t):
            ex, ey  = image_error_fn(t["source_target"]["left_px"],
                                     t["source_target"]["right_px"])
            penalty = (duplicate_penalty
                       if self.is_duplicate_of_actual(t["target_xy_mm"], actual_hits,
                                                      tol_mm=duplicate_tol_mm)
                       else 0.0)
            return (image_err_w * np.hypot(ex, ey)
                    + world_err_w * _dist(t["target_xy_mm"], planned_xy)
                    + penalty)

        return min(pool, key=cost), len(solved_local) - len(pool), bool(pool)

    def save_workspace_targets(self, targets, filename="predicted_workspace_targets.json"):
        save_path = self.planning_dir / filename
        data = [
            {"id": i, "target_xy_mm": [float(t["target_xy_mm"][0]),
                                        float(t["target_xy_mm"][1])]}
            for i, t in enumerate(targets, 1)
        ]
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        return save_path

    def clear_actual_targets_log(self, filename="actual_pd_targets.json"):
        with open(self.planning_dir / filename, "w") as f:
            json.dump([], f)

    def append_actual_target(self, planned_target, selected_target, final_xy,
                             filename="actual_pd_targets.json"):
        save_path = self.planning_dir / filename
        data = json.load(open(save_path)) if save_path.exists() else []
        entry = {
            "planned_xy_mm":  [float(planned_target["target_xy_mm"][0]),
                                float(planned_target["target_xy_mm"][1])],
            "final_xy_mm":    [float(final_xy[0]), float(final_xy[1])],
            "reid_tri_xy_mm": [float(selected_target["target_xy_mm"][0]),
                                float(selected_target["target_xy_mm"][1])],
            "left_px":        list(selected_target["source_target"]["left_px"]),
            "right_px":       list(selected_target["source_target"]["right_px"]),
        }
        data.append(entry)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        return entry

    def is_duplicate_of_actual(self, xy, actual_hits, tol_mm=8.0):
        return any(_dist(xy, hit["final_xy_mm"]) <= tol_mm for hit in actual_hits)

    # ── debug / diagnostics ──────────────────────────────────────────────────

    def triangulation_manifest(self, target, ref_x, ref_y, gantry=None):
        """
        Print and return a full debug breakdown of one triangulation solve.

        Parameters
        ----------
        target  : {"left_px": (x,y), "right_px": (x,y)}
        ref_x/y : gantry reference position used for the solve (mm)
        gantry  : optional Gantry instance — queries actual machine position

        Returns the manifest as a dict so callers can log it further.
        """
        sep = "=" * 52
        print(sep)
        print("  TRIANGULATION MANIFEST")
        print(sep)

        xl, yl = target["left_px"]
        xr, yr = target["right_px"]
        print(f"  Pixel coords")
        print(f"    Left  px : ({xl}, {yl})")
        print(f"    Right px : ({xr}, {yr})")

        try:
            X_rect, X_mid, X_laser, dx_mm, dy_mm = self._solve_geometry(target)
        except Exception as e:
            print(f"  ERROR in _solve_geometry: {e}")
            print(sep)
            return {}

        print(f"\n  Camera-frame geometry  (after fisheye undistort + triangulate)")
        print(f"    X_rect (m) : {X_rect}")
        print(f"    X_mid  (m) : {X_mid}")
        print(f"    X_laser(m) : {X_laser}")
        print(f"    dx offset  : {dx_mm:+.2f} mm   (applied to ref_x)")
        print(f"    dy offset  : {dy_mm:+.2f} mm   (applied to ref_y)")

        raw_tx = float(ref_x + dx_mm)
        raw_ty = float(ref_y + dy_mm)
        corr_dx, corr_dy = 0.0, 0.0
        if self.pixel_err_model:
            corr_dx, corr_dy = self.pixel_err_model.predict_error(
                target["left_px"], target["right_px"]
            )
        tx_final = raw_tx + corr_dx
        ty_final = raw_ty + corr_dy
        if self.xy_correction is not None:
            tx_final, ty_final = self.xy_correction.apply(tx_final, ty_final)

        print(f"\n  Reference position (what ref_x/y was passed in)")
        print(f"    ref_x : {ref_x:.2f} mm")
        print(f"    ref_y : {ref_y:.2f} mm")

        print(f"\n  Solve result")
        print(f"    raw target  : ({raw_tx:.2f}, {raw_ty:.2f}) mm")
        print(f"    pixel corr  : ({corr_dx:+.3f}, {corr_dy:+.3f}) mm")
        print(f"    final target: ({tx_final:.2f}, {ty_final:.2f}) mm")
        print(f"    workspace   : {'PASS' if is_in_workspace(tx_final, ty_final) else 'FAIL — outside bounds'}")

        if gantry is not None:
            pos = gantry.get_position()
            if pos:
                print(f"\n  Gantry position (live query)")
                print(f"    machine pos : ({pos['x']:.2f}, {pos['y']:.2f}) mm")
                drift_x = pos["x"] - ref_x
                drift_y = pos["y"] - ref_y
                if abs(drift_x) > 1.0 or abs(drift_y) > 1.0:
                    print(f"    WARNING: ref vs machine drift = ({drift_x:+.2f}, {drift_y:+.2f}) mm")
                    print(f"    This means ref_x/y was stale — target will be wrong by that amount.")
            else:
                print(f"\n  Gantry position : could not query (check serial)")

        print(sep)

        return {
            "left_px":         (xl, yl),
            "right_px":        (xr, yr),
            "X_rect_m":        X_rect.tolist(),
            "dx_mm":           dx_mm,
            "dy_mm":           dy_mm,
            "ref_x":           ref_x,
            "ref_y":           ref_y,
            "raw_target_mm":   (raw_tx, raw_ty),
            "pixel_corr_mm":   (corr_dx, corr_dy),
            "final_target_mm": (tx_final, ty_final),
            "in_workspace":    is_in_workspace(tx_final, ty_final),
        }

    # ── motion ────────────────────────────────────────────────────────────────

    def move_to_absolute_target(self, gantry, solved_target, feed=12000):
        tx, ty = solved_target["target_xy_mm"]

        if not is_in_workspace(tx, ty):
            print(f"[WARN] Target ({tx:.1f}, {ty:.1f}) mm is outside workspace bounds "
                  f"[{WORKSPACE_X_MIN}..{WORKSPACE_X_MAX}, "
                  f"{WORKSPACE_Y_MIN}..{WORKSPACE_Y_MAX}]. Skipping move.")
            return False

        gantry.move_absolute(tx, ty, feed=feed)
        return True
