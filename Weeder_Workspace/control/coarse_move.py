import json
from pathlib import Path
import time
import cv2
import numpy as np
from control.calibration_correction import AffineXYCorrection

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CALIB_NPZ_PATH,
    RECT_NPZ_PATH,
    CALIBRATION_EXPECTS_UNFLIPPED,
    TRI_SIGN_X,
    TRI_SIGN_Y,
    LASER_OFFSET_X_MM,
    LASER_OFFSET_Y_MM,
    TRI_X_GAIN,
    TRI_Y_GAIN,
    SURVEY_CLUSTER_RADIUS_PX,
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    SURVEY_BOX_IOU_THRESH,
)

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


# ── workspace bounds helpers ──────────────────────────────────────────────────
_WS_MARGIN_MM = 5.0

def is_in_workspace(x, y, margin=_WS_MARGIN_MM):
    """Return True if (x, y) mm is safely inside the workspace."""
    return (WORKSPACE_X_MIN + margin <= x <= WORKSPACE_X_MAX - margin
            and WORKSPACE_Y_MIN + margin <= y <= WORKSPACE_Y_MAX - margin)

def clamp_to_workspace(x, y, margin=_WS_MARGIN_MM):
    """Clamp (x, y) to the safe interior of the workspace."""
    import numpy as np
    return (float(np.clip(x, WORKSPACE_X_MIN + margin, WORKSPACE_X_MAX - margin)),
            float(np.clip(y, WORKSPACE_Y_MIN + margin, WORKSPACE_Y_MAX - margin)))



# ── workspace bounds helpers ──────────────────────────────────────────────────
_WS_MARGIN_MM = 5.0

def is_in_workspace(x, y, margin=_WS_MARGIN_MM):
    return (WORKSPACE_X_MIN + margin <= x <= WORKSPACE_X_MAX - margin
            and WORKSPACE_Y_MIN + margin <= y <= WORKSPACE_Y_MAX - margin)

def clamp_to_workspace(x, y, margin=_WS_MARGIN_MM):
    import numpy as np
    return (float(np.clip(x, WORKSPACE_X_MIN+margin, WORKSPACE_X_MAX-margin)),
            float(np.clip(y, WORKSPACE_Y_MIN+margin, WORKSPACE_Y_MAX-margin)))


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

        dx_mm = TRI_SIGN_X * TRI_X_GAIN * float(X_laser[0] * 1000.0)
        dy_mm = TRI_SIGN_Y * TRI_Y_GAIN * float(X_laser[1] * 1000.0)
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
        survey_classes=None,    # int | list[int] | None — per-survey class override
        survey_iou_thresh=SURVEY_BOX_IOU_THRESH,  # box IoU grouping threshold
    ):
        if detector_mode == "manual" and cameras:
            ptsL, ptsR = detector.detect_live(cameras)
            self.last_survey_frameL = getattr(detector, "last_displayed_left",  None)
            self.last_survey_frameR = getattr(detector, "last_displayed_right", None)
            return ptsL, ptsR

        if detector_mode != "ai":
            raise ValueError(f"Unknown detector mode: {detector_mode}")

        left_frames, right_frames = [], []
        attempts = 0
        while len(left_frames) < burst_count and attempts < (burst_count * 3):
            fL, fR = cameras.read_pair()
            attempts += 1
            if fL is None or fR is None:
                time.sleep(0.05)
                continue
            left_frames.append(fL)
            right_frames.append(fR)

        if not left_frames:
            return [], []

        self.last_survey_frameL = left_frames[-1]
        self.last_survey_frameR = right_frames[-1]

        stable_left  = detector.cv_left.return_burst_stable(
            left_frames, min_stable_views=min_hits,
            group_iou_thresh=survey_iou_thresh,
            group_radius_px=cluster_radius_px, classes_override=survey_classes)
        stable_right = detector.cv_right.return_burst_stable(
            right_frames, min_stable_views=min_hits,
            group_iou_thresh=survey_iou_thresh,
            group_radius_px=cluster_radius_px, classes_override=survey_classes)
        return stable_left, stable_right

    # ── planning / logging ────────────────────────────────────────────────────

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
            "planned_xy_mm": [float(planned_target["target_xy_mm"][0]),
                               float(planned_target["target_xy_mm"][1])],
            "final_xy_mm":   [float(final_xy[0]), float(final_xy[1])],
            "left_px":       list(selected_target["source_target"]["left_px"]),
            "right_px":      list(selected_target["source_target"]["right_px"]),
        }
        data.append(entry)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        return entry

    def is_duplicate_of_actual(self, xy, actual_hits, tol_mm=8.0):
        return any(_dist(xy, hit["final_xy_mm"]) <= tol_mm for hit in actual_hits)

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