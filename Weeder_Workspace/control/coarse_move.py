import json
from pathlib import Path

import cv2
import numpy as np
from control.calibration_correction import AffineXYCorrection
from config import USE_AFFINE_CORRECTION, AFFINE_X_COEFFS, AFFINE_Y_COEFFS

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
)


def _unflip_point_180(u, v, width, height):
    return (width - 1 - u, height - 1 - v)


def _triangulate_point_rectified(uL, vL, uR, vR, K1, D1, K2, D2, R1, P1, R2, P2):
    ptsL = np.array([[[uL, vL]]], dtype=np.float64)
    ptsR = np.array([[[uR, vR]]], dtype=np.float64)

    ptsLr = cv2.fisheye.undistortPoints(ptsL, K1, D1, R=R1, P=P1)
    ptsRr = cv2.fisheye.undistortPoints(ptsR, K2, D2, R=R2, P=P2)

    xL, yL = float(ptsLr[0, 0, 0]), float(ptsLr[0, 0, 1])
    xR, yR = float(ptsRr[0, 0, 0]), float(ptsRr[0, 0, 1])

    X_h = cv2.triangulatePoints(
        P1,
        P2,
        np.array([[xL], [yL]], dtype=np.float64),
        np.array([[xR], [yR]], dtype=np.float64),
    )

    X = (X_h[:3] / X_h[3]).reshape(3)
    return X


def _dist(a, b):
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
                center = (c["sum_x"] / c["count"], c["sum_y"] / c["count"])
                d = _dist(p, center)
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


class TriangulationCoarseMover:
    def __init__(self):
        self.xy_correction = None
        if USE_AFFINE_CORRECTION:
            self.xy_correction = AffineXYCorrection(AFFINE_X_COEFFS, AFFINE_Y_COEFFS)

        calib = np.load(CALIB_NPZ_PATH)
        rect = np.load(RECT_NPZ_PATH)

        self.K1, self.D1 = calib["K1"], calib["D1"]
        self.K2, self.D2 = calib["K2"], calib["D2"]
        self.T = calib["T"].reshape(3)

        self.R1, self.P1 = rect["R1"], rect["P1"]
        self.R2, self.P2 = rect["R2"], rect["P2"]

        self.T_rect = (self.R1 @ self.T.reshape(3, 1)).reshape(3)

        self.planning_dir = Path(__file__).resolve().parent.parent / "planning"
        self.planning_dir.mkdir(parents=True, exist_ok=True)

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

        X_mid = X_rect - 0.5 * self.T_rect

        offset_m = np.array([
            LASER_OFFSET_X_MM / 1000.0,
            LASER_OFFSET_Y_MM / 1000.0,
            0.0,
        ], dtype=float)

        X_laser = X_mid - offset_m

        dx_mm = TRI_SIGN_X * TRI_X_GAIN * float(X_laser[0] * 1000.0)
        dy_mm = TRI_SIGN_Y * TRI_Y_GAIN * float(X_laser[1] * 1000.0)

        return X_rect, X_mid, X_laser, dx_mm, dy_mm

    def solve_target_from_pose(self, target, ref_x, ref_y):
        X_rect, X_mid, X_laser, dx_mm, dy_mm = self._solve_geometry(target)

        tx_raw = float(ref_x + dx_mm)
        ty_raw = float(ref_y + dy_mm)

        if self.xy_correction is not None:
            tx, ty = self.xy_correction.apply(tx_raw, ty_raw)
        else:
            tx, ty = tx_raw, ty_raw

        return {
            "source_target": target,
            "X_rect_m": X_rect,
            "X_mid_m": X_mid,
            "X_laser_m": X_laser,
            "delta_xy_mm": (dx_mm, dy_mm),
            "target_xy_mm": (float(tx), float(ty)),
        }

    def solve_target_from_survey(self, target, survey_x, survey_y):
        return self.solve_target_from_pose(target, survey_x, survey_y)

    def solve_all_from_pose(self, matched_targets, ref_x, ref_y):
        solved = []
        for target in matched_targets:
            solved.append(
                self.solve_target_from_pose(
                    target,
                    ref_x=ref_x,
                    ref_y=ref_y,
                )
            )
        return solved

    def solve_all_from_survey(self, matched_targets, survey_x, survey_y):
        return self.solve_all_from_pose(matched_targets, survey_x, survey_y)

    def detect_stable_points(
        self,
        cameras,
        detector,
        detector_mode,
        burst_count=5,
        min_hits=3,
        cluster_radius_px=12.0,
    ):
        if detector_mode == "manual":
            return detector.detect_live(cameras)

        if detector_mode != "ai":
            raise ValueError(f"Unknown detector mode: {detector_mode}")

        print(f"\n=== AI BURST SURVEY ({burst_count} frames) ===")

        left_frames = []
        right_frames = []

        for _ in range(burst_count):
            frameL, frameR = cameras.read_pair()
            left_frames.append(detector.cv_left.detect_points(frameL))
            right_frames.append(detector.cv_right.detect_points(frameR))

        stable_left = _cluster_burst_points(
            left_frames,
            radius_px=cluster_radius_px,
            min_hits=min_hits,
        )
        stable_right = _cluster_burst_points(
            right_frames,
            radius_px=cluster_radius_px,
            min_hits=min_hits,
        )

        print(f"Stable left points : {len(stable_left)}")
        print(f"Stable right points: {len(stable_right)}")

        return stable_left, stable_right

    def select_best_local_candidate(
        self,
        solved_local,
        planned_xy,
        actual_hits,
        image_error_fn,
        image_err_w=1.0,
        world_err_w=10.0,
        max_local_world_err_mm=25.0,
        duplicate_tol_mm=20.0,
        duplicate_penalty=1e6,
    ):
        valid = []
        rejected_far = 0

        for t in solved_local:
            world_err = _dist(t["target_xy_mm"], planned_xy)
            if world_err <= max_local_world_err_mm:
                valid.append(t)
            else:
                rejected_far += 1

        pool = valid if valid else solved_local

        def cost(t):
            ex, ey = image_error_fn(
                t["source_target"]["left_px"],
                t["source_target"]["right_px"],
            )
            image_err = float(np.hypot(ex, ey))
            world_err = _dist(t["target_xy_mm"], planned_xy)

            penalty = 0.0
            if self.is_duplicate_of_actual(
                t["target_xy_mm"],
                actual_hits,
                tol_mm=duplicate_tol_mm,
            ):
                penalty += duplicate_penalty

            return image_err_w * image_err + world_err_w * world_err + penalty

        best = min(pool, key=cost)
        return best, rejected_far, bool(valid)

    def save_workspace_targets(self, targets, filename="predicted_workspace_targets.json"):
        save_path = self.planning_dir / filename

        data = []
        for i, t in enumerate(targets, start=1):
            tx, ty = t["target_xy_mm"]

            row = {
                "id": i,
                "target_xy_mm": [float(tx), float(ty)],
            }

            if "source_target" in t:
                row["left_px"] = list(t["source_target"]["left_px"])
                row["right_px"] = list(t["source_target"]["right_px"])
                row["score"] = float(t["source_target"].get("score", 0.0))

            data.append(row)

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved workspace targets -> {save_path}")

    def clear_actual_targets_log(self, filename="actual_pd_targets.json"):
        save_path = self.planning_dir / filename
        with open(save_path, "w") as f:
            json.dump([], f, indent=2)

    def append_actual_target(
        self,
        planned_target,
        selected_target,
        final_xy,
        filename="actual_pd_targets.json",
    ):
        save_path = self.planning_dir / filename

        if save_path.exists():
            with open(save_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        planned_x, planned_y = planned_target["target_xy_mm"]
        selected_x, selected_y = selected_target["target_xy_mm"]
        final_x, final_y = final_xy

        entry = {
            "planned_xy_mm": [float(planned_x), float(planned_y)],
            "selected_local_xy_mm": [float(selected_x), float(selected_y)],
            "final_xy_mm": [float(final_x), float(final_y)],
            "left_px": list(selected_target["source_target"]["left_px"]),
            "right_px": list(selected_target["source_target"]["right_px"]),
            "score": float(selected_target["source_target"].get("score", 0.0)),
        }

        data.append(entry)

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved actual PD lock -> {save_path}")
        return entry

    def is_duplicate_of_actual(self, xy_mm, actual_hits, tol_mm=8.0):
        x, y = xy_mm

        for hit in actual_hits:
            hx, hy = hit["final_xy_mm"]
            if _dist((x, y), (hx, hy)) <= tol_mm:
                return True

        return False

    def move_to_absolute_target(self, gantry, solved_target):
        tx, ty = solved_target["target_xy_mm"]
        print(f"Move target (mm): X={tx:.2f}, Y={ty:.2f}")
        gantry.move_absolute(tx, ty)
        return solved_target