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

    def solve_target_from_survey(self, target, survey_x, survey_y):
        X_rect, X_mid, X_laser, dx_mm, dy_mm = self._solve_geometry(target)

        tx_raw = float(survey_x + dx_mm)
        ty_raw = float(survey_y + dy_mm)

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
            "target_xy_mm": (tx, ty),
        }

    def move_to_absolute_target(self, gantry, solved_target):
        tx, ty = solved_target["target_xy_mm"]
        print(f"Move target (mm): X={tx:.2f}, Y={ty:.2f}")
        gantry.move_absolute(tx, ty)
        return solved_target
