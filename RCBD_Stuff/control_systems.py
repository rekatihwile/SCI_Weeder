import numpy as np
import cv2
import time


def _triangulate_point_rectified(uL, vL, uR, vR, K1, D1, K2, D2, R1, P1, R2, P2):
    ptsL = np.array([[[uL, vL]]], dtype=np.float32)
    ptsR = np.array([[[uR, vR]]], dtype=np.float32)

    ptsLr = cv2.undistortPoints(ptsL, K1, D1, R=R1, P=P1)
    ptsRr = cv2.undistortPoints(ptsR, K2, D2, R=R2, P=P2)

    xL, yL = float(ptsLr[0, 0, 0]), float(ptsLr[0, 0, 1])
    xR, yR = float(ptsRr[0, 0, 0]), float(ptsRr[0, 0, 1])

    X_h = cv2.triangulatePoints(
        P1, P2,
        np.array([[xL], [yL]], dtype=np.float32),
        np.array([[xR], [yR]], dtype=np.float32),
    )
    X = (X_h[:3] / X_h[3]).reshape(3)
    return X


class BaseControlSystem:
    name = "BASE"
    def reset(self):
        pass
    def update(self, xl, yl, xr, yr, laser):
        raise NotImplementedError


class PixelPDControl(BaseControlSystem):
    name = "PIXEL_PD"

    def __init__(self, w=640, target_y_l=240, target_y_r=240):
        self.W = w
        self.target_y_l = target_y_l
        self.target_y_r = target_y_r
        self.Kp_x, self.Kd_x = 5.0, 1.0
        self.Kp_y, self.Kd_y = 5.0, 1.0
        self.STEP_MM = 0.001
        self.DEADZONE = 4
        self.MAX_JOG = 10.0
        self.FEED = 5000
        self.prev_ex = 0.0
        self.prev_ey = 0.0

    def reset(self):
        self.prev_ex = 0.0
        self.prev_ey = 0.0

    def update(self, xl, yl, xr, yr, laser):
        err_x = (xl - self.W + xr)
        err_y = -(((yl - self.target_y_l) + (yr - self.target_y_r)) / 2.0)

        dex = err_x - self.prev_ex
        dey = err_y - self.prev_ey
        self.prev_ex, self.prev_ey = err_x, err_y

        dx = round((err_x * self.Kp_x + dex * self.Kd_x) * self.STEP_MM, 3) if abs(err_x) > self.DEADZONE else 0.0
        dy = round((err_y * self.Kp_y + dey * self.Kd_y) * self.STEP_MM, 3) if abs(err_y) > self.DEADZONE else 0.0

        if abs(err_x) <= self.DEADZONE and abs(err_y) <= self.DEADZONE:
            laser.stop()
            return True

        if abs(dx) > 0.0 or abs(dy) > 0.0:
            dx = float(np.clip(dx, -self.MAX_JOG, self.MAX_JOG))
            dy = float(np.clip(dy, -self.MAX_JOG, self.MAX_JOG))
            laser.jog(dx, dy, self.FEED)

        return False


class TriangulateAbsoluteControl(BaseControlSystem):
    name = "TRI_ABS"

    def __init__(self, calib_npz_path, rect_npz_path, feed=12000):
        calib = np.load(calib_npz_path)
        rect  = np.load(rect_npz_path)

        self.K1, self.D1 = calib["K1"], calib["D1"]
        self.K2, self.D2 = calib["K2"], calib["D2"]
        self.T           = calib["T"].reshape(3)   # meters (left->right) in LEFT frame

        self.R1, self.P1 = rect["R1"], rect["P1"]
        self.R2, self.P2 = rect["R2"], rect["P2"]

        self.T_half = 0.5 * self.T  # meters

        # Signs (you already tuned these)
        self.SIGN_X = +1.0
        self.SIGN_Y = -1.0

        # ---- NEW: laser axis offset from camera-midpoint (mm) ----
        # If you overshoot to the right, this is almost certainly nonzero.
        # Tune LASER_OFFSET_X_MM first.
        self.LASER_OFFSET_X_MM = 0.0
        self.LASER_OFFSET_Y_MM = 0.0

        # ---- NEW: optional scale correction on X (dimensionless) ----
        # If X is proportionally too large at all distances, tune this.
        self.X_GAIN = 1.0
        self.Y_GAIN = 1.0

        # Keep Z unconstrained but printed
        self.MIN_Z_M = -2.0
        self.MAX_Z_M =  2.0

        self.TOL_MM = 0.30
        self.feed = int(feed)

        self._sent = False
        self._target_x = None
        self._target_y = None
        self._last_print = 0.0

    def reset(self):
        self._sent = False
        self._target_x = None
        self._target_y = None
        self._last_print = 0.0

    def _print_throttled(self, msg, period_s=0.25):
        t = time.time()
        if t - self._last_print >= period_s:
            print(msg)
            self._last_print = t

    def update(self, xl, yl, xr, yr, laser):
        if not self._sent:
            # 1) Triangulate in rectified-left frame (meters)
            X_left = _triangulate_point_rectified(
                xl, yl, xr, yr,
                self.K1, self.D1, self.K2, self.D2,
                self.R1, self.P1, self.R2, self.P2
            )

            # 2) Convert left-cam frame -> camera midpoint frame
            X_mid = X_left + self.T_half  # meters

            if not np.isfinite(X_mid).all():
                print("TRI_ABS: NaN/Inf triangulation. No move sent.")
                laser.stop()
                return False

            z = float(X_mid[2])
            if not (self.MIN_Z_M <= z <= self.MAX_Z_M):
                print(f"TRI_ABS: depth out of bounds z={z:.3f} m. No move sent.")
                laser.stop()
                return False

            # 3) Convert midpoint frame -> laser axis frame by subtracting fixed offset
            # (Offset defined in mm; convert to meters)
            O_m = np.array([
                self.LASER_OFFSET_X_MM / 1000.0,
                self.LASER_OFFSET_Y_MM / 1000.0,
                0.0
            ], dtype=float)
            X_laser = X_mid - O_m  # meters

            # 4) meters -> mm + sign + optional gain
            dx_mm = self.SIGN_X * self.X_GAIN * float(X_laser[0] * 1000.0)
            dy_mm = self.SIGN_Y * self.Y_GAIN * float(X_laser[1] * 1000.0)

            mpos = laser.update_status()
            if mpos is None:
                print("TRI_ABS: couldn't read MPos. No move sent.")
                return False

            tx = float(mpos["x"] + dx_mm)
            ty = float(mpos["y"] + dy_mm)

            print("\n=== TRI_ABS DEBUG ===")
            print(f"T (m):       [{self.T[0]: .4f}, {self.T[1]: .4f}, {self.T[2]: .4f}]")
            print(f"T_half (m):  [{self.T_half[0]: .4f}, {self.T_half[1]: .4f}, {self.T_half[2]: .4f}]")
            print(f"X_left (m):  [{X_left[0]: .4f}, {X_left[1]: .4f}, {X_left[2]: .4f}]")
            print(f"X_mid (m):   [{X_mid[0]: .4f}, {X_mid[1]: .4f}, {X_mid[2]: .4f}]")
            print(f"Offset (mm): ox={self.LASER_OFFSET_X_MM: .3f}, oy={self.LASER_OFFSET_Y_MM: .3f}")
            print(f"X_laser (m): [{X_laser[0]: .4f}, {X_laser[1]: .4f}, {X_laser[2]: .4f}]")
            print(f"Gains:       X_GAIN={self.X_GAIN:.3f}, Y_GAIN={self.Y_GAIN:.3f}")
            print(f"Δ (mm):      dx={dx_mm: .3f}, dy={dy_mm: .3f}")
            print(f"MPos (mm):   x={mpos['x']: .3f}, y={mpos['y']: .3f}")
            print(f"TARGET (mm): X={tx: .3f}, Y={ty: .3f}")
            print("=====================\n")

            laser.send_raw("G90")
            laser.send_raw(f"G1 X{tx:.3f} Y{ty:.3f} F{self.feed}")

            self._sent = True
            self._target_x = tx
            self._target_y = ty
            return False

        mpos = laser.update_status()
        if mpos is None:
            return False

        ex = self._target_x - mpos["x"]
        ey = self._target_y - mpos["y"]
        err = float(np.hypot(ex, ey))

        if err <= self.TOL_MM:
            laser.stop()
            return True

        return False