from pathlib import Path

import cv2
import numpy as np


def _normalize_rectified_calibration_units_to_meters(T, P1, P2):
    T = np.asarray(T, dtype=np.float64).reshape(3)
    P1 = np.asarray(P1, dtype=np.float64).copy()
    P2 = np.asarray(P2, dtype=np.float64).copy()

    baseline = float(np.linalg.norm(T))
    if baseline > 1.0:
        scale = 1e-3
        T *= scale
        P1[:, 3] *= scale
        P2[:, 3] *= scale

    return T, P1, P2


def _workspace_sample_points_xy(
    workspace_x_min,
    workspace_x_max,
    workspace_y_min,
    workspace_y_max,
):
    xm = 0.5 * (workspace_x_min + workspace_x_max)
    ym = 0.5 * (workspace_y_min + workspace_y_max)
    return [
        (workspace_x_min, workspace_y_min),
        (workspace_x_min, workspace_y_max),
        (workspace_x_max, workspace_y_min),
        (workspace_x_max, workspace_y_max),
        (xm, workspace_y_min),
        (xm, workspace_y_max),
        (workspace_x_min, ym),
        (workspace_x_max, ym),
        (xm, ym),
    ]


def _flip_point_180(u, v, width, height):
    return (float(width - 1) - float(u), float(height - 1) - float(v))


def clamp_crop_rect(rect, frame_width, frame_height, min_size_px=24):
    if rect is None:
        return None

    x0, y0, x1, y1 = rect
    x0 = int(np.floor(float(x0)))
    y0 = int(np.floor(float(y0)))
    x1 = int(np.ceil(float(x1)))
    y1 = int(np.ceil(float(y1)))

    x0 = max(0, min(frame_width, x0))
    x1 = max(0, min(frame_width, x1))
    y0 = max(0, min(frame_height, y0))
    y1 = max(0, min(frame_height, y1))

    if (x1 - x0) < int(min_size_px) or (y1 - y0) < int(min_size_px):
        return None

    return x0, y0, x1, y1


def crop_frame_with_rect(frame, rect):
    x0, y0, x1, y1 = rect
    return frame[y0:y1, x0:x1]


def project_workspace_crop_left_right(
    frame_width,
    frame_height,
    calib_npz_path,
    rect_npz_path,
    workspace_x_min,
    workspace_x_max,
    workspace_y_min,
    workspace_y_max,
    survey_pos_x,
    survey_pos_y,
    tri_sign_x,
    tri_sign_y,
    tri_x_gain,
    tri_y_gain,
    laser_offset_x_mm,
    laser_offset_y_mm,
    z_min_mm,
    z_max_mm,
    z_samples,
    margin_px,
    calibration_expects_unflipped=False,
):
    """Project workspace bounds into left/right survey images and return crop rects.

    Coordinate note:
    _solve_geometry() computes:
      dx_mm = TRI_X_SIGN * TRI_X_GAIN * X_laser.x_mm
      dy_mm = TRI_Y_SIGN * TRI_Y_GAIN * X_laser.y_mm
    We invert that mapping here to recover camera-frame X/Y from workspace dx/dy,
    then assume positive Z points in front of the cameras. If a calibration uses a
    different handedness convention, the caller should keep a generous pixel margin.
    """
    info = {"ok": False, "reason": "unknown"}

    try:
        if tri_sign_x == 0.0 or tri_sign_y == 0.0 or tri_x_gain == 0.0 or tri_y_gain == 0.0:
            info["reason"] = "invalid TRI sign/gain (zero)"
            return None, None, info

        z_samples = int(max(2, z_samples))
        z_min_mm = float(z_min_mm)
        z_max_mm = float(z_max_mm)
        if z_max_mm <= z_min_mm:
            info["reason"] = "invalid Z range"
            return None, None, info

        calib = np.load(Path(calib_npz_path))
        rect = np.load(Path(rect_npz_path))

        K1 = np.asarray(calib["K1"], dtype=np.float64)
        D1 = np.asarray(calib["D1"], dtype=np.float64)
        K2 = np.asarray(calib["K2"], dtype=np.float64)
        D2 = np.asarray(calib["D2"], dtype=np.float64)
        R = np.asarray(calib["R"], dtype=np.float64)
        T = np.asarray(calib["T"], dtype=np.float64).reshape(3)

        R1 = np.asarray(rect["R1"], dtype=np.float64)
        P1 = np.asarray(rect["P1"], dtype=np.float64)
        P2 = np.asarray(rect["P2"], dtype=np.float64)

        T, P1, P2 = _normalize_rectified_calibration_units_to_meters(T, P1, P2)
        T_rect = (R1 @ T.reshape(3, 1)).reshape(3)

        rvec_left = np.zeros((3, 1), dtype=np.float64)
        tvec_left = np.zeros((3, 1), dtype=np.float64)
        rvec_right, _ = cv2.Rodrigues(R)
        tvec_right = T.reshape(3, 1)

        samples_xy = _workspace_sample_points_xy(
            workspace_x_min,
            workspace_x_max,
            workspace_y_min,
            workspace_y_max,
        )
        z_values_mm = np.linspace(z_min_mm, z_max_mm, z_samples)

        points_left_cam = []

        laser_offset_x_m = float(laser_offset_x_mm) / 1000.0
        laser_offset_y_m = float(laser_offset_y_mm) / 1000.0

        for wx, wy in samples_xy:
            dx_mm = float(wx) - float(survey_pos_x)
            dy_mm = float(wy) - float(survey_pos_y)

            x_laser_m = (dx_mm / (float(tri_sign_x) * float(tri_x_gain))) / 1000.0
            y_laser_m = (dy_mm / (float(tri_sign_y) * float(tri_y_gain))) / 1000.0

            for z_mm in z_values_mm:
                z_m = float(z_mm) / 1000.0
                if z_m <= 0.0:
                    continue

                # Invert coarse triangulation mapping to produce a 3D point in the
                # left rectified camera frame, then rotate back into the left raw frame.
                x_mid = x_laser_m + laser_offset_x_m
                y_mid = y_laser_m + laser_offset_y_m
                x_rect = x_mid + 0.5 * float(T_rect[0])
                y_rect = y_mid + 0.5 * float(T_rect[1])
                z_rect = z_m + 0.5 * float(T_rect[2])

                X_rect = np.array([x_rect, y_rect, z_rect], dtype=np.float64)
                X_left = R1.T @ X_rect
                if X_left[2] <= 1e-6:
                    continue
                points_left_cam.append(X_left)

        if not points_left_cam:
            info["reason"] = "no valid 3D samples in front of cameras"
            return None, None, info

        obj = np.asarray(points_left_cam, dtype=np.float64).reshape(-1, 1, 3)
        projL, _ = cv2.fisheye.projectPoints(obj, rvec_left, tvec_left, K1, D1)
        projR, _ = cv2.fisheye.projectPoints(obj, rvec_right, tvec_right, K2, D2)

        uvL = projL.reshape(-1, 2)
        uvR = projR.reshape(-1, 2)

        if calibration_expects_unflipped:
            uvL = np.array([_flip_point_180(u, v, frame_width, frame_height) for (u, v) in uvL], dtype=np.float64)
            uvR = np.array([_flip_point_180(u, v, frame_width, frame_height) for (u, v) in uvR], dtype=np.float64)

        finiteL = np.isfinite(uvL).all(axis=1)
        finiteR = np.isfinite(uvR).all(axis=1)
        uvL = uvL[finiteL]
        uvR = uvR[finiteR]

        if uvL.shape[0] == 0 or uvR.shape[0] == 0:
            info["reason"] = "projection produced no finite points"
            return None, None, info

        margin = float(margin_px)
        left_rect = (
            float(np.min(uvL[:, 0]) - margin),
            float(np.min(uvL[:, 1]) - margin),
            float(np.max(uvL[:, 0]) + margin),
            float(np.max(uvL[:, 1]) + margin),
        )
        right_rect = (
            float(np.min(uvR[:, 0]) - margin),
            float(np.min(uvR[:, 1]) - margin),
            float(np.max(uvR[:, 0]) + margin),
            float(np.max(uvR[:, 1]) + margin),
        )

        left_rect = clamp_crop_rect(left_rect, frame_width, frame_height)
        right_rect = clamp_crop_rect(right_rect, frame_width, frame_height)

        if left_rect is None or right_rect is None:
            info["reason"] = "clamped crop too small/invalid"
            return None, None, info

        info.update({
            "ok": True,
            "reason": "ok",
            "sample_count": int(len(points_left_cam)),
            "left_rect": left_rect,
            "right_rect": right_rect,
        })
        return left_rect, right_rect, info

    except Exception as exc:
        info["reason"] = f"projection exception: {exc}"
        return None, None, info
