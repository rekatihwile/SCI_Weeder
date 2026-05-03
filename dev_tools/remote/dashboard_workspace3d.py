"""3D workspace reprojection page for the remote dashboard.

This projects real gantry/workspace coordinates back into both stereo images.

It is the inverse of the geometry used by TriangulationCoarseMover._solve_geometry():

    matched pixels
      -> X_rect
      -> X_mid = X_rect - 0.5*T_rect
      -> X_laser = X_mid - laser_offset
      -> dx/dy mm
      -> target_xy_mm = ref_xy_mm + dx/dy

This module does the reverse:

    target_xy_mm
      -> dx/dy from reference gantry pose
      -> X_laser
      -> X_mid
      -> X_rect
      -> projected left/right pixels

Routes owned:
    /workspace3d
    /api/workspace3d/project
"""

import base64

import cv2
import numpy as np
from flask import jsonify, render_template, request

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CALIB_NPZ_PATH,
    RECT_NPZ_PATH,
    LASER_OFFSET_X_MM,
    LASER_OFFSET_Y_MM,
    TRI_SIGN_X,
    TRI_SIGN_Y,
    TRI_X_GAIN,
    TRI_Y_GAIN,
    WORKSPACE_X_MIN,
    WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Y_MAX,
)

from dashboard_state import state
from dashboard_camera import ensure_cameras
from dashboard_rectify import maybe_rectify_pair
from dashboard_gantry import gantry_position_payload


# =============================================================================
# Encoding helpers
# =============================================================================

def _encode_bgr(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return base64.b64encode(buf).decode("utf-8")


# =============================================================================
# Calibration helpers
# =============================================================================

def _normalize_rectified_calibration_units_to_meters(T, P1, P2):
    """
    Match the same normalization convention used in control/coarse_move.py.

    Some calibration files store baseline in mm, some in m. If T norm is > 1,
    assume mm and convert T/P translation columns to meters.
    """
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


class Workspace3DProjector:
    """Inverse-projection model matching TriangulationCoarseMover geometry."""

    def __init__(self):
        calib = np.load(CALIB_NPZ_PATH)
        rect = np.load(RECT_NPZ_PATH)

        self.K1 = np.asarray(calib["K1"], dtype=np.float64)
        self.D1 = np.asarray(calib["D1"], dtype=np.float64)
        self.K2 = np.asarray(calib["K2"], dtype=np.float64)
        self.D2 = np.asarray(calib["D2"], dtype=np.float64)

        # Raw stereo extrinsics: left camera -> right camera
        self.R = np.asarray(calib["R"], dtype=np.float64)
        self.T = np.asarray(calib["T"], dtype=np.float64).reshape(3)

        self.R1 = np.asarray(rect["R1"], dtype=np.float64)
        self.R2 = np.asarray(rect["R2"], dtype=np.float64)
        self.P1 = np.asarray(rect["P1"], dtype=np.float64)
        self.P2 = np.asarray(rect["P2"], dtype=np.float64)

        self.T, self.P1, self.P2 = _normalize_rectified_calibration_units_to_meters(
            self.T,
            self.P1,
            self.P2,
        )

        self.T_rect = (self.R1 @ self.T.reshape(3, 1)).reshape(3)

    def workspace_xy_to_xrect(self, x_abs_mm, y_abs_mm, ref_x_mm, ref_y_mm, z_laser_mm):
        dx_mm = float(x_abs_mm) - float(ref_x_mm)
        dy_mm = float(y_abs_mm) - float(ref_y_mm)

        denom_x = TRI_SIGN_X * TRI_X_GAIN
        denom_y = TRI_SIGN_Y * TRI_Y_GAIN

        if abs(denom_x) < 1e-12:
            raise RuntimeError("TRI_SIGN_X * TRI_X_GAIN is zero.")
        if abs(denom_y) < 1e-12:
            raise RuntimeError("TRI_SIGN_Y * TRI_Y_GAIN is zero.")

        x_laser_m = (dx_mm / denom_x) / 1000.0
        y_laser_m = (dy_mm / denom_y) / 1000.0
        z_laser_m = float(z_laser_mm) / 1000.0

        X_laser = np.array([x_laser_m, y_laser_m, z_laser_m], dtype=np.float64)

        laser_offset_m = np.array(
            [
                LASER_OFFSET_X_MM / 1000.0,
                LASER_OFFSET_Y_MM / 1000.0,
                0.0,
            ],
            dtype=np.float64,
        )

        X_mid = X_laser + laser_offset_m
        X_rect = X_mid + 0.5 * self.T_rect

        return X_rect

    def project_rectified(self, X_rect):
        Xh = np.array(
            [X_rect[0], X_rect[1], X_rect[2], 1.0],
            dtype=np.float64,
        )

        pL = self.P1 @ Xh
        pR = self.P2 @ Xh

        if abs(pL[2]) < 1e-12 or abs(pR[2]) < 1e-12:
            return None, None

        left_px = (float(pL[0] / pL[2]), float(pL[1] / pL[2]))
        right_px = (float(pR[0] / pR[2]), float(pR[1] / pR[2]))

        return left_px, right_px

    def project_raw(self, X_rect):
        """
        Project into the ORIGINAL raw distorted camera coordinates.

        X_rect is in the rectified-left camera frame.
        So:
          1) undo left rectification
          2) transform raw-left -> raw-right using stereo extrinsics
          3) project each into its own raw camera model
        """
        X_left_raw = self.R1.T @ X_rect
        X_right_raw = self.R @ X_left_raw + self.T

        objL = X_left_raw.reshape(1, 1, 3).astype(np.float64)
        objR = X_right_raw.reshape(1, 1, 3).astype(np.float64)

        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        pL, _ = cv2.fisheye.projectPoints(objL, rvec, tvec, self.K1, self.D1)
        pR, _ = cv2.fisheye.projectPoints(objR, rvec, tvec, self.K2, self.D2)

        left_px = tuple(map(float, pL.reshape(2)))
        right_px = tuple(map(float, pR.reshape(2)))

        return left_px, right_px

    def project_workspace_point(
        self,
        x_abs_mm,
        y_abs_mm,
        ref_x_mm,
        ref_y_mm,
        z_laser_mm,
        rectified=True,
    ):
        X_rect = self.workspace_xy_to_xrect(
            x_abs_mm=x_abs_mm,
            y_abs_mm=y_abs_mm,
            ref_x_mm=ref_x_mm,
            ref_y_mm=ref_y_mm,
            z_laser_mm=z_laser_mm,
        )

        if rectified:
            left_px, right_px = self.project_rectified(X_rect)
        else:
            left_px, right_px = self.project_raw(X_rect)

        return {
            "x_mm": float(x_abs_mm),
            "y_mm": float(y_abs_mm),
            "z_laser_mm": float(z_laser_mm),
            "X_rect_m": [float(v) for v in X_rect],
            "left_px": left_px,
            "right_px": right_px,
        }

def get_projector():
    if state.workspace_projector is None:
        state.workspace_projector = Workspace3DProjector()
    return state.workspace_projector


# =============================================================================
# Grid helpers
# =============================================================================

def _get_gantry_ref_xy():
    """
    Read current gantry position from the existing gantry API helper.
    """
    try:
        payload = gantry_position_payload()
    except Exception as e:
        raise RuntimeError(
            f"[read_gantry_position] Could not connect to gantry: {e}"
        ) from e

    pos = payload.get("position") or {}
    if "x" not in pos or "y" not in pos:
        raise RuntimeError(
            "[read_gantry_position] Gantry is connected but position payload "
            f"is missing x/y. Got payload keys: {list(payload.keys())}"
        )

    return float(pos["x"]), float(pos["y"])


def generate_grid_lines(x_min, x_max, y_min, y_max, nx, ny):
    xs = np.linspace(float(x_min), float(x_max), int(nx) + 1)
    ys = np.linspace(float(y_min), float(y_max), int(ny) + 1)

    lines = []

    for x in xs:
        lines.append({
            "kind": "x",
            "value": float(x),
            "points_xy": [(float(x), float(y)) for y in ys],
        })

    for y in ys:
        lines.append({
            "kind": "y",
            "value": float(y),
            "points_xy": [(float(x), float(y)) for x in xs],
        })

    return lines


def _inside_image(px):
    if px is None:
        return False

    x, y = px
    return 0 <= x < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT

def _draw_projected_grid(img, projected_lines, side):
    """
    Draw projected grid with small axis tick labels.

    Labels are only drawn on the workspace border:
    - x labels along the y-min edge
    - y labels along the x-min edge

    This avoids the old cluttered constant_x / constant_y labels.
    """
    out = img.copy()

    grid_color = (0, 255, 255)
    point_color = (0, 0, 255)
    tick_color = (255, 255, 255)
    tick_shadow = (0, 0, 0)

    # Draw grid lines and node points
    for line in projected_lines:
        img_pts = []

        for p in line["points"]:
            px = p[f"{side}_px"]

            if px is not None and _inside_image(px):
                img_pts.append((int(round(px[0])), int(round(px[1]))))
            else:
                img_pts.append(None)

        for a, b in zip(img_pts[:-1], img_pts[1:]):
            if a is not None and b is not None:
                cv2.line(out, a, b, grid_color, 2, cv2.LINE_AA)

        for pt in img_pts:
            if pt is not None:
                cv2.circle(out, pt, 4, point_color, -1, cv2.LINE_AA)

    # Draw small tick labels only on border lines
    # For x-lines, label the first valid point, which corresponds to y_min.
    # For y-lines, label the first valid point, which corresponds to x_min.
    for line in projected_lines:
        kind = line["kind"]
        value = line["value"]

        pts = []
        for p in line["points"]:
            px = p[f"{side}_px"]
            if px is not None and _inside_image(px):
                pts.append((int(round(px[0])), int(round(px[1]))))
            else:
                pts.append(None)

        valid = [p for p in pts if p is not None]
        if not valid:
            continue

        if kind == "x":
            x0, y0 = valid[0]
            label = f"{value:.0f}"
            pos = (x0 - 10, y0 + 18)
        elif kind == "y":
            x0, y0 = valid[0]
            label = f"{value:.0f}"
            pos = (x0 + 6, y0 + 4)
        else:
            continue

        # black shadow then white text
        cv2.putText(
            out,
            label,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            tick_shadow,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            tick_color,
            1,
            cv2.LINE_AA,
        )

    return out

def _draw_text_box(img, lines):
    out = img.copy()

    x0, y0 = 12, 26
    line_h = 24

    w = 520
    h = 12 + line_h * len(lines)
    cv2.rectangle(out, (6, 6), (6 + w, 6 + h), (0, 0, 0), -1)

    for i, text in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(
            out,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


def build_projected_workspace_grid(params):
    rectified = bool(params.get("rectified", True))

    ref_mode = params.get("ref_mode", "gantry")
    ref_x = float(params.get("ref_x", 200.0))
    ref_y = float(params.get("ref_y", 200.0))

    if ref_mode == "gantry":
        ref_x, ref_y = _get_gantry_ref_xy()

    x_min = float(params.get("x_min", WORKSPACE_X_MIN))
    x_max = float(params.get("x_max", WORKSPACE_X_MAX))
    y_min = float(params.get("y_min", WORKSPACE_Y_MIN))
    y_max = float(params.get("y_max", WORKSPACE_Y_MAX))

    nx = int(params.get("nx", 10))
    ny = int(params.get("ny", 10))

    z_laser_mm = float(params.get("z_laser_mm", 250.0))

    cam = ensure_cameras()

    fL, fR = None, None
    for _ in range(5):
        fL, fR = cam.read_pair()
        if fL is not None and fR is not None:
            break

    if fL is None or fR is None:
        raise RuntimeError(
            "[read_camera_pair] Could not read a valid stereo pair after 5 attempts."
        )

    if rectified:
        fL, fR, _ = maybe_rectify_pair(fL, fR, True)

    projector = get_projector()
    grid_lines = generate_grid_lines(x_min, x_max, y_min, y_max, nx, ny)

    projected_lines = []

    for line in grid_lines:
        projected_points = []

        for x, y in line["points_xy"]:
            projected_points.append(
                projector.project_workspace_point(
                    x_abs_mm=x,
                    y_abs_mm=y,
                    ref_x_mm=ref_x,
                    ref_y_mm=ref_y,
                    z_laser_mm=z_laser_mm,
                    rectified=rectified,
                )
            )

        projected_lines.append({
            "kind": line["kind"],
            "value": line["value"],
            "points": projected_points,
        })

    left_overlay = _draw_projected_grid(fL, projected_lines, "left")
    right_overlay = _draw_projected_grid(fR, projected_lines, "right")

    # text_lines = [
    #     f"3D reprojection: {'rectified' if rectified else 'raw'}",
    #     f"ref_xy = ({ref_x:.1f}, {ref_y:.1f}) mm  mode={ref_mode}",
    #     f"Z_laser = {z_laser_mm:.1f} mm",
    #     f"X = {x_min:.0f}..{x_max:.0f} mm  nx={nx}",
    #     f"Y = {y_min:.0f}..{y_max:.0f} mm  ny={ny}",
    # # ]

    # left_overlay = _draw_text_box(left_overlay, text_lines)
    # right_overlay = _draw_text_box(right_overlay, text_lines)

    return {
        "ok": True,
        "left_image": _encode_bgr(left_overlay),
        "right_image": _encode_bgr(right_overlay),
        "rectified": rectified,
        "ref_mode": ref_mode,
        "ref_x": ref_x,
        "ref_y": ref_y,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "nx": nx,
        "ny": ny,
        "z_laser_mm": z_laser_mm,
        "projected_lines_count": len(projected_lines),
    }


# =============================================================================
# Flask route registration
# =============================================================================

def register_workspace3d_routes(app):
    """Register /workspace3d and its API route on the Flask app."""

    @app.route("/workspace3d")
    def workspace3d_page():
        return render_template(
            "workspace3d.html",
            workspace_x_min=WORKSPACE_X_MIN,
            workspace_x_max=WORKSPACE_X_MAX,
            workspace_y_min=WORKSPACE_Y_MIN,
            workspace_y_max=WORKSPACE_Y_MAX,
        )

    @app.route("/api/workspace3d/project", methods=["POST"])
    def api_workspace3d_project():
        try:
            params = request.get_json(force=True)

            with state.camera_lock:
                result = build_projected_workspace_grid(params)

            return jsonify(result)

        except Exception as e:
            return jsonify({
                "ok": False,
                "error": repr(e),
            }), 500