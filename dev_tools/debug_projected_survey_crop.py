#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CALIB_NPZ_PATH,
    RECT_NPZ_PATH,
    CALIBRATION_EXPECTS_UNFLIPPED,
    TRI_X_SIGN,
    TRI_Y_SIGN,
    TRI_X_GAIN,
    TRI_Y_GAIN,
    LASER_OFFSET_X_MM,
    LASER_OFFSET_Y_MM,
    SURVEY_POS_X,
    SURVEY_POS_Y,
    WORKSPACE_X_MIN,
    WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Y_MAX,
    SURVEY_PROJECT_CROP_MARGIN_PX,
    SURVEY_PROJECT_Z_MIN_MM,
    SURVEY_PROJECT_Z_MAX_MM,
    SURVEY_PROJECT_Z_SAMPLES,
)
from vision.workspace_crop import project_workspace_crop_left_right


OUT_PATH = ROOT / "planning" / "projected_survey_crop_debug.png"


def _load_input_frames(left_path, right_path, use_camera):
    if left_path and right_path:
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError("Could not load one or both input images.")
        return left, right

    if not use_camera:
        raise RuntimeError("Provide both --left/--right image paths, or pass --capture.")

    from hardware.cameras import StereoCameras

    cameras = StereoCameras()
    try:
        cameras.open()
        left, right = None, None
        for _ in range(12):
            left, right = cameras.read_pair()
            if left is not None and right is not None:
                break
        if left is None or right is None:
            raise RuntimeError("Failed to capture a stereo pair from cameras.")
        return left, right
    finally:
        cameras.close()


def _draw_rect(frame, rect, color, label):
    x0, y0, x1, y1 = rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    text = f"{label} x={x0} y={y0} w={x1 - x0} h={y1 - y0}"
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)


def main():
    parser = argparse.ArgumentParser(description="Debug projected survey crop rectangles.")
    parser.add_argument("--left", type=Path, default=None, help="Path to a left image file.")
    parser.add_argument("--right", type=Path, default=None, help="Path to a right image file.")
    parser.add_argument("--capture", action="store_true", help="Capture one stereo pair from cameras.")
    args = parser.parse_args()

    left, right = _load_input_frames(args.left, args.right, args.capture)

    h, w = left.shape[:2]
    if right.shape[:2] != (h, w):
        raise RuntimeError("Left/right image sizes do not match.")

    left_rect, right_rect, info = project_workspace_crop_left_right(
        frame_width=w,
        frame_height=h,
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

    if left_rect is None or right_rect is None:
        raise RuntimeError(f"Projected crop failed: {(info or {}).get('reason', 'unknown')}")

    _draw_rect(left, left_rect, (0, 255, 0), "LEFT")
    _draw_rect(right, right_rect, (0, 255, 255), "RIGHT")

    divider = 255 * (left[:, :8] * 0 + 1)
    canvas = cv2.hconcat([left, divider, right])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), canvas)

    print(f"Saved debug image: {OUT_PATH}")
    print(f"LEFT  rect: {left_rect}")
    print(f"RIGHT rect: {right_rect}")
    print(f"Frame size: {w}x{h} (calib reference {FRAME_WIDTH}x{FRAME_HEIGHT})")


if __name__ == "__main__":
    main()
