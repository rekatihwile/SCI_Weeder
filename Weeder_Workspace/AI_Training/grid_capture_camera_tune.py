from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import IS_WINDOWS, LEFT_CAMERA_INDEX, RIGHT_CAMERA_INDEX

WINDOW_NAME = "Stereo Sync Tuner"

MAX_PREVIEW_WIDTH = 1400

# Hard reset values applied to BOTH cameras on startup
INITIAL = {
    "auto_exposure": 0,   # backend dependent; may need 1 or 0.25 on some systems
    "auto_wb": 0,
    "brightness": 128,
    "contrast": 128,
    "exposure": -6,
    "wb_temp": 4500,
    "gain": 0,
    "saturation": 128,
    "hue": 0,
    "sharpness": 128,
    "gamma": 100,
}


def nothing(_: int) -> None:
    pass


def _load_camera_devices() -> dict[str, str]:
    hardware_config_path = WORKSPACE_ROOT / "params" / "hardware_config.json"
    if not hardware_config_path.exists():
        return {}

    try:
        hardware = json.loads(hardware_config_path.read_text())
        cameras = hardware.get("cameras", {})
        return {
            side: str(info["device"])
            for side, info in cameras.items()
            if isinstance(info, dict) and info.get("device")
        }
    except Exception:
        return {}


def open_camera(index: int, side: str) -> cv2.VideoCapture:
    if IS_WINDOWS:
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        source_desc = f"index {index}"
    else:
        camera_devices = _load_camera_devices()
        source = camera_devices.get(side, f"/dev/video{index}")
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        source_desc = source

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {side} camera ({source_desc})")
    return cap


def apply_full_reset(cap: cv2.VideoCapture) -> None:
    # Kill auto modes first
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(INITIAL["auto_exposure"]))
    cap.set(cv2.CAP_PROP_AUTO_WB, float(INITIAL["auto_wb"]))

    # Force same baseline on both cameras
    cap.set(cv2.CAP_PROP_BRIGHTNESS, INITIAL["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, INITIAL["contrast"])
    cap.set(cv2.CAP_PROP_EXPOSURE, INITIAL["exposure"])
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, INITIAL["wb_temp"])

    # Extra baseline controls so both start closer
    cap.set(cv2.CAP_PROP_GAIN, INITIAL["gain"])
    cap.set(cv2.CAP_PROP_SATURATION, INITIAL["saturation"])
    cap.set(cv2.CAP_PROP_HUE, INITIAL["hue"])
    cap.set(cv2.CAP_PROP_SHARPNESS, INITIAL["sharpness"])
    cap.set(cv2.CAP_PROP_GAMMA, INITIAL["gamma"])


def apply_shared_settings(left: cv2.VideoCapture, right: cv2.VideoCapture, settings: dict) -> None:
    for cap in (left, right):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,  float(settings["auto_exposure"]))
        cap.set(cv2.CAP_PROP_AUTO_WB,        float(settings["auto_wb"]))
        cap.set(cv2.CAP_PROP_BRIGHTNESS,     settings["brightness"])
        cap.set(cv2.CAP_PROP_CONTRAST,       settings["contrast"])
        cap.set(cv2.CAP_PROP_EXPOSURE,       settings["exposure"])
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, settings["wb_temp"])


def create_trackbars() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("brightness", WINDOW_NAME, INITIAL["brightness"], 255, nothing)
    cv2.createTrackbar("contrast", WINDOW_NAME, INITIAL["contrast"], 255, nothing)

    # exposure range stored as offset because OpenCV trackbars cannot go negative
    # real exposure = slider - 13  -> range [-13, 0]
    cv2.createTrackbar("exposure", WINDOW_NAME, INITIAL["exposure"] + 13, 13, nothing)

    # WB temp range [2000, 9000]
    # stored as slider offset: real = 2000 + slider
    cv2.createTrackbar("wb_temp", WINDOW_NAME, INITIAL["wb_temp"] - 2000, 7000, nothing)


def reset_trackbars() -> None:
    cv2.setTrackbarPos("brightness", WINDOW_NAME, INITIAL["brightness"])
    cv2.setTrackbarPos("contrast", WINDOW_NAME, INITIAL["contrast"])
    cv2.setTrackbarPos("exposure", WINDOW_NAME, INITIAL["exposure"] + 13)
    cv2.setTrackbarPos("wb_temp", WINDOW_NAME, INITIAL["wb_temp"] - 2000)


def read_ui_settings() -> dict:
    return {
        "auto_exposure": 0,
        "auto_wb": 0,
        "brightness": cv2.getTrackbarPos("brightness", WINDOW_NAME),
        "contrast": cv2.getTrackbarPos("contrast", WINDOW_NAME),
        "exposure": cv2.getTrackbarPos("exposure", WINDOW_NAME) - 13,
        "wb_temp": 2000 + cv2.getTrackbarPos("wb_temp", WINDOW_NAME),
    }


def read_frame(cap: cv2.VideoCapture, name: str):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read from {name}")
    return frame


def stack_frames(left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
    if left_frame.shape[0] != right_frame.shape[0]:
        h = min(left_frame.shape[0], right_frame.shape[0])
        left_frame = cv2.resize(
            left_frame,
            (int(left_frame.shape[1] * h / left_frame.shape[0]), h),
        )
        right_frame = cv2.resize(
            right_frame,
            (int(right_frame.shape[1] * h / right_frame.shape[0]), h),
        )

    combined = cv2.hconcat([left_frame, right_frame])

    h, w = combined.shape[:2]
    if w > MAX_PREVIEW_WIDTH:
        scale = MAX_PREVIEW_WIDTH / w
        combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

    return combined


def draw_overlay(frame: np.ndarray, settings: dict) -> np.ndarray:
    out = frame.copy()
    lines = [
        f"brightness={settings['brightness']}  contrast={settings['contrast']}  exposure={settings['exposure']}  wb_temp={settings['wb_temp']}",
        "Controls affect BOTH cameras",
        "r = reset to initial values | q = quit",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return out


def main() -> None:
    left = None
    right = None

    try:
        left = open_camera(LEFT_CAMERA_INDEX, "left")
        right = open_camera(RIGHT_CAMERA_INDEX, "right")

        # Hit both cameras repeatedly so the driver actually accepts the reset
        for _ in range(6):
            apply_full_reset(left)
            apply_full_reset(right)

        create_trackbars()
        reset_trackbars()

        print("Running.")
        print("Controls affect both cameras.")
        print("Press 'r' to reset, 'q' to quit.")

        while True:
            settings = read_ui_settings()

            # Re-apply every loop to keep both pinned to the same state
            apply_shared_settings(left, right, settings)

            frame_l = read_frame(left, "left camera")
            frame_r = read_frame(right, "right camera")

            preview = stack_frames(frame_l, frame_r)
            preview = draw_overlay(preview, settings)

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                for _ in range(3):
                    apply_full_reset(left)
                    apply_full_reset(right)
                reset_trackbars()

    finally:
        if left is not None:
            left.release()
        if right is not None:
            right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
