#!/usr/bin/env python3
"""Minimal OpenCV probe for the stereo UVC cameras."""

import json
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "diagnostics"
CAMERA_CONFIG = ROOT / "params" / "hardware" / "camera_config.json"
HARDWARE_CONFIG = ROOT / "params" / "hardware" / "hardware_config.json"


def _load_json(path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _camera_entries():
    hw = _load_json(HARDWARE_CONFIG)
    cams = hw.get("cameras", {})
    return [
        ("camera0", int(cams.get("left", {}).get("index", 0)), cams.get("left", {}).get("device", "/dev/video0"), "left"),
        ("camera2", int(cams.get("right", {}).get("index", 2)), cams.get("right", {}).get("device", "/dev/video2"), "right"),
    ]


def _apply_requested_settings(cap, side):
    cfg = _load_json(CAMERA_CONFIG).get(side, {})
    requested = {
        "width": int(cfg.get("width", 1280)),
        "height": int(cfg.get("height", 720)),
        "fps": float(cfg.get("fps", 30.0)),
        "fourcc": str(cfg.get("fourcc", "MJPG")),
    }
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested["height"])
    cap.set(cv2.CAP_PROP_FPS, requested["fps"])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*requested["fourcc"]))

    props = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
        "auto_wb": cv2.CAP_PROP_AUTO_WB,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    }
    for key, prop in props.items():
        if key in cfg:
            cap.set(prop, float(cfg[key]))
    return requested


def _describe_frame(frame):
    if frame is None:
        return {
            "frame_is_none": True,
            "shape": None,
            "dtype": None,
            "mean_brightness": None,
        }
    return {
        "frame_is_none": False,
        "shape": tuple(frame.shape),
        "dtype": str(frame.dtype),
        "mean_brightness": float(np.mean(frame)),
    }


def _run_one(label, index, side, force_settings):
    suffix = "" if force_settings else "_default"
    out_path = DIAG / f"{label}_probe{suffix}.jpg"
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    print(f"\n[{label}] open index={index} backend=CAP_V4L2 force_settings={force_settings}")
    print(f"[{label}] isOpened={cap.isOpened()}")

    requested = None
    if cap.isOpened() and force_settings:
        requested = _apply_requested_settings(cap, side)
        time.sleep(0.25)
        print(f"[{label}] requested={requested}")

    actual = {
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH) if cap.isOpened() else None,
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if cap.isOpened() else None,
        "fps": cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else None,
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)) if cap.isOpened() else None,
    }
    print(f"[{label}] actual={actual}")

    ok_count = 0
    first_valid = None
    last_frame = None
    none_count = 0
    failed_count = 0
    for i in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok_count += 1
            last_frame = frame
            if first_valid is None:
                first_valid = frame.copy()
        else:
            failed_count += 1
            if frame is None:
                none_count += 1
        time.sleep(0.01)

    desc = _describe_frame(last_frame)
    print(f"[{label}] successful_reads={ok_count}/30 failed_reads={failed_count} none_frames={none_count}")
    print(f"[{label}] last_frame={desc}")
    if first_valid is not None:
        saved = cv2.imwrite(str(out_path), first_valid)
        print(f"[{label}] saved_first_valid={saved} path={out_path}")
    else:
        print(f"[{label}] saved_first_valid=False path={out_path}")

    cap.release()


def main():
    DIAG.mkdir(parents=True, exist_ok=True)
    print(f"probe_root={ROOT}")
    print(f"opencv_version={cv2.__version__}")
    for label, index, _device, side in _camera_entries():
        _run_one(label, index, side, force_settings=True)
    for label, index, _device, side in _camera_entries():
        _run_one(label, index, side, force_settings=False)


if __name__ == "__main__":
    main()
