import cv2
import time
import os
import json
from pathlib import Path

from config import (
    LEFT_CAMERA_INDEX,
    RIGHT_CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_SETTINGS,
    IS_WINDOWS,
    AUTO_MODE,
    BASE_DIR
)

# Windows uses Media Foundation, Linux (Jetson) uses V4L2
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

def apply_camera_settings(cap, props, dev_path=None):
    if not props:
        return

    auto_exposure = float(props.get("auto_exposure", 1))
    auto_wb = float(props.get("auto_wb", 1))

    if os.name == "nt":
        # Windows: Use standard OpenCV
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
        cap.set(cv2.CAP_PROP_AUTO_WB, auto_wb)
        if "exposure" in props: cap.set(cv2.CAP_PROP_EXPOSURE, float(props["exposure"]))
        if "gain" in props: cap.set(cv2.CAP_PROP_GAIN, float(props["gain"]))
        if "brightness" in props: cap.set(cv2.CAP_PROP_BRIGHTNESS, float(props["brightness"]))
        if "contrast" in props: cap.set(cv2.CAP_PROP_CONTRAST, float(props["contrast"]))
        if "saturation" in props: cap.set(cv2.CAP_PROP_SATURATION, float(props["saturation"]))
        if "white_balance" in props: cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(props["white_balance"]))
    else:
        # Jetson / Linux: Completely bypass OpenCV to stop "cross-talk"
        if dev_path is not None:
            print(f"[V4L2 Target] Applying startup settings strictly to -> {dev_path}")
            
            # V4L2 manual is 1, auto is 3. WB manual is 0, auto is 1.
            v4l2_auto_exp = 1 if auto_exposure == 1 else 3
            v4l2_auto_wb = 0 if auto_wb == 1 else 1 

            os.system(f"v4l2-ctl -d {dev_path} -c exposure_auto={v4l2_auto_exp} > /dev/null 2>&1")
            os.system(f"v4l2-ctl -d {dev_path} -c white_balance_temperature_auto={v4l2_auto_wb} > /dev/null 2>&1")

            if "exposure" in props: os.system(f"v4l2-ctl -d {dev_path} -c exposure_absolute={int(props['exposure'])} > /dev/null 2>&1")
            if "gain" in props: os.system(f"v4l2-ctl -d {dev_path} -c gain={int(props['gain'])} > /dev/null 2>&1")
            if "brightness" in props: os.system(f"v4l2-ctl -d {dev_path} -c brightness={int(props['brightness'])} > /dev/null 2>&1")
            if "contrast" in props: os.system(f"v4l2-ctl -d {dev_path} -c contrast={int(props['contrast'])} > /dev/null 2>&1")
            if "saturation" in props: os.system(f"v4l2-ctl -d {dev_path} -c saturation={int(props['saturation'])} > /dev/null 2>&1")
            if "white_balance" in props: os.system(f"v4l2-ctl -d {dev_path} -c white_balance_temperature={int(props['white_balance'])} > /dev/null 2>&1")


class StereoCameras:
    def __init__(self):
        self.left = None
        self.right = None
        
        # Default directly to the indices set in config.py
        self.dev_paths = {
            "left": f"/dev/video{LEFT_CAMERA_INDEX}",
            "right": f"/dev/video{RIGHT_CAMERA_INDEX}"
        }

        # Automatically fetch dev paths from hardware_config.json if overridden
        hw_cfg = BASE_DIR / "params" / "hardware_config.json"
        if hw_cfg.exists():
            with open(hw_cfg, "r") as f:
                hw = json.load(f)
                if "cameras" in hw:
                    if hw["cameras"].get("left") and "device" in hw["cameras"]["left"]:
                        self.dev_paths["left"] = hw["cameras"]["left"]["device"]
                    if hw["cameras"].get("right") and "device" in hw["cameras"]["right"]:
                        self.dev_paths["right"] = hw["cameras"]["right"]["device"]

    def open(self):
        print("\n=== OPENING CAMERAS ===")
        print(f"Opening Left : {LEFT_CAMERA_INDEX}")
        print(f"Opening Right: {RIGHT_CAMERA_INDEX}")

        self.left = cv2.VideoCapture(LEFT_CAMERA_INDEX, BACKEND)
        self.right = cv2.VideoCapture(RIGHT_CAMERA_INDEX, BACKEND)

        for name, cap in [("Left", self.left), ("Right", self.right)]:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {name} camera.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        time.sleep(0.5)

        # Apply settings strictly to individual nodes
        apply_camera_settings(self.left, CAMERA_SETTINGS.get("left"), self.dev_paths["left"])
        apply_camera_settings(self.right, CAMERA_SETTINGS.get("right"), self.dev_paths["right"])

        if AUTO_MODE:
            if os.name == "nt":
                self.left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.left.set(cv2.CAP_PROP_AUTO_WB, 1)
                self.right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.right.set(cv2.CAP_PROP_AUTO_WB, 1)
            else:
                for side in ["left", "right"]:
                    dev = self.dev_paths[side]
                    if dev:
                        os.system(f"v4l2-ctl -d {dev} -c exposure_auto=3 > /dev/null 2>&1")
                        os.system(f"v4l2-ctl -d {dev} -c white_balance_temperature_auto=1 > /dev/null 2>&1")

        for _ in range(5):
            self.left.grab()
            self.right.grab()

        print("Stereo cameras opened.")

    def _flip_frame(self, frame):
        return cv2.rotate(frame, cv2.ROTATE_180)

    def read_pair(self, retries=5):
        if self.left is None or self.right is None:
            raise RuntimeError("Cameras are not open.")

        for attempt in range(retries):
            # Using grab() then retrieve() synchronizes the stereo pair much better 
            # on Linux and helps bypass stale buffers.
            self.left.grab()
            self.right.grab()
            
            ret_l, frame_l = self.left.retrieve()
            ret_r, frame_r = self.right.retrieve()
            
            if ret_l and ret_r:
                return self._flip_frame(frame_l), self._flip_frame(frame_r)
            
            print(f"[WARN] Frame dropped by USB bus (attempt {attempt+1}/{retries}). Retrying...")
            time.sleep(0.05)

        raise RuntimeError("Failed to read stereo pair after multiple retries.")

    def close(self):
        if self.left is not None:
            self.left.release()
        if self.right is not None:
            self.right.release()