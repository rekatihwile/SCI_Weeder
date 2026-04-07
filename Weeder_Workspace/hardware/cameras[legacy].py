import cv2
import time

from config import (
    LEFT_CAMERA_INDEX,
    RIGHT_CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_SETTINGS,
    IS_WINDOWS,
    AUTO_MODE,
)

# Windows uses Media Foundation, Linux (Jetson) uses V4L2
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2


def apply_camera_settings(cap, props):
    if not props:
        return

    auto_exposure = float(props.get("auto_exposure", 0))
    auto_wb = float(props.get("auto_wb", 0))

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
    cap.set(cv2.CAP_PROP_AUTO_WB, auto_wb)

    if "exposure" in props:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(props["exposure"]))
    if "gain" in props:
        cap.set(cv2.CAP_PROP_GAIN, float(props["gain"]))
    if "brightness" in props:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(props["brightness"]))
    if "contrast" in props:
        cap.set(cv2.CAP_PROP_CONTRAST, float(props["contrast"]))
    if "saturation" in props:
        cap.set(cv2.CAP_PROP_SATURATION, float(props["saturation"]))
    if "white_balance" in props:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(props["white_balance"]))
    if hasattr(cv2, "CAP_PROP_SHARPNESS") and "sharpness" in props:
        cap.set(cv2.CAP_PROP_SHARPNESS, float(props["sharpness"]))


class StereoCameras:
    def __init__(self):
        self.left = None
        self.right = None

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

        apply_camera_settings(self.left, CAMERA_SETTINGS.get("left"))
        apply_camera_settings(self.right, CAMERA_SETTINGS.get("right"))

        if AUTO_MODE:
            self.left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.left.set(cv2.CAP_PROP_AUTO_WB, 1)
            self.right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.right.set(cv2.CAP_PROP_AUTO_WB, 1)

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
            import time
            time.sleep(0.05)

        raise RuntimeError("Failed to read stereo pair after multiple retries.")

    def close(self):
        if self.left is not None:
            self.left.release()
        if self.right is not None:
            self.right.release()