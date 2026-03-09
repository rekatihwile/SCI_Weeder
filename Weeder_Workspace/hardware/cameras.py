import cv2
import time

from config import (
    LEFT_CAMERA_INDEX,
    RIGHT_CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_SETTINGS,
    IS_WINDOWS,
)


BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2


def apply_camera_settings(cap, props):
    if not props:
        return

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    cap.set(cv2.CAP_PROP_BRIGHTNESS, props.get("brightness", 0))
    cap.set(cv2.CAP_PROP_CONTRAST, props.get("contrast", 0))
    cap.set(cv2.CAP_PROP_EXPOSURE, props.get("exposure", -6))
    cap.set(cv2.CAP_PROP_GAIN, props.get("gain", 0))
    cap.set(cv2.CAP_PROP_SATURATION, props.get("saturation", 64))
    cap.set(cv2.CAP_PROP_SHARPNESS, props.get("sharpness", 100))
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, props.get("white_balance", 4000))


class StereoCameras:
    def __init__(self):
        self.left = None
        self.right = None

    def open(self):
        print("\n=== OPENING CAMERAS ===")

        self.left = cv2.VideoCapture(LEFT_CAMERA_INDEX, BACKEND)
        self.right = cv2.VideoCapture(RIGHT_CAMERA_INDEX, BACKEND)

        for cap in [self.left, self.right]:
            if not cap.isOpened():
                raise RuntimeError("Failed to open one or both cameras.")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        time.sleep(0.5)

        apply_camera_settings(self.left, CAMERA_SETTINGS.get("left"))
        apply_camera_settings(self.right, CAMERA_SETTINGS.get("right"))

        for _ in range(5):
            self.left.grab()
            self.right.grab()

        print("Stereo cameras opened.")

    def _flip_frame(self, frame):
        return cv2.rotate(frame, cv2.ROTATE_180)

    def read_pair(self):
        if self.left is None or self.right is None:
            raise RuntimeError("Cameras are not open.")

        ret_l, frame_l = self.left.read()
        ret_r, frame_r = self.right.read()

        if not ret_l or not ret_r:
            raise RuntimeError("Failed to read stereo pair.")

        frame_l = self._flip_frame(frame_l)
        frame_r = self._flip_frame(frame_r)

        return frame_l, frame_r

    def close(self):
        if self.left is not None:
            self.left.release()
        if self.right is not None:
            self.right.release()