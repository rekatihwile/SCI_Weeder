import cv2
import numpy as np
import json
import sys
import threading
from pathlib import Path

from motion_helpers import B1LaserController
from control_systems import PixelPDControl, TriangulateAbsoluteControl

BASE_DIR = Path(__file__).resolve().parent
HARDWARE_CONFIG = BASE_DIR / "hardware_config.json"
CAMERA_CONFIG   = BASE_DIR / "camera_config.json"

CALIB_NPZ = BASE_DIR / "stereo_charuco_calib.npz"
RECT_NPZ  = BASE_DIR / "stereo_rectify_maps.npz"

IS_WINDOWS = sys.platform.startswith("win")
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

W, H = 640, 480
TARGET_Y_L, TARGET_Y_R = 240, 240
ZOOM_CROP_SIZE = 50
ZOOM_DISPLAY_SIZE = 600

# IMPORTANT: cameras are physically upside down → rotate for UI + tracking
ROTATE_180_LEFT = True
ROTATE_180_RIGHT = True

LK_PARAMS = dict(
    winSize=(31, 31),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

def update_camera(cap, props):
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


class MainUI:
    def __init__(self):
        with open(HARDWARE_CONFIG, "r") as f:
            cfg = json.load(f)

        self.PORT = cfg["serial"]["grbl_port"]
        self.CAM_L_IDX = cfg["cameras"]["left"]["index"]
        self.CAM_R_IDX = cfg["cameras"]["right"]["index"]

        self.laser = B1LaserController(self.PORT)
        self.laser.send_raw("$X")

        self.cap_L = cv2.VideoCapture(self.CAM_L_IDX, BACKEND)
        self.cap_R = cv2.VideoCapture(self.CAM_R_IDX, BACKEND)

        for cap in (self.cap_L, self.cap_R):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if CAMERA_CONFIG.exists():
            try:
                with open(CAMERA_CONFIG, "r") as f:
                    cam_cfg = json.load(f)
                update_camera(self.cap_L, cam_cfg.get("left", {}))
                update_camera(self.cap_R, cam_cfg.get("right", {}))
            except Exception as e:
                print(f"Camera settings load error: {e}")

        self.track_pt_L = None
        self.track_pt_R = None
        self.old_gray_L = None
        self.old_gray_R = None

        self.crop_x1 = 0
        self.crop_y1 = 0

        self.state = "SELECTING"  # SELECTING -> ZOOM_L/ZOOM_R -> WAITING_ENTER -> RUNNING -> PROMPT_FIRE
        self.fire_thread_active = False

        self.pd_mode = PixelPDControl(w=W, target_y_l=TARGET_Y_L, target_y_r=TARGET_Y_R)
        self.tri_abs_mode = None
        self.control = self.pd_mode  # default

    def click_event(self, event, x, y, flags, which_cam):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.state != "SELECTING":
            return

        self.crop_x1 = max(0, x - ZOOM_CROP_SIZE)
        self.crop_y1 = max(0, y - ZOOM_CROP_SIZE)
        self.state = "ZOOM_L" if which_cam == "L" else "ZOOM_R"

        cv2.namedWindow("Magnified View", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Magnified View", self.zoom_click_event)

    def zoom_click_event(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        inv_scale = ZOOM_DISPLAY_SIZE / (ZOOM_CROP_SIZE * 2.0)
        rel_x = x / inv_scale
        rel_y = y / inv_scale

        orig_x = self.crop_x1 + rel_x
        orig_y = self.crop_y1 + rel_y

        pt = np.array([[orig_x, orig_y]], dtype=np.float32).reshape(-1, 1, 2)

        if self.state == "ZOOM_L":
            self.track_pt_L = pt
        elif self.state == "ZOOM_R":
            self.track_pt_R = pt

        self.state = "SELECTING"

        try:
            cv2.destroyWindow("Magnified View")
        except cv2.error:
            pass

        if self.track_pt_L is not None and self.track_pt_R is not None:
            self.state = "WAITING_ENTER"

    def terminal_fire_prompt(self):
        self.fire_thread_active = True
        print("\n" + "=" * 40)
        print("LASER IN POSITION")
        print(f"Mode: {self.control.name}")
        print("=" * 40)

        user_input = input("Press ENTER to FIRE (or type 'n' to cancel): ")
        if user_input.lower() != "n":
            self.laser.fire_low()
            print("Fired.")
        else:
            print("Cancelled.")

        self.state = "SELECTING"
        self.track_pt_L = None
        self.track_pt_R = None
        self.control.reset()
        self.fire_thread_active = False

    def set_mode_pd(self):
        self.control = self.pd_mode
        self.control.reset()
        print("Mode set: PIXEL_PD (1)")

    def set_mode_tri_abs(self):
        if self.tri_abs_mode is None:
            if not CALIB_NPZ.exists() or not RECT_NPZ.exists():
                print("Triangulation mode needs stereo_charuco_calib.npz and stereo_rectify_maps.npz in this folder.")
                return
            self.tri_abs_mode = TriangulateAbsoluteControl(CALIB_NPZ, RECT_NPZ, feed=12000)

        self.control = self.tri_abs_mode
        self.control.reset()
        print("Mode set: TRI_ABS (2)")

    def run(self):
        winL, winR = "Left", "Right"
        cv2.namedWindow(winL)
        cv2.namedWindow(winR)
        cv2.setMouseCallback(winL, lambda e, x, y, f, p: self.click_event(e, x, y, f, "L"))
        cv2.setMouseCallback(winR, lambda e, x, y, f, p: self.click_event(e, x, y, f, "R"))

        print("Ready.")
        print("Click a target in BOTH windows, then press ENTER.")
        print("Keys: [ENTER]=run  [r]=reset  [1]=PIXEL_PD  [2]=TRI_ABS  [q]=quit")

        while True:
            retL, frameL = self.cap_L.read()
            retR, frameR = self.cap_R.read()
            if not retL or not retR:
                break

            # Rotate the UI frames (this also rotates the tracking coordinate system)
            if ROTATE_180_LEFT:
                frameL = cv2.rotate(frameL, cv2.ROTATE_180)
            if ROTATE_180_RIGHT:
                frameR = cv2.rotate(frameR, cv2.ROTATE_180)

            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

            # magnified view
            if self.state in ("ZOOM_L", "ZOOM_R"):
                src = frameL if self.state == "ZOOM_L" else frameR

                x1 = self.crop_x1
                y1 = self.crop_y1
                x2 = min(W, x1 + (ZOOM_CROP_SIZE * 2))
                y2 = min(H, y1 + (ZOOM_CROP_SIZE * 2))

                crop = src[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    zoom_display = cv2.resize(crop, (ZOOM_DISPLAY_SIZE, ZOOM_DISPLAY_SIZE), interpolation=cv2.INTER_LANCZOS4)
                    ch, cw = ZOOM_DISPLAY_SIZE, ZOOM_DISPLAY_SIZE
                    cv2.line(zoom_display, (cw // 2, 0), (cw // 2, ch), (255, 255, 255), 1)
                    cv2.line(zoom_display, (0, ch // 2), (cw, ch // 2), (255, 255, 255), 1)
                    cv2.imshow("Magnified View", zoom_display)

            xl = yl = xr = yr = None

            if self.track_pt_L is not None and self.old_gray_L is not None:
                new_pt, st, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_L, grayL, self.track_pt_L, None, **LK_PARAMS)
                if st[0]:
                    self.track_pt_L = new_pt
                    xl, yl = float(new_pt[0, 0, 0]), float(new_pt[0, 0, 1])
                    cv2.circle(frameL, (int(xl), int(yl)), 5, (0, 0, 255), -1)

            if self.track_pt_R is not None and self.old_gray_R is not None:
                new_pt, st, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_R, grayR, self.track_pt_R, None, **LK_PARAMS)
                if st[0]:
                    self.track_pt_R = new_pt
                    xr, yr = float(new_pt[0, 0, 0]), float(new_pt[0, 0, 1])
                    cv2.circle(frameR, (int(xr), int(yr)), 5, (0, 0, 255), -1)

            if self.state == "RUNNING" and (xl is not None and xr is not None):
                done = self.control.update(xl, yl, xr, yr, self.laser)
                if done:
                    self.state = "PROMPT_FIRE"

            # crosshairs
            cv2.line(frameL, (W // 2, 0), (W // 2, H), (255, 0, 0), 1)
            cv2.line(frameL, (0, TARGET_Y_L), (W, TARGET_Y_L), (255, 0, 0), 1)
            cv2.line(frameR, (W // 2, 0), (W // 2, H), (255, 0, 0), 1)
            cv2.line(frameR, (0, TARGET_Y_R), (W, TARGET_Y_R), (255, 0, 0), 1)

            status = {
                "SELECTING": "Click target (both cams)",
                "ZOOM_L": "Magnify LEFT: click center",
                "ZOOM_R": "Magnify RIGHT: click center",
                "WAITING_ENTER": "ENTER to run | r to redo",
                "RUNNING": f"Running... ({self.control.name}) | r to redo",
                "PROMPT_FIRE": "READY"
            }.get(self.state, "---")

            cv2.putText(frameL, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frameL, f"Mode: {self.control.name} (1/2)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(winL, frameL)
            cv2.imshow(winR, frameR)

            if self.state == "PROMPT_FIRE" and not self.fire_thread_active:
                threading.Thread(target=self.terminal_fire_prompt, daemon=True).start()

            self.old_gray_L = grayL.copy()
            self.old_gray_R = grayR.copy()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                self.set_mode_pd()
            elif key == ord("2"):
                self.set_mode_tri_abs()
            elif key == 13 and self.state == "WAITING_ENTER":
                self.control.reset()
                self.state = "RUNNING"
            elif key == ord("r"):
                print("Reset.")
                self.laser.stop()
                self.state = "SELECTING"
                self.track_pt_L = None
                self.track_pt_R = None
                self.control.reset()
                try:
                    cv2.destroyWindow("Magnified View")
                except cv2.error:
                    pass

        self.cap_L.release()
        self.cap_R.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MainUI().run()