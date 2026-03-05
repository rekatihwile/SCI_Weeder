import cv2
import numpy as np
import json
import time
import sys
import threading
from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
HARDWARE_CONFIG = BASE_DIR / "hardware_config.json"
CAMERA_CONFIG = BASE_DIR / "camera_config.json"

IS_WINDOWS = sys.platform.startswith('win')
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

from motion_helpers import B1LaserController

# --- CONFIGURATION LOADING ---
with open(HARDWARE_CONFIG, "r") as f:
    cfg = json.load(f)

PORT = cfg["serial"]["grbl_port"]
CAM_L_IDX = cfg["cameras"]["left"]["index"]
CAM_R_IDX = cfg["cameras"]["right"]["index"]

# --- CONTROL CONSTANTS ---
W, H = 640, 480
TARGET_Y_L, TARGET_Y_R = 240, 240
ZOOM_CROP_SIZE = 50     # 100x100 pixel area from original frame
ZOOM_DISPLAY_SIZE = 600 # Big visual window size on screen

# PD Gains
Kp_x, Kd_x = 10.0, 1.0 
Kp_y, Kd_y = 10.0, 1.0
STEP_MM = 0.001
DEADZONE = 4 
MAX_JOG = 10.0 

LK_PARAMS = dict(winSize=(31, 31), maxLevel=3, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def update_camera(cap, props):
    if not props: return
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, props.get('brightness', 0))
    cap.set(cv2.CAP_PROP_CONTRAST, props.get('contrast', 0))
    cap.set(cv2.CAP_PROP_EXPOSURE, props.get('exposure', -6))
    cap.set(cv2.CAP_PROP_GAIN, props.get('gain', 0))
    cap.set(cv2.CAP_PROP_SATURATION, props.get('saturation', 64))
    cap.set(cv2.CAP_PROP_SHARPNESS, props.get('sharpness', 100))
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, props.get('white_balance', 4000))

class RCBD_Ablator:
    def __init__(self):
        self.laser = B1LaserController(PORT)
        self.cap_L = cv2.VideoCapture(CAM_L_IDX, BACKEND)
        self.cap_R = cv2.VideoCapture(CAM_R_IDX, BACKEND)
        
        for cap in [self.cap_L, self.cap_R]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.apply_camera_settings()
        self.track_pt_L = self.track_pt_R = None
        self.old_gray_L = self.old_gray_R = None
        self.state = "SELECTING" 
        
        # Mapping Anchors (Correct pixel data to global mapping)
        self.crop_x1 = 0
        self.crop_y1 = 0

        self.prev_ex = self.prev_ey = 0
        self.fire_thread_active = False

    def apply_camera_settings(self):
        try:
            with open(CAMERA_CONFIG, "r") as f:
                cam_cfg = json.load(f)
            update_camera(self.cap_L, cam_cfg.get("left"))
            update_camera(self.cap_R, cam_cfg.get("right"))
        except Exception as e: print(f"Settings Error: {e}")

    def zoom_click_event(self, event, x, y, flags, param):
        """Handler for the magnified sub-window with precise coordinate mapping."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # We explicitly resize the crop to the display size, so we reverse the scale
            inv_scale = ZOOM_DISPLAY_SIZE / (ZOOM_CROP_SIZE * 2)
            rel_x_on_crop = x / inv_scale
            rel_y_on_crop = y / inv_scale
            
            orig_x = self.crop_x1 + rel_x_on_crop
            orig_y = self.crop_y1 + rel_y_on_crop
            
            if self.state == "ZOOM_L":
                self.track_pt_L = np.array([[orig_x, orig_y]], dtype=np.float32).reshape(-1, 1, 2)
                self.state = "SELECTING"
            elif self.state == "ZOOM_R":
                self.track_pt_R = np.array([[orig_x, orig_y]], dtype=np.float32).reshape(-1, 1, 2)
                self.state = "SELECTING"

            try:
                cv2.destroyWindow("Magnified View")
            except cv2.error:
                pass

            if self.track_pt_L is not None and self.track_pt_R is not None:
                self.state = "WAITING_ENTER"

    def click_event(self, event, x, y, flags, param):
        """Initial click to trigger magnification."""
        if event == cv2.EVENT_LBUTTONDOWN and self.state == "SELECTING":
            self.crop_x1 = max(0, x - ZOOM_CROP_SIZE)
            self.crop_y1 = max(0, y - ZOOM_CROP_SIZE)
            
            self.state = "ZOOM_L" if param == "left" else "ZOOM_R"
            
            # Use WINDOW_AUTOSIZE because we resize the data explicitly before showing it
            cv2.namedWindow("Magnified View", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("Magnified View", self.zoom_click_event)

    def terminal_fire_prompt(self):
        self.fire_thread_active = True
        print("\n" + "="*40 + "\n🔥 ABLATION READY 🔥\n" + "="*40)
        p = input("Power (0-1000) [100]: ")
        d = input("Time (s) [0.5]: ")
        try:
            power = int(p) if p.strip() else 100
            duration = float(d) if d.strip() else 0.5
            self.laser.serial.write(b"\x85") # Hard stop
            time.sleep(0.1)
            self.laser.send_raw(f"M3 S{power}")
            self.laser.send_raw(f"G4 P{duration}")
            self.laser.send_raw("M5")
            print("✅ Strike Complete.")
        except: print("❌ Error in firing.")
        
        self.state = "SELECTING"
        self.track_pt_L = self.track_pt_R = None
        self.fire_thread_active = False

    def run(self):
        win_l, win_r = "RCBD Tool - LEFT", "RCBD Tool - RIGHT"
        cv2.namedWindow(win_l); cv2.namedWindow(win_r)
        cv2.setMouseCallback(win_l, self.click_event, "left") 
        cv2.setMouseCallback(win_r, self.click_event, "right") 

        while True:
            retL, frameL = self.cap_L.read()
            retR, frameR = self.cap_R.read()
            if not retL or not retR: break

            grayL, grayR = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            xl, yl, xr, yr = None, None, None, None

            # --- MAGNIFIED VIEW LOGIC ---
            if self.state in ["ZOOM_L", "ZOOM_R"]:
                src = frameL if self.state == "ZOOM_L" else frameR
                
                # Get clamp bounds for this selection
                x1 = self.crop_x1
                y1 = self.crop_y1
                x2 = min(W, x1 + (ZOOM_CROP_SIZE * 2))
                y2 = min(H, y1 + (ZOOM_CROP_SIZE * 2))
                
                crop = src[y1 : y2, x1 : x2].copy()
                
                # Resize to make the window visually "large" on your monitor
                zoom_display = cv2.resize(crop, (ZOOM_DISPLAY_SIZE, ZOOM_DISPLAY_SIZE), interpolation=cv2.INTER_LANCZOS4)
                
                # DrawGuide on the visible window coordinate space
                ch, cw = ZOOM_DISPLAY_SIZE, ZOOM_DISPLAY_SIZE
                cv2.line(zoom_display, (cw//2, 0), (cw//2, ch), (255, 100, 0), 1)
                cv2.line(zoom_display, (0, ch//2), (cw, ch//2), (255, 100, 0), 1)
                cv2.imshow("Magnified View", zoom_display)

            # --- TRACKING ---
            if self.track_pt_L is not None and self.old_gray_L is not None:
                new_pt, st, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_L, grayL, self.track_pt_L, None, **LK_PARAMS)
                if st[0]: 
                    self.track_pt_L, (xl, yl) = new_pt, new_pt.ravel()
                    cv2.circle(frameL, (int(xl), int(yl)), 5, (0, 0, 255), -1)

            if self.track_pt_R is not None and self.old_gray_R is not None:
                new_pt, st, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_R, grayR, self.track_pt_R, None, **LK_PARAMS)
                if st[0]:
                    self.track_pt_R, (xr, yr) = new_pt, new_pt.ravel()
                    cv2.circle(frameR, (int(xr), int(yr)), 5, (0, 0, 255), -1)

            # --- HOMING ---
            if self.state == "HOMING" and (xl is not None and xr is not None):
                err_x = -(xl - W + xr)
                err_y = (((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
                dex, dey = err_x - self.prev_ex, err_y - self.prev_ey
                self.prev_ex, self.prev_ey = err_x, err_y

                dx = round((err_x * Kp_x + dex * Kd_x) * STEP_MM, 3) if abs(err_x) > DEADZONE else 0.0
                dy = round((err_y * Kp_y + dey * Kd_y) * STEP_MM, 3) if abs(err_y) > DEADZONE else 0.0

                if abs(err_x) <= DEADZONE and abs(err_y) <= DEADZONE:
                    self.laser.stop()
                    self.state = "PROMPT_FIRE"
                elif abs(dx) > 0.0 or abs(dy) > 0.0:
                    self.laser.jog(np.clip(dx, -MAX_JOG, MAX_JOG), np.clip(dy, -MAX_JOG, MAX_JOG), 5000)

            # --- UI HUD ---
            status = {"SELECTING": "Step 1: Click stem", "ZOOM_L": "Step 2: Center (Magnified View)", "ZOOM_R": "Step 2: Center (Magnified View)", "WAITING_ENTER": "ENTER to Home | 'r' to Redo", "HOMING": "Homing... | 'r' to Redo", "PROMPT_FIRE": "FIRE READY"}.get(self.state, "---")
            cv2.putText(frameL, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win_l, frameL); cv2.imshow(win_r, frameR)

            if self.state == "PROMPT_FIRE" and not self.fire_thread_active:
                threading.Thread(target=self.terminal_fire_prompt, daemon=True).start()

            self.old_gray_L, self.old_gray_R = grayL.copy(), grayR.copy()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == 13 and self.state == "WAITING_ENTER": 
                self.state, self.prev_ex, self.prev_ey = "HOMING", 0, 0
            elif key == ord('r'): 
                print("🔄 Resetting Selection & Stopping Laser.")
                self.laser.stop() # Force laser stop on reset
                self.state = "SELECTING"
                self.track_pt_L = self.track_pt_R = None
                
                # Robustly close the zoom window if it exists
                try:
                    cv2.destroyWindow("Magnified View")
                except cv2.error:
                    pass

        self.cap_L.release(); self.cap_R.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    RCBD_Ablator().run()