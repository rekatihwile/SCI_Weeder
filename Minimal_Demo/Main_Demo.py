import cv2
import numpy as np
import time
import sys
import json
import platform
from pathlib import Path

# Local imports
from motion_helpers import B1LaserController
from cv_helpers import WeedCV

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
CAMERA_SETTINGS = BASE_DIR / "camera_config.json"
HARDWARE_CONFIG = BASE_DIR / "hardware_config.json"
IS_WINDOWS = platform.system() == "Windows"
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

class SCIWeederMain:
    def __init__(self):
        self.frame_count = 0
        self.load_configs()
        self.init_hardware()
        # Initialize your CV and Motion classes
        # self.laser = B1LaserController(self.hw['serial']['grbl_port'])
        # self.cv = WeedCV(...)

    def load_configs(self):
        with open(HARDWARE_CONFIG, 'r') as f: self.hw = json.load(f)
        with open(CAMERA_SETTINGS, 'r') as f: self.cam_settings = json.load(f)

    def init_hardware(self):
        print(f"🚀 Initializing Hardware on {platform.system()}...")
        idx_l = self.hw['cameras']['left']['index']
        idx_r = self.hw['cameras']['right']['index']
        
        self.cap_L = cv2.VideoCapture(idx_l, BACKEND)
        self.cap_R = cv2.VideoCapture(idx_r, BACKEND)
        
        # Apply the visual lock immediately
        self.apply_nuclear_lock()

    def apply_nuclear_lock(self):
        """Forces the hardware to match our saved JSON configuration."""
        for side, cap in [("left", self.cap_L), ("right", self.cap_R)]:
            s = self.cam_settings[side]
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # Manual
            cap.set(cv2.CAP_PROP_BRIGHTNESS, s['brightness'])
            cap.set(cv2.CAP_PROP_CONTRAST, s['contrast'])
            cap.set(cv2.CAP_PROP_EXPOSURE, s['exposure'])
            
            # Flush 5 frames to let the auto-gain settle on the new manual settings
            for _ in range(5): cap.grab()
        print(f"🔒 Hardware Lock Applied (Frame {self.frame_count})")

    def run(self):
        while True:
            self.frame_count += 1
            
            # PERIODIC RE-LOCK: Checks every 500 frames to prevent driver reset
            if self.frame_count % 500 == 0:
                self.apply_nuclear_lock()

            ret_l, frame_l = self.cap_L.read()
            ret_r, frame_r = self.cap_R.read()
            
            if not (ret_l and ret_r): break

            # --- YOUR YOLO / MOTION LOGIC HERE ---
            # ...
            
            cv2.imshow("SCI_Weeder_Stream", cv2.hconcat([frame_l, frame_r]))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == "__main__":
    app = SCIWeederMain()
    app.run()