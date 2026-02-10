#!/usr/bin/env python3
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURATION ---
CAM_ID = 0  # Using Camera ID 1 as requested
W, H = 1280, 720
IMG_SIZE = 640
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Paths (Ensure these exist in your workspace)
SEG_MODEL_PATH = "best.pt"
REG_MODEL_PATH = "v2_hub_regressor_640.pth"

# Tracking Config
TEMPLATE_SIZE = 50
RESET_INTERVAL = 0.5 
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ==========================================
# ============ HYBRID TRACKER ==============
# ==========================================
class HybridTracker:
    def __init__(self):
        self.target_pt = None  # [x, y]
        self.template = None
        self.prev_gray = None
        self.last_reset_time = 0
        self.is_locked = False

    def lock_onto(self, frame, x, y):
        """Initializes tracking from a YOLO detection point."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.target_pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        
        # Capture Ground Truth Template
        half = TEMPLATE_SIZE // 2
        ix, iy = int(x), int(y)
        self.template = gray[max(0, iy-half):min(H, iy+half), 
                             max(0, ix-half):min(W, ix+half)].copy()
        
        self.last_reset_time = time.time()
        self.is_locked = True
        print(f"🎯 HW Locked onto Plant at {ix}, {iy}")

    def update(self, frame):
        if not self.is_locked: return None
        
        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Periodic Drift Correction (Template Match)
        if time.time() - self.last_reset_time > RESET_INTERVAL:
            self._correct_drift(gray)
            self.last_reset_time = time.time()

        # 2. LK Flow Update
        if self.prev_gray is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.target_pt, None, **LK_PARAMS
            )
            if status[0][0] == 1:
                self.target_pt = new_pts
                x, y = new_pts.ravel()
                # Draw Lock UI
                cv2.circle(vis, (int(x), int(y)), 12, (0, 255, 0), 2)
                cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            else:
                self.is_locked = False
                print("❌ Target Lost - Returning to Search Mode")

        self.prev_gray = gray
        return vis

    def _correct_drift(self, gray_frame):
        x, y = int(self.target_pt[0][0][0]), int(self.target_pt[0][0][1])
        margin = 40
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(W, x+margin), min(H, y+margin)
        roi = gray_frame[y1:y2, x1:x2]
        
        if roi.size > 0 and self.template.size > 0:
            res = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(res)
            if val > 0.7:
                new_x = loc[0] + x1 + (TEMPLATE_SIZE // 2)
                new_y = loc[1] + y1 + (TEMPLATE_SIZE // 2)
                self.target_pt = np.array([[new_x, new_y]], dtype=np.float32).reshape(-1, 1, 2)

# ==========================================
# ============ VISION ENGINE ===============
# ==========================================
class VisionEngine:
    def __init__(self):
        print("📦 Loading AI Models...")
        self.yolo = YOLO(SEG_MODEL_PATH)
        
        # Setup Regressor
        base = models.mobilenet_v3_small(weights=None)
        num_ftrs = base.classifier[3].in_features
        base.classifier[3] = nn.Linear(num_ftrs, 2)
        self.regressor = nn.Sequential(base, nn.Sigmoid()).to(DEVICE)
        self.regressor.load_state_dict(torch.load(REG_MODEL_PATH, map_location=DEVICE))
        self.regressor.eval()

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def find_plant(self, frame):
        """Runs heavy YOLO + Regressor to find a starting point."""
        results = self.yolo.predict(frame, conf=0.6, verbose=False)
        if not results[0].boxes: return None
        
        # Get primary box and run regressor
        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        crop = frame[box[1]:box[3], box[0]:box[2]]
        if crop.size == 0: return None
        
        tf = self.transform(image=crop)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.regressor(tf)[0].cpu().numpy()
        
        # Simple coordinate mapping back to global frame
        # (Using a simplified version for this test script)
        gx = box[0] + (out[0] * (box[2]-box[0]))
        gy = box[1] + (out[1] * (box[3]-box[1]))
        return (gx, gy)

# ==========================================
# ============ MAIN TEST LOOP ==============
# ==========================================
def main():
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 60)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Auto ON
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 1000) 
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual OFF
    cap.set(cv2.CAP_PROP_EXPOSURE, 200)      
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 10)
    cap.set(cv2.CAP_PROP_SATURATION, 70)

    ai_engine = VisionEngine()
    tracker = HybridTracker()

    print("🚀 System Online. Searching for plants...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        if not tracker.is_locked:
            # --- SEARCH MODE (Heavy AI) ---
            target = ai_engine.find_plant(frame)
            if target:
                tracker.lock_onto(frame, target[0], target[1])
            cv2.putText(frame, "SEARCHING...", (50, 50), 1, 2, (0, 0, 255), 2)
            display_frame = frame
        else:
            # --- LOCK MODE (Fast LK Tracking) ---
            tracked_frame = tracker.update(frame)
            display_frame = tracked_frame if tracked_frame is not None else frame
            cv2.putText(display_frame, "LOCKED", (50, 50), 1, 2, (0, 255, 0), 2)

        cv2.imshow("Hybrid AI Tracker Test", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('c'): tracker.is_locked = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()