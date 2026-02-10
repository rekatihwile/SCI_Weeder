#!/usr/bin/env python3
import time
import cv2
import torch
import numpy as np
import threading
from collections import deque
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURATION ---
W, H = 1280, 720
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIL_LENGTH = 15  # Length of the visual trail

# winSize 31 handles rapid movement better
LK_PARAMS = dict(winSize=(31, 31), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ==========================================
# ============ 1. CAMERA THREAD ============
# ==========================================
class CameraStream:
    def __init__(self, src=0):
        # CAP_V4L2 is crucial for speed on Jetson
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# ==========================================
# ============ 2. AI THREAD ================
# ==========================================
class AIWorker(threading.Thread):
    def __init__(self, seg_path, reg_path):
        super().__init__(daemon=True)
        self.yolo = YOLO(seg_path)
        # Load Regressor
        base = models.mobilenet_v3_small(weights=None)
        num_ftrs = base.classifier[3].in_features
        base.classifier[3] = nn.Linear(num_ftrs, 2)
        self.regressor = nn.Sequential(base, nn.Sigmoid()).to(DEVICE)
        self.regressor.load_state_dict(torch.load(reg_path, map_location=DEVICE))
        self.regressor.eval()
        
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        self.frame_to_proc = None
        self.result = None
        self.new_data = False
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            img = None
            with self.lock:
                if self.frame_to_proc is not None:
                    img = self.frame_to_proc.copy()
                    self.frame_to_proc = None
            
            if img is not None:
                res = self.yolo.predict(img, conf=0.6, verbose=False)
                if res[0].boxes:
                    box = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
                    crop = img[box[1]:box[3], box[0]:box[2]]
                    if crop.size != 0:
                        tf = self.transform(image=crop)["image"].unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            out = self.regressor(tf)[0].cpu().numpy()
                        gx = box[0] + (out[0] * (box[2]-box[0]))
                        gy = box[1] + (out[1] * (box[3]-box[1]))
                        with self.lock:
                            self.result = (gx, gy)
                            self.new_data = True
            else:
                time.sleep(0.01)

# ==========================================
# ============ 3. MAIN LOOP ================
# ==========================================
def main():
    vs = CameraStream(src=0).start()
    ai = AIWorker("best.pt", "v2_hub_regressor_640.pth")
    ai.start()
    
    target_pt = None
    prev_gray = None
    trail = deque(maxlen=TRAIL_LENGTH)
    
    fps_start_time = time.time()
    fps_counter = 0
    fps_display = 0

    print("🚀 High-Speed Threaded Pipeline Active.")

    while True:
        frame = vs.read()
        if frame is None: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # Update AI with current frame if it's ready
        if not ai.new_data:
            with ai.lock:
                ai.frame_to_proc = frame

        # Check for AI Snap (Ground Truth)
        if ai.new_data:
            with ai.lock:
                gx, gy = ai.result
                target_pt = np.array([[gx, gy]], dtype=np.float32).reshape(-1, 1, 2)
                ai.new_data = False
            # Clear trail on new AI snap to prevent old "ghost" trails
            trail.clear()

        # LK Multi-Point (Filling the Gaps)
        if target_pt is not None and prev_gray is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, target_pt, None, **LK_PARAMS)
            if status[0][0] == 1:
                target_pt = new_pts
                tx, ty = target_pt.ravel()
                trail.append((int(tx), int(ty)))
            else:
                target_pt = None

        # --- VISUALS ---
        # Draw Trail
        for i in range(1, len(trail)):
            thickness = int(np.sqrt(TRAIL_LENGTH / float(i + 1)) * 2.5)
            cv2.line(display_frame, trail[i - 1], trail[i], (0, 255, 255), thickness)

        if target_pt is not None:
            tx, ty = target_pt.ravel()
            cv2.circle(display_frame, (int(tx), int(ty)), 10, (0, 255, 0), -1)
            cv2.drawMarker(display_frame, (int(tx), int(ty)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # FPS Stats
        fps_counter += 1
        if (time.time() - fps_start_time) > 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        cv2.putText(display_frame, f"FPS: {fps_display}", (W-180, 50), 1, 2, (0, 255, 0), 2)
        cv2.imshow("Nitro Hybrid Tracker", display_frame)

        prev_gray = gray
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()