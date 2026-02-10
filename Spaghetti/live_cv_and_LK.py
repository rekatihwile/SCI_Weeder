#!/usr/bin/env python3
import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial import distance

# ==========================================
# ============== CONFIGURATION =============
# ==========================================
CAMERA_PORT = 0                          
SEG_MODEL_PATH = "best.pt"               
REG_MODEL_PATH = "v2_hub_regressor_640.pth" 
IMG_SIZE = 640
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stability Logic
STABILITY_TIME = 1.0     # 1 second of confirmed position
STABILITY_THRESHOLD = 20  # Max pixel jitter allowed during stabilization

# LK Parameters (Matched to your reference)
LK_PARAMS = dict(winSize=(21, 21), 
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ==========================================
# ============ 1. MODEL SETUP ==============
# ==========================================
def get_regressor():
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model = nn.Sequential(model, nn.Sigmoid())
    return model

reg_transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def snap_to_plant(pred_point, crop_image):
    x, y = int(pred_point[0]), int(pred_point[1])
    h, w = crop_image.shape[:2]
    x, y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
    if np.sum(crop_image[y, x]) > 10: return (x, y)
    gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    plant_pixels = cv2.findNonZero(mask)
    if plant_pixels is None: return (x, y)
    plant_pixels = plant_pixels.squeeze()
    target = np.array([[x, y]])
    if len(plant_pixels.shape) == 1: plant_pixels = np.expand_dims(plant_pixels, 0)
    dists = distance.cdist(target, plant_pixels, 'euclidean')
    return tuple(plant_pixels[np.argmin(dists)])

# ==========================================
# ============ 2. TRACKER CLASS ============
# ==========================================
class PlantTracker:
    def __init__(self):
        self.yolo = YOLO(SEG_MODEL_PATH)
        self.regressor = get_regressor().to(DEVICE)
        self.regressor.load_state_dict(torch.load(REG_MODEL_PATH, map_location=DEVICE))
        self.regressor.eval()
        
        self.state = "SEARCHING" # SEARCHING, STABILIZING, LOCKED
        self.stable_start_time = None
        self.points_buffer = []
        self.target_pt = None   
        self.prev_gray = None

    def run_inference(self, frame_bgr, frame_rgb):
        results = self.yolo.predict(frame_bgr, conf=0.5, verbose=False)
        res0 = results[0]

        if getattr(res0, "masks", None) is not None and len(res0.boxes) > 0:
            masks = res0.masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            combined_mask = cv2.resize(combined_mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
            masked_img = cv2.bitwise_and(frame_rgb, frame_rgb, mask=combined_mask)

            boxes = res0.boxes.xyxy.cpu().numpy()
            x1, y1, x2, y2 = boxes[0].astype(int)
            h, w, _ = frame_rgb.shape
            px, py = int((x2-x1)*0.1), int((y2-y1)*0.1)
            x1, y1 = max(0, x1-px), max(0, y1-py)
            x2, y2 = min(w, x2+px), min(h, y2+py)

            crop = masked_img[y1:y2, x1:x2]
            if crop.size != 0:
                transformed = reg_transform(image=crop)["image"].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = self.regressor(transformed)[0].cpu().numpy()

                pred_x, pred_y = output[0] * IMG_SIZE, output[1] * IMG_SIZE
                ch, cw = crop.shape[:2]
                scale = IMG_SIZE / max(ch, cw)
                pad_w, pad_h = (IMG_SIZE - cw * scale) // 2, (IMG_SIZE - ch * scale) // 2
                rx, ry = (pred_x - pad_w) / scale, (pred_y - pad_h) / scale
                
                sx, sy = snap_to_plant((rx, ry), crop)
                return (float(x1 + sx), float(y1 + sy)), (x1, y1, x2, y2)
        return None, None

    def update(self, frame_bgr):
        vis = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.state == "LOCKED":
            if self.prev_gray is not None and self.target_pt is not None:
                # LK movement calculation
                new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.target_pt, None, **LK_PARAMS
                )

                if status[0][0] == 1:
                    self.target_pt = new_pts
                    x, y = new_pts.ravel()
                    cv2.circle(vis, (int(x), int(y)), 10, (0, 255, 0), -1)
                    cv2.putText(vis, "LK LOCKED", (int(x)+15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    print("❌ LK Slip - Returning to Search")
                    self.state = "SEARCHING"
                    self.target_pt = None

        else:
            # Running heavy AI inference
            found_pt, box = self.run_inference(frame_bgr, frame_rgb)
            if found_pt:
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 1)
                cv2.drawMarker(vis, (int(found_pt[0]), int(found_pt[1])), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

                if self.state == "SEARCHING":
                    self.state = "STABILIZING"
                    self.stable_start_time = time.time()
                    self.points_buffer = [found_pt]
                
                elif self.state == "STABILIZING":
                    dist = np.linalg.norm(np.array(found_pt) - self.points_buffer[0])
                    if dist < STABILITY_THRESHOLD:
                        self.points_buffer.append(found_pt)
                        elapsed = time.time() - self.stable_start_time
                        progress = int((elapsed / STABILITY_TIME) * 100)
                        cv2.putText(vis, f"STABILIZING: {progress}%", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        if elapsed >= STABILITY_TIME:
                            avg_pt = np.mean(self.points_buffer, axis=0)
                            # Convert to shape (1, 1, 2) float32 for LK consistency
                            self.target_pt = np.array([[avg_pt]], dtype=np.float32)
                            self.state = "LOCKED"
                            print(f"🎯 Locked Target at {avg_pt}")
                    else:
                        self.state = "SEARCHING"
            else:
                self.state = "SEARCHING"

        # IMPORTANT: Always update prev_gray to keep LK relative movement accurate
        self.prev_gray = gray
        return vis

# ==========================================
# ============ 3. MAIN LOOP ================
# ==========================================
def main():
    cap = cv2.VideoCapture(CAMERA_PORT, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Matching your provided reference settings
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    cap.set(cv2.CAP_PROP_EXPOSURE, 550)
    cap.set(cv2.CAP_PROP_GAIN, 44)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
    cap.set(cv2.CAP_PROP_CONTRAST, 40)
    cap.set(cv2.CAP_PROP_SATURATION, 63)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)            
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4079)
    
    tracker = PlantTracker()
    print("🚀 Hybrid Engine Active. Press 'q' to quit, 'r' to reset search.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        start = time.perf_counter()
        out = tracker.update(frame)
        ms = (time.perf_counter() - start) * 1000
        
        cv2.putText(out, f"STATE: {tracker.state} | {ms:.1f}ms", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Hybrid AI Tracker", out)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            tracker.state = "SEARCHING"
            tracker.target_pt = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()