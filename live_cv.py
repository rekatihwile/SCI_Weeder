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
WINDOW_NAME = "AI Stress Test: YOLO + CNN"

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

# ==========================================
# ============ 2. UTILITIES ================
# ==========================================
def snap_to_plant(pred_point, crop_image):
    x, y = int(pred_point[0]), int(pred_point[1])
    h, w = crop_image.shape[:2]
    x, y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)

    if np.sum(crop_image[y, x]) > 10:
        return (x, y)

    gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    plant_pixels = cv2.findNonZero(mask)

    if plant_pixels is None:
        return (x, y)

    plant_pixels = plant_pixels.squeeze()
    target = np.array([[x, y]])
    if len(plant_pixels.shape) == 1: plant_pixels = np.expand_dims(plant_pixels, 0)
    
    dists = distance.cdist(target, plant_pixels, 'euclidean')
    return tuple(plant_pixels[np.argmin(dists)])

# ==========================================
# ============ 3. MAIN LOOP ================
# ==========================================
def run_live_inference():
    print(f"🚀 Initializing AI Engine on {DEVICE}...")

    # Load Models
    yolo = YOLO(SEG_MODEL_PATH)
    regressor = get_regressor().to(DEVICE)
    regressor.load_state_dict(torch.load(REG_MODEL_PATH, map_location=DEVICE))
    regressor.eval()
    
    # Setup Camera 1
    cap = cv2.VideoCapture(CAMERA_PORT, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Hardcoded Camera Settings
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      # Manual Mode
    cap.set(cv2.CAP_PROP_EXPOSURE, 600)
    cap.set(cv2.CAP_PROP_GAIN, 50)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 60)
    cap.set(cv2.CAP_PROP_CONTRAST, 28)
    cap.set(cv2.CAP_PROP_SATURATION, 63)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)            # Manual WB
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4079)

    print("▶️ Starting Full Inference Loop. Press 'q' to quit.")

    # FPS Calculation
    fps_avg = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, orig_bgr = cap.read()
            if not ret: break
            
            loop_start = time.perf_counter()
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            
            # --- STEP 1: YOLO Segmentation ---
            results = yolo.predict(orig_bgr, conf=0.5, verbose=False)
            res0 = results[0]

            detection_successful = False
            if getattr(res0, "masks", None) is not None and len(res0.boxes) > 0:
                detection_successful = True
                
                # --- STEP 2: CROP & MASK ---
                masks = res0.masks.data.cpu().numpy()
                combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                combined_mask = cv2.resize(combined_mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
                masked_img = cv2.bitwise_and(orig_rgb, orig_rgb, mask=combined_mask)

                boxes = res0.boxes.xyxy.cpu().numpy()
                x1, y1, x2, y2 = boxes[0].astype(int)

                # Padding Logic
                h, w, _ = orig_rgb.shape
                px, py = int((x2-x1)*0.1), int((y2-y1)*0.1)
                x1, y1 = max(0, x1-px), max(0, y1-py)
                x2, y2 = min(w, x2+px), min(h, y2+py)

                crop = masked_img[y1:y2, x1:x2]
                
                if crop.size != 0:
                    # --- STEP 3: REGRESSOR ---
                    transformed = reg_transform(image=crop)["image"].unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        output = regressor(transformed)[0].cpu().numpy()

                    # Coordinate Mapping
                    pred_x, pred_y = output[0] * IMG_SIZE, output[1] * IMG_SIZE
                    ch, cw = crop.shape[:2]
                    scale = IMG_SIZE / max(ch, cw)
                    pad_w, pad_h = (IMG_SIZE - cw * scale) // 2, (IMG_SIZE - ch * scale) // 2

                    rx = (pred_x - pad_w) / scale
                    ry = (pred_y - pad_h) / scale

                    sx, sy = snap_to_plant((rx, ry), crop)
                    gx, gy = x1 + sx, y1 + sy

                    # DRAWING
                    cv2.rectangle(orig_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(orig_bgr, (int(gx), int(gy)), 10, (0, 255, 0), -1)

            # Performance Display
            loop_time = (time.perf_counter() - loop_start) * 1000
            frame_count += 1
            if frame_count >= 5:
                fps_avg = frame_count / (time.time() - start_time)
                start_time = time.time()
                frame_count = 0

            cv2.putText(orig_bgr, f"FPS: {fps_avg:.1f} | Latency: {loop_time:.1f}ms", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow(WINDOW_NAME, orig_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inference()