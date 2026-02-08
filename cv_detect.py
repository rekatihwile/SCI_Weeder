import time
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial import distance
import platform

# --- CONFIGURATION ---
CAMERA_LEFT = 4
CAMERA_RIGHT = 1
SEG_MODEL_PATH = "best.pt"
REG_MODEL_PATH = "v2_hub_regressor_640.pth"
IMG_SIZE = 640
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_NAME = "Dual Camera Plant Hub Detection"

# --- 1. MODEL DEFINITIONS ---
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

# --- 2. HELPER FUNCTIONS ---
def snap_to_plant(pred_point, crop_image):
    x, y = int(pred_point[0]), int(pred_point[1])
    h, w = crop_image.shape[:2]
    x = np.clip(x, 0, w-1)
    y = np.clip(y, 0, h-1)

    if np.sum(crop_image[y, x]) > 10:
        return (x, y)

    gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    plant_pixels = cv2.findNonZero(mask)

    if plant_pixels is None:
        return (x, y)

    plant_pixels = plant_pixels.squeeze()
    target = np.array([[x, y]])
    dists = distance.cdist(target, plant_pixels, 'euclidean')
    return tuple(plant_pixels[np.argmin(dists)])

def setup_camera(port_id):
    """Initializes a camera with the specific user settings."""
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(port_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(port_id, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera on port {port_id}.")
        return None

    # Force MJPG for high bandwidth
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Manual Settings
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 1000) # Neutral daylight
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode trigger
    cap.set(cv2.CAP_PROP_EXPOSURE, 200)        # Specific exposure
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
    cap.set(cv2.CAP_PROP_SATURATION, 70)       # Specific saturation
    
    return cap

def process_frame(frame_bgr, yolo_model, reg_model, label_prefix=""):
    """Runs the full detection pipeline on a single frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    vis_bgr = frame_bgr.copy()
    
    t0 = time.perf_counter()
    
    # 1. Segmentation
    results = yolo_model.predict(frame_bgr, conf=0.5, verbose=False)
    
    detection_successful = False
    global_x, global_y = 0, 0
    res0 = results[0]

    if getattr(res0, "masks", None) is not None and len(res0.boxes) > 0:
        detection_successful = True
        
        # 2. Crop & Mask
        masks = res0.masks.data.cpu().numpy()
        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
        combined_mask = cv2.resize(combined_mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_img = cv2.bitwise_and(frame_rgb, frame_rgb, mask=combined_mask)

        boxes = res0.boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = boxes[0].astype(int)

        h, w, _ = frame_rgb.shape
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        crop = masked_img[y1:y2, x1:x2].copy()

        if crop.size != 0:
            # 3. Preprocess
            transformed = reg_transform(image=crop)["image"].unsqueeze(0).to(DEVICE)

            # 4. Inference
            with torch.no_grad():
                output = reg_model(transformed)[0].cpu().numpy()

            # 5. Postprocess
            pred_x, pred_y = output[0] * IMG_SIZE, output[1] * IMG_SIZE
            ch, cw = crop.shape[:2]
            scale = IMG_SIZE / max(ch, cw)
            pad_w = (IMG_SIZE - cw * scale) // 2
            pad_h = (IMG_SIZE - ch * scale) // 2

            real_crop_x = (pred_x - pad_w) / scale
            real_crop_y = (pred_y - pad_h) / scale

            snapped_x, snapped_y = snap_to_plant((real_crop_x, real_crop_y), crop)

            global_x = x1 + snapped_x
            global_y = y1 + snapped_y
        else:
            detection_successful = False

    pipeline_time = (time.perf_counter() - t0) * 1000

    # Draw Logic
    if detection_successful:
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(vis_bgr, (int(global_x), int(global_y)), 10, (0, 255, 0), -1)
        cv2.circle(vis_bgr, (int(global_x), int(global_y)), 12, (0, 0, 0), 2)
        status = f"Hub: {int(global_x)},{int(global_y)}"
    else:
        status = "No Detection"

    # Overlay Text
    cv2.putText(vis_bgr, f"{label_prefix} | {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_bgr, f"Lat: {pipeline_time:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    return vis_bgr

# --- 3. LIVE DETECTION MAIN FUNCTION ---
def run_live_detection():
    print(f"🚀 Loading models on {DEVICE}...")

    # Load Models
    yolo = YOLO(SEG_MODEL_PATH)
    regressor = get_regressor().to(DEVICE)
    try:
        regressor.load_state_dict(torch.load(REG_MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"❌ Error: Regressor model not found at {REG_MODEL_PATH}")
        return
    regressor.eval()
    
    # Initialize Cameras
    print(f"📷 Opening Left Camera (Port {CAMERA_LEFT})...")
    cap_l = setup_camera(CAMERA_LEFT)
    print(f"📷 Opening Right Camera (Port {CAMERA_RIGHT})...")
    cap_r = setup_camera(CAMERA_RIGHT)

    if cap_l is None or cap_r is None:
        print("❌ Critical Error: One or both cameras failed to open.")
        if cap_l: cap_l.release()
        if cap_r: cap_r.release()
        return

    print("▶️ Starting dual live detection. Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Read Both Frames
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()

            if not ret_l or not ret_r:
                print("⚠️ Warning: Drop frame.")
                continue
            
            # Process Both Frames
            # Note: Doing this sequentially doubles latency. 
            # For true parallel performance, we would need python threading, but this is simpler.
            vis_l = process_frame(frame_l, yolo, regressor, label_prefix="LEFT")
            vis_r = process_frame(frame_r, yolo, regressor, label_prefix="RIGHT")
            
            # Stack images side-by-side
            combined_vis = cv2.hconcat([vis_l, vis_r])
            
            # Calculate Total FPS
            frame_count += 1
            if frame_count >= 10:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                start_time = time.time()
                frame_count = 0

            # Draw FPS on the combined image (Top center)
            cv2.putText(combined_vis, f"SYS FPS: {fps:.1f}", (combined_vis.shape[1]//2 - 50, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, combined_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()
        print("👋 Exiting.")

if __name__ == "__main__":
    run_live_detection()