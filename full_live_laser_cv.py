#!/usr/bin/env python3
import time
import cv2
import torch
import numpy as np
import serial
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial import distance
import platform

# Import your existing laser tools
import SCI_Weeder.b1_test.Laser_Helpers as laser

# ==========================================
# ============== CONFIGURATION =============
# ==========================================
# Hardware
LEFT_CAM_ID  = 4
RIGHT_CAM_ID = 1
SERIAL_PORT  = "/dev/ttyUSB0" 
BAUD_RATE    = 115200

# Models
SEG_MODEL_PATH = "best.pt"
REG_MODEL_PATH = "v2_hub_regressor_640.pth"
IMG_SIZE       = 640
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calibration 
mL, bL = -0.0061461658179720905, 368.03593073199704
mR, bR = -0.008112495386117758, 359.70747786328616

# Control Settings
TOLERANCE_MM = 0.5 
MAX_STEP_MM  = 10.0  
PIXELS_TO_MM = 0.01 

# ==========================================
# ============ 1. PID CONTROLLER ===========
# ==========================================
class PID:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self._prev_error = 0
        self._integral = 0
        self._last_time = time.time()

    def compute(self, measurement):
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0: dt = 1e-3

        error = self.setpoint - measurement
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        
        self._prev_error = error
        self._last_time = current_time
        return output
    
    def reset(self):
        self._prev_error = 0
        self._integral = 0
        self._last_time = time.time()

# ==========================================
# ============ 2. VISION SYSTEM ============
# ==========================================
class CameraCV:
    def __init__(self):
        print(f"\n🚀 Initializing Vision System...")
        
        # DEBUG STEP 1: CUDA CHECK
        print("   [1/6] Checking Hardware Acceleration...")
        if torch.cuda.is_available():
            print(f"         ✅ CUDA Device Found: {torch.cuda.get_device_name(0)}")
            # Simple tensor operation to warm up/test CUDA
            try:
                x = torch.ones(1).to(DEVICE)
                print("         ✅ CUDA Tensor Test Passed.")
            except Exception as e:
                print(f"         ❌ CUDA ERROR: {e}")
        else:
            print("         ⚠️ CUDA NOT FOUND (Running on CPU)")

        self.setup_models()
        
        # DEBUG STEP 4: CAMERAS
        print("   [4/6] Opening Cameras...")
        self.capL = self.setup_camera(LEFT_CAM_ID, "Left")
        self.capR = self.setup_camera(RIGHT_CAM_ID, "Right")
        
        # Internal state
        self.posL = None 
        self.posR = None 
        self.vis_frame = None 
        print("   ✅ Vision Initialization Complete.\n")

    def setup_models(self):
        # DEBUG STEP 2: YOLO
        print("   [2/6] Loading YOLO Segmentation Model...")
        try:
            self.yolo = YOLO(SEG_MODEL_PATH)
            # Force a dummy prediction to compile kernels now (prevents lag later)
            print("         ...Running YOLO warm-up inference...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.yolo.predict(dummy, verbose=False)
            print("         ✅ YOLO Loaded & Warmed Up.")
        except Exception as e:
            print(f"         ❌ ERROR Loading YOLO: {e}")
        
        # DEBUG STEP 3: REGRESSOR
        print("   [3/6] Building & Loading Regressor...")
        self.regressor = models.mobilenet_v3_small(weights=None)
        num_ftrs = self.regressor.classifier[3].in_features
        self.regressor.classifier[3] = nn.Linear(num_ftrs, 2)
        self.regressor = nn.Sequential(self.regressor, nn.Sigmoid())
        
        print(f"         ...Moving Regressor to {DEVICE}...")
        self.regressor.to(DEVICE) # THIS IS A COMMON HANG POINT
        
        try:
            print(f"         ...Loading Weights from {REG_MODEL_PATH}...")
            self.regressor.load_state_dict(torch.load(REG_MODEL_PATH, map_location=DEVICE))
            self.regressor.eval()
            print("         ✅ Regressor Ready.")
        except FileNotFoundError:
            print("         ❌ Regressor weights not found!")

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def setup_camera(self, port_id, name="Cam"):
        print(f"         ...Connecting to {name} Camera (Port {port_id})...")
        backend = cv2.CAP_V4L2 
        cap = cv2.VideoCapture(port_id, backend)
        
        if not cap.isOpened():
            print(f"         ❌ Failed to open {name} Camera!")
            return None

        # High-Speed Config
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Dark/Manual Fix
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Auto ON
        cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 1000) 
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual OFF
        cap.set(cv2.CAP_PROP_EXPOSURE, 200)      
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 10)
        cap.set(cv2.CAP_PROP_SATURATION, 70)
        
        print(f"         ✅ {name} Camera Configured.")
        return cap

    def snap_to_plant(self, pred_point, crop_image):
        x, y = int(pred_point[0]), int(pred_point[1])
        h, w = crop_image.shape[:2]
        x = np.clip(x, 0, w-1)
        y = np.clip(y, 0, h-1)

        if np.sum(crop_image[y, x]) > 10: return (x, y)

        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        pts = cv2.findNonZero(mask)
        
        if pts is None: return (x, y)
        
        pts = pts.squeeze()
        target = np.array([[x, y]])
        if len(pts.shape) == 1: pts = np.expand_dims(pts, 0)
        
        dists = distance.cdist(target, pts, 'euclidean')
        return tuple(pts[np.argmin(dists)])

    def predict_single(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.yolo.predict(frame_bgr, conf=0.5, verbose=False)
        res = results[0]
        
        if getattr(res, "masks", None) is None or len(res.boxes) == 0:
            return None, frame_bgr 

        masks = res.masks.data.cpu().numpy()
        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
        combined_mask = cv2.resize(combined_mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_img = cv2.bitwise_and(frame_rgb, frame_rgb, mask=combined_mask)

        boxes = res.boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = boxes[0].astype(int)
        
        h, w, _ = frame_rgb.shape
        px, py = int((x2-x1)*0.1), int((y2-y1)*0.1)
        x1, y1 = max(0, x1-px), max(0, y1-py)
        x2, y2 = min(w, x2+px), min(h, y2+py)
        
        crop = masked_img[y1:y2, x1:x2]
        if crop.size == 0: return None, frame_bgr

        tf = self.transform(image=crop)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.regressor(tf)[0].cpu().numpy()

        pred_x, pred_y = output[0] * IMG_SIZE, output[1] * IMG_SIZE
        
        ch, cw = crop.shape[:2]
        scale = IMG_SIZE / max(ch, cw)
        pad_w = (IMG_SIZE - cw * scale) // 2
        pad_h = (IMG_SIZE - ch * scale) // 2
        
        real_crop_x = (pred_x - pad_w) / scale
        real_crop_y = (pred_y - pad_h) / scale
        
        sx, sy = self.snap_to_plant((real_crop_x, real_crop_y), crop)
        global_x, global_y = x1 + sx, y1 + sy
        
        vis = frame_bgr.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(vis, (int(global_x), int(global_y)), 8, (0, 255, 0), -1)
        
        return (global_x, global_y), vis

    def update(self):
        retL, frameL = self.capL.read()
        retR, frameR = self.capR.read()

        if not retL or not retR:
            return False

        ptL, visL = self.predict_single(frameL)
        ptR, visR = self.predict_single(frameR)

        self.posL = ptL
        self.posR = ptR
        
        p0 = (0, int(bL))
        p1 = (1279, int(mL*1279 + bL))
        cv2.line(visL, p0, p1, (0, 0, 255), 1)

        q0 = (0, int(bR))
        q1 = (1279, int(mR*1279 + bR))
        cv2.line(visR, q0, q1, (0, 0, 255), 1)

        self.vis_frame = cv2.hconcat([visL, visR])
        return True

    def get_coordinates(self):
        return self.posL, self.posR

    def show(self):
        if self.vis_frame is not None:
            cv2.imshow("Laser Vision Hub", self.vis_frame)

    def close(self):
        if self.capL: self.capL.release()
        if self.capR: self.capR.release()
        cv2.destroyAllWindows()

# ==========================================
# ============ 3. MAIN CONTROL =============
# ==========================================
def main():
    print("🚀 STARTING MAIN CONTROLLER")
    
    # 1. Setup Systems
    cv_system = CameraCV()
    
    print("🔌 Connecting to Laser Serial...")
    try:
        ser = laser.connect() 
        time.sleep(3)
        #ser.reset_input_buffer()
        #laser.send(ser,"$H")
        laser.move_to(ser, 200, 200)
    except Exception as e:
        print(f"❌ SERIAL ERROR: {e}")
        return

    pid_x = PID(kp=0.5, ki=0.0, kd=0.05, setpoint=0) 
    pid_y = PID(kp=0.5, ki=0.0, kd=0.05, setpoint=0)

    print("\n✅ System Ready. Press SPACE to Burn, 'q' to Quit.")

    try:
        while True:
            # A. Vision Update
            if not cv_system.update():
                print("⚠️ Camera Drop")
                continue

            cv_system.show()
            
            # B. Get Target
            posL, posR = cv_system.get_coordinates()

            # C. Control Logic 
            if posL and posR:
                xL, yL = posL
                xR, yR = posR
                
                # Math
                error_x_pix = (1280 - xL) - xR 
                
                target_y_L = mL * xL + bL
                target_y_R = mR * xR + bR
                diff_L = target_y_L - yL
                diff_R = target_y_R - yR
                error_y_pix = (diff_L + diff_R) / 2.0

                move_x_raw = pid_x.compute(error_x_pix) # Removed inversion
                move_y_raw = -pid_y.compute(error_y_pix)

                dx_mm = move_x_raw * PIXELS_TO_MM
                dy_mm = move_y_raw * PIXELS_TO_MM

                dx_mm = max(min(dx_mm, MAX_STEP_MM), -MAX_STEP_MM)
                dy_mm = max(min(dy_mm, MAX_STEP_MM), -MAX_STEP_MM)

                if abs(dx_mm) < 0.05: dx_mm = 0
                if abs(dy_mm) < 0.05: dy_mm = 0
                
                print(f"eX: {error_x_pix:6.1f} -> dx: {dx_mm:6.3f} | eY: {error_y_pix:6.1f} -> dy: {dy_mm:6.3f}")

                if dx_mm != 0 or dy_mm != 0:
                    cmd = f"G91\nG1 X{dx_mm:.3f} Y{dy_mm:.3f} F3000\nG90\n"
                    ser.write(cmd.encode())
                    
                if abs(error_x_pix) < 10 and abs(error_y_pix) < 10:
                    pass 
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                print("🔥 FIRING LASER!")
                laser.burn(ser, power=800, duration=1.0)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cv_system.close()
        laser.close(ser)
        print("👋 System Closed.")

if __name__ == "__main__":
    main()