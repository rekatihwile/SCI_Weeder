import cv2
import json
import platform
import numpy as np
from pathlib import Path

# Local imports
from cv_helpers import WeedCV

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
YOLO_PT = str(WEIGHTS_DIR / "yolo_w_kale.pt")
SNIPER_PT = str(WEIGHTS_DIR / "sniper.pt")   
CAMERA_SETTINGS = Path("/home/laser/Documents/Laser_Workspace/SCI_Weeder/Brian_UNR/camera_config.json")
HARDWARE_CONFIG = BASE_DIR / "hardware_config.json"

# --- SYSTEM CONSTANTS ---
IS_WINDOWS = platform.system() == "Windows"
CAM_BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2
W, H = 640, 480

class VisionTester:
    def __init__(self):
        print("🔍 Loading Configuration...")
        self.load_hardware_config()
        self.load_camera_tunings()

        # Initialize the CV model
        self.cv_L = WeedCV(YOLO_PT, SNIPER_PT)
        
        print(f"📷 Initializing Left Camera (ID: {self.left_id})...")
        self.cap_L = cv2.VideoCapture(self.left_id, CAM_BACKEND)
        self.cap_L.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        
        self.apply_nuclear_hardware_lock()

    def load_hardware_config(self):
        with open(HARDWARE_CONFIG, "r") as f:
            cfg = json.load(f)
        self.left_id = cfg["cameras"]["left"]["index"]

    def load_camera_tunings(self):
        with open(CAMERA_SETTINGS, "r") as f:
            self.cam_cfg = json.load(f)

    def apply_nuclear_hardware_lock(self):
        s = self.cam_cfg["left"]
        self.cap_L.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        self.cap_L.set(cv2.CAP_PROP_BRIGHTNESS, s['brightness'])
        self.cap_L.set(cv2.CAP_PROP_CONTRAST, s['contrast'])
        self.cap_L.set(cv2.CAP_PROP_EXPOSURE, s['exposure'])
        self.cap_L.set(cv2.CAP_PROP_GAIN, s['gain'])
        self.cap_L.set(cv2.CAP_PROP_SATURATION, s['saturation'])
        self.cap_L.set(cv2.CAP_PROP_SHARPNESS, s['sharpness'])
        self.cap_L.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap_L.set(cv2.CAP_PROP_WB_TEMPERATURE, s['white_balance'])
        for _ in range(5): self.cap_L.grab()

    def draw_boxes(self, img, boxes, color, base_label):
        for box in boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].item())
            label_with_conf = f"{base_label} {conf:.2f}"
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 2)
            cv2.putText(img, label_with_conf, (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run_test(self):
        print("\n🚀 Starting Real-Time Vision Test...")
        print("Press 'q' to quit.")
        
        cv2.namedWindow("Kale Safety Test (2x2 Grid)", cv2.WINDOW_NORMAL)
        
        # CHANGED: We now store pre-processed coordinate data, not raw images
        burst_history = [] 
        stable_targets = []
        
        while True:
            self.cap_L.grab()
            ret, frame = self.cap_L.read()
            if not ret:
                print("⚠️ Failed to grab frame.")
                break

            # 1. Run bounding box detections
            boxes, masks = self.cv_L._get_detections(
                frame, weed_classes=[0, 2], kale_class=1, 
                kale_thresh=0.05, weed_conf=0.40, kale_conf=1.0
            )

            # 2. Get live per-frame keypoints (for Panel 3)
            live_coords = self.cv_L.return_full(frame, precalc_boxes=boxes, precalc_masks=masks)

            # 3. CHANGED: Format this frame's data and add to our rolling history
            current_frame_data = []
            for i, box in enumerate(self.cv_L.filtered_boxes):
                if i < len(live_coords):
                    current_frame_data.append({
                        'box': box.xyxy[0].cpu().numpy(),
                        'point': live_coords[i]
                    })
            
            burst_history.append(current_frame_data)

            # 4. CHANGED: Every 10 frames, run the pure math function
            if len(burst_history) == 10:
                # Make sure calculate_burst_math is added to your weed_cv.py!
                stable_targets = self.cv_L.calculate_burst_math(burst_history)
                burst_history.clear()

            # Set up the 4 canvases
            img_before = frame.copy()
            img_after = frame.copy()
            img_live = frame.copy()
            img_burst = frame.copy()

            # --- PANEL 1 (Top Left): BEFORE (Raw Weeds + Kale) ---
            if hasattr(self.cv_L, 'kale_boxes'):
                self.draw_boxes(img_before, self.cv_L.kale_boxes, (0, 255, 0), "KALE")
            if hasattr(self.cv_L, 'raw_weed_boxes'):
                self.draw_boxes(img_before, self.cv_L.raw_weed_boxes, (0, 0, 255), "WEED (RAW)")
            cv2.putText(img_before, "1. BEFORE SAFETY CHECK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --- PANEL 2 (Top Right): AFTER (Safe Weeds + Kale) ---
            if hasattr(self.cv_L, 'kale_boxes'):
                self.draw_boxes(img_after, self.cv_L.kale_boxes, (0, 255, 0), "KALE")
            if hasattr(self.cv_L, 'filtered_boxes'):
                self.draw_boxes(img_after, self.cv_L.filtered_boxes, (255, 100, 0), "WEED (SAFE)")
            cv2.putText(img_after, "2. AFTER SAFETY CHECK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

            # --- PANEL 3 (Bottom Left): LIVE DOTS (Per-Frame Keypoints) ---
            if hasattr(self.cv_L, 'filtered_boxes'):
                self.draw_boxes(img_live, self.cv_L.filtered_boxes, (255, 100, 0), "TARGET BOX")
            for (x, y) in live_coords:
                cv2.circle(img_live, (x, y), 8, (0, 0, 255), -1) # Red live dots
                cv2.drawMarker(img_live, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img_live, f"3. LIVE DOTS ({len(live_coords)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --- PANEL 4 (Bottom Right): BURST DOTS (Stabilized Medians) ---
            if hasattr(self.cv_L, 'filtered_boxes'):
                self.draw_boxes(img_burst, self.cv_L.filtered_boxes, (255, 100, 0), "TARGET BOX")
            for (x, y) in stable_targets:
                cv2.circle(img_burst, (x, y), 8, (0, 255, 0), -1) # Green stabilized dots
                cv2.drawMarker(img_burst, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img_burst, f"4. BURST STABLE ({len(stable_targets)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Stack into a 2x2 grid
            top_row = np.hstack((img_before, img_after))
            bottom_row = np.hstack((img_live, img_burst))
            combined = np.vstack((top_row, bottom_row))
            
            cv2.imshow("Kale Safety Test (2x2 Grid)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap_L.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = VisionTester()
    tester.run_test()