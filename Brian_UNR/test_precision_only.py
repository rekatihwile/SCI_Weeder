import cv2
import numpy as np
import time
import sys
import json
import select
import platform
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
TARGET_Y_L, TARGET_Y_R = 240, 240
START_POS = (225, 220)

class PrecisionTuner:
    def __init__(self):
        print("🔍 Loading Tuning Configuration...")
        self.load_hardware_config()
        self.load_camera_tunings()

        # Initialize CV purely
        self.cv_L = WeedCV(YOLO_PT, SNIPER_PT)
        self.cv_R = WeedCV(YOLO_PT, SNIPER_PT)

        self.cap_L = cv2.VideoCapture(self.left_id, CAM_BACKEND)
        self.cap_R = cv2.VideoCapture(self.right_id, CAM_BACKEND)

        for cap in [self.cap_L, self.cap_R]:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        time.sleep(1.0)
        self.clicks_L, self.clicks_R, self.lk_targets = [], [], {}
        self.path_order = [] 
        self.last_survey_fL, self.last_survey_fR = None, None
        self.apply_nuclear_hardware_lock()

    def load_hardware_config(self):
        with open(HARDWARE_CONFIG, "r") as f:
            cfg = json.load(f)
        self.left_id = cfg["cameras"]["left"]["index"]
        self.right_id = cfg["cameras"]["right"]["index"]

    def load_camera_tunings(self):
        with open(CAMERA_SETTINGS, "r") as f:
            self.cam_cfg = json.load(f)

    def apply_nuclear_hardware_lock(self):
        for side, cap in [("left", self.cap_L), ("right", self.cap_R)]:
            s = self.cam_cfg[side]
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
            cap.set(cv2.CAP_PROP_BRIGHTNESS, s['brightness'])
            cap.set(cv2.CAP_PROP_CONTRAST, s['contrast'])
            cap.set(cv2.CAP_PROP_EXPOSURE, s['exposure'])
            cap.set(cv2.CAP_PROP_GAIN, s['gain'])
            cap.set(cv2.CAP_PROP_SATURATION, s['saturation'])
            cap.set(cv2.CAP_PROP_SHARPNESS, s['sharpness'])
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, s['white_balance'])
            for _ in range(5): cap.grab()

    def get_cluster(self, x, y):
        pts = [[float(x + i), float(y + j)] for i in [-2, 0, 2] for j in [-2, 0, 2]]
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def match_and_optimize(self):
        if not (self.clicks_L and self.clicks_R): return False
        
        pts_l, pts_r = np.array(self.clicks_L), np.array(self.clicks_R)
        best_matches, max_matches = {}, 0
        for i, a_l in enumerate(pts_l):
            for j, a_r in enumerate(pts_r):
                dx, dy = a_r[0] - a_l[0], a_r[1] - a_l[1]
                if not (30 < abs(dx) < 350): continue 
                cfg, shifted_l = {}, pts_l + [dx, dy]
                for l_idx, p_s in enumerate(shifted_l):
                    dists = np.linalg.norm(pts_r - p_s, axis=1)
                    r_idx = np.argmin(dists)
                    if dists[r_idx] < 20: cfg[l_idx] = r_idx
                if len(cfg) > max_matches: max_matches = len(cfg); best_matches = cfg
        
        matched = {}
        for m_id, (l_idx, r_idx) in enumerate(best_matches.items()):
            p_l, p_r = pts_l[l_idx], pts_r[r_idx]
            matched[m_id] = {'l': self.get_cluster(*p_l), 'r': self.get_cluster(*p_r), 'orig_l': p_l}
        
        self.lk_targets = matched
        self.path_order = list(matched.keys())
        print(f"\n✅ Stereo Matching Complete. Targets to Test: {len(self.path_order)}")
        return True

    def get_roi(self, img, cx, cy, size):
        h, w = img.shape[:2]
        x1, y1 = int(np.clip(cx-size, 0, w)), int(np.clip(cy-size, 0, h))
        x2, y2 = int(np.clip(cx+size, 0, w)), int(np.clip(cy+size, 0, h))
        return img[y1:y2, x1:x2], x1, y1

    def show_overview(self):
        """Displays the full L/R frames with matched targets labeled."""
        if self.last_survey_fL is None or self.last_survey_fR is None: return
        
        img_L, img_R = self.last_survey_fL.copy(), self.last_survey_fR.copy()

        for tid in self.path_order:
            t = self.lk_targets[tid]
            lk_L = np.median(t['l'].reshape(-1, 2), axis=0).astype(int)
            lk_R = np.median(t['r'].reshape(-1, 2), axis=0).astype(int)

            cv2.circle(img_L, tuple(lk_L), 8, (255, 100, 0), -1)
            cv2.putText(img_L, f"ID: {tid}", (lk_L[0]+10, lk_L[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.circle(img_R, tuple(lk_R), 8, (255, 100, 0), -1)
            cv2.putText(img_R, f"ID: {tid}", (lk_R[0]+10, lk_R[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = np.hstack((img_L, img_R))
        cv2.namedWindow("Survey Overview Map", cv2.WINDOW_NORMAL)
        cv2.imshow("Survey Overview Map", combined)
        print("\n🗺️ OVERVIEW MAP GENERATED.")
        print("👉 Press ANY KEY on the image window to close it and begin fine-tuning...")
        cv2.waitKey(0)
        cv2.destroyWindow("Survey Overview Map")

    def test_precision_strikes(self):
        print("\n" + "="*60)
        print("🎯 INITIATING DIAGNOSTIC PRECISION STRIKE TUNING")
        print("="*60)

        # Let's set this back to a reasonable sniper crop (e.g., 160x160 pixels)
        crop_s = 80
        
        # ---------------------------------------------------------
        # 🎛️ TUNE THESE PARAMETERS HERE
        WEED_CONF = 0.10
        KALE_CONF = 0.80  
        # ---------------------------------------------------------

        print(f"Current Tuning -> WEED_CONF: {WEED_CONF} | KALE_CONF: {KALE_CONF}")

        for tid in self.path_order:
            print(f"\n\n--- TARGET ID: {tid} ---")
            t_data = self.lk_targets[tid]
            
            lk_L = np.median(t_data['l'].reshape(-1, 2), axis=0).astype(int)
            lk_R = np.median(t_data['r'].reshape(-1, 2), axis=0).astype(int)
            
            valid_L, valid_R = [], []
            kale_veto_count = 0

            print("📸 Executing 10-Frame Fine-Tune Sequence...")

            for sample in range(10):
                self.cap_L.grab(); self.cap_R.grab()
                retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
                if not (retL and retR): continue

                roi_L, offX_L, offY_L = self.get_roi(fL, lk_L[0], lk_L[1], crop_s)
                roi_R, offX_R, offY_R = self.get_roi(fR, lk_R[0], lk_R[1], crop_s)

                rel_lk_L = (lk_L[0] - offX_L, lk_L[1] - offY_L)
                rel_lk_R = (lk_R[0] - offX_R, lk_R[1] - offY_R)

                boxes_L, masks_L = self.cv_L._get_detections(roi_L, weed_classes=[0, 2], 
                                                             kale_class=1, kale_thresh=0.05, 
                                                             weed_conf=WEED_CONF, kale_conf=KALE_CONF)
                boxes_R, masks_R = self.cv_R._get_detections(roi_R, weed_classes=[0, 2], 
                                                             kale_class=1, kale_thresh=0.05, 
                                                             weed_conf=WEED_CONF, kale_conf=KALE_CONF)

                # 1. The Kale Veto Check
                kale_found = False
                for side_cv, rel_lk in [(self.cv_L, rel_lk_L), (self.cv_R, rel_lk_R)]:
                    if hasattr(side_cv, 'kale_boxes'):
                        for k_box in side_cv.kale_boxes:
                            bk = k_box.xyxy[0].cpu().numpy()
                            if (bk[0] <= rel_lk[0] <= bk[2]) and (bk[1] <= rel_lk[1] <= bk[3]):
                                kale_found = True
                                break

                # 2. Extract keypoints if safe
                pts_L, pts_R = [], []
                frame_val_L, frame_val_R = None, None
                
                if not kale_found:
                    pts_L = self.cv_L.return_full(roi_L, precalc_boxes=boxes_L, precalc_masks=masks_L)
                    pts_R = self.cv_R.return_full(roi_R, precalc_boxes=boxes_R, precalc_masks=masks_R)

                    for j, box in enumerate(self.cv_L.filtered_boxes):
                        b = box.xyxy[0].cpu().numpy()
                        if (b[0] <= rel_lk_L[0] <= b[2]) and (b[1] <= rel_lk_L[1] <= b[3]):
                            if j < len(pts_L):
                                frame_val_L = np.array([offX_L + pts_L[j][0], offY_L + pts_L[j][1]])
                                valid_L.append(frame_val_L)
                            break
                    
                    for j, box in enumerate(self.cv_R.filtered_boxes):
                        b = box.xyxy[0].cpu().numpy()
                        if (b[0] <= rel_lk_R[0] <= b[2]) and (b[1] <= rel_lk_R[1] <= b[3]):
                            if j < len(pts_R):
                                frame_val_R = np.array([offX_R + pts_R[j][0], offY_R + pts_R[j][1]])
                                valid_R.append(frame_val_R)
                            break

                # --- 3. RENDERING THE DIAGNOSTIC CROPS ---
                draw_L, draw_R = roi_L.copy(), roi_R.copy()

                for side_cv, draw_img in [(self.cv_L, draw_L), (self.cv_R, draw_R)]:
                    if hasattr(side_cv, 'kale_boxes'):
                        for box in side_cv.kale_boxes:
                            b = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(draw_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                            cv2.putText(draw_img, "KALE", (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                cv2.drawMarker(draw_L, (int(rel_lk_L[0]), int(rel_lk_L[1])), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.drawMarker(draw_R, (int(rel_lk_R[0]), int(rel_lk_R[1])), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)

                if frame_val_L is not None:
                    pt_L = (int(frame_val_L[0] - offX_L), int(frame_val_L[1] - offY_L))
                    cv2.circle(draw_L, pt_L, 4, (0, 0, 255), -1)
                
                if frame_val_R is not None:
                    pt_R = (int(frame_val_R[0] - offX_R), int(frame_val_R[1] - offY_R))
                    cv2.circle(draw_R, pt_R, 4, (0, 0, 255), -1)

                if kale_found:
                    cv2.putText(draw_L, "VETOED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(draw_R, "VETOED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    kale_veto_count += 1
                    print(f"   [Frame {sample+1:02d}] 🛑 KALE VETO! (Crop unsafe)")
                else:
                    print(f"   [Frame {sample+1:02d}] ✅ Safe. Points extracted -> L: {'Found' if frame_val_L is not None else 'Miss'}, R: {'Found' if frame_val_R is not None else 'Miss'}")

                # Force UI sizing to prevent numpy crash on edge-case crops
                draw_L_safe = cv2.resize(draw_L, (320, 320))
                draw_R_safe = cv2.resize(draw_R, (320, 320))

                combined_crop = np.hstack((draw_L_safe, draw_R_safe))
                win_name = f"Target {tid} - Sniper Scope"
                cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(win_name, combined_crop)
                
                cv2.waitKey(200) 

            # --- THE FINAL ULTIMATUM ---
            print("\n   ==> FINAL ULTIMATUM FOR TARGET:")
            if kale_veto_count >= 3:
                print(f"   ❌ STRIKE ABORTED. Kale overlapped target {kale_veto_count}/10 times.")
            elif len(valid_L) < 2 or len(valid_R) < 2:
                print(f"   ⚠️ ISOLATION FAIL. Valid keypoints not found in crop.")
            else:
                final_L = np.median(np.stack(valid_L), axis=0).astype(int)
                final_R = np.median(np.stack(valid_R), axis=0).astype(int)
                
                off_L = final_L - lk_L
                off_R = final_R - lk_R
                
                print(f"   🔥 STRIKE APPROVED.")
                print(f"   📏 Shift From Survey -> L offset: {off_L}, R offset: {off_R}")

            print("\n👉 Press ANY KEY on the image window to proceed to the next target...")
            cv2.waitKey(0)
            cv2.destroyWindow(win_name)

    def start(self):
        try:
            print("\n--- 🔍 STABLE BURST SURVEY MODE ---")
            while True:
                burst_history_L, burst_history_R = [], []
                
                for _ in range(10):
                    self.cap_L.grab(); self.cap_R.grab()
                    retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
                    
                    if retL and retR:
                        boxes_L, masks_L = self.cv_L._get_detections(fL, weed_classes=[0, 2], kale_class=1, kale_thresh=0.05, weed_conf=0.20, kale_conf=1.0)
                        coords_L = self.cv_L.return_full(fL, precalc_boxes=boxes_L, precalc_masks=masks_L)
                        frame_data_L = [{'box': box.xyxy[0].cpu().numpy(), 'point': coords_L[i]} for i, box in enumerate(self.cv_L.filtered_boxes) if i < len(coords_L)]
                        burst_history_L.append(frame_data_L)

                        boxes_R, masks_R = self.cv_R._get_detections(fR, weed_classes=[0, 2], kale_class=1, kale_thresh=0.05, weed_conf=0.20, kale_conf=1.0)
                        coords_R = self.cv_R.return_full(fR, precalc_boxes=boxes_R, precalc_masks=masks_R)
                        frame_data_R = [{'box': box.xyxy[0].cpu().numpy(), 'point': coords_R[i]} for i, box in enumerate(self.cv_R.filtered_boxes) if i < len(coords_R)]
                        burst_history_R.append(frame_data_R)

                    time.sleep(0.1)                        
                
                if len(burst_history_L) < 3: continue

                # Save the freshest frames so the Overview Map can use them!
                self.last_survey_fL = fL.copy()
                self.last_survey_fR = fR.copy()
                
                self.clicks_L = self.cv_L.calculate_burst_math(burst_history_L)
                self.clicks_R = self.cv_R.calculate_burst_math(burst_history_R)

                sys.stdout.write(f"\r📊 SURVEY TARGETS [L:{len(self.clicks_L)} R:{len(self.clicks_R)}] | 'p' ACCEPT & TUNE STRIKES | 'q' QUIT")
                sys.stdout.flush()

                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    cmd = sys.stdin.readline().strip().lower()
                    if cmd == 'p': 
                        break
                    elif cmd == 'q': return

            if self.match_and_optimize():
                self.show_overview()
                self.test_precision_strikes()

        except KeyboardInterrupt:
            print("\n🛑 Manual Interruption.")
        finally:
            self.cap_L.release()
            self.cap_R.release()

if __name__ == "__main__":
    PrecisionTuner().start()
