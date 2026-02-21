import cv2
import numpy as np
import time
import sys
import json
import select
import platform
import os
import threading
from datetime import datetime
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
RECORDED_DATA_DIR = BASE_DIR / "Recorded_Data"

# --- SYSTEM CONSTANTS ---
IS_WINDOWS = platform.system() == "Windows"
CAM_BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

# --- CONTROL CONSTANTS ---
W, H = 640, 480
TARGET_Y_L, TARGET_Y_R = 240, 240
START_POS = (225, 220)

class Logger(object):
    def __init__(self, folder_path, mission_obj):
        self.terminal = sys.stdout
        self.log_file = open(folder_path / "Terminal Output.txt", "a")
        self.mission = mission_obj

    def write(self, message):
        self.terminal.write(message)
        if self.mission.start_time and "\r" not in message:
            if message.strip(): 
                elapsed = time.time() - self.mission.start_time
                stamped_message = f"{message.strip()} [{elapsed:.2f}s]\n"
                self.log_file.write(stamped_message)
            else:
                self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        pass

class HardwareFreeSurveyTester:
    def __init__(self):
        print(f"🔍 Loading Configuration for {platform.system()}...")
        self.load_hardware_config()
        self.load_camera_tunings()

        self.start_time = None
        self.trial_folder = None
        
        # VIDEO THREADING & BUFFERING
        self.frame_buffer = [] 
        self.recording_active = False
        self.stop_recording_thread = threading.Event()
        self.latest_fL = None
        self.latest_fR = None
        self.frame_lock = threading.Lock()
        
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
        self.frame_count = 0
        self.clicks_L, self.clicks_R, self.lk_targets = [], [], {}
        self.path_order, self.spatial_anchors = [], {} 
        self.current_step, self.mode = 0, "SURVEY_TEST"
        self.current_anchor_id = "START"
        self.apply_nuclear_hardware_lock()

    def recording_loop(self):
        """Background thread that captures frames at a steady 15 FPS."""
        fps = 15.0
        interval = 1.0 / fps
        print(f"🧵 Recording Thread Started at {fps} FPS")
        
        while not self.stop_recording_thread.is_set():
            loop_start = time.time()
            
            with self.frame_lock:
                fL, fR = self.latest_fL, self.latest_fR
            
            if fL is not None and fR is not None:
                combined = np.hstack((fL, fR))
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    cv2.putText(combined, f"T+{elapsed:.2f}s", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(combined, f"MODE: {self.mode}", (20, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(combined, current_time, (20, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                self.frame_buffer.append(combined)
            
            loop_time = time.time() - loop_start
            time.sleep(max(0, interval - loop_time))

    def setup_recording(self):
        if not RECORDED_DATA_DIR.exists():
            RECORDED_DATA_DIR.mkdir()
        
        folder_name = datetime.now().strftime("%m-%d-%y_%H-%M-%S_SURVEY_TEST")
        self.trial_folder = RECORDED_DATA_DIR / folder_name
        self.trial_folder.mkdir()

        sys.stdout = Logger(self.trial_folder, self)
        
        for _ in range(3):
            self.cap_L.grab(); self.cap_R.grab()
        retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
        if retL and retR:
            self.update_frame_share(fL, fR)

        self.start_time = time.time()
        self.recording_active = True
        
        self.stop_recording_thread.clear()
        self.rec_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.rec_thread.start()
        
        print(f"📁 Trial folder created: {folder_name}")

    def finalize_recording(self):
        self.stop_recording_thread.set()
        if hasattr(self, 'rec_thread'):
            self.rec_thread.join(timeout=2.0)
            
        print(f"\n💾 Compilation Phase: Processing {len(self.frame_buffer)} frames...")
        
        if len(self.frame_buffer) > 0:
            video_path = str(self.trial_folder / "video_of_trial.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 15.0, (W*2, H))
            
            if not writer.isOpened():
                print("❌ ERROR: Could not open VideoWriter.")
            else:
                for idx, frame in enumerate(self.frame_buffer):
                    writer.write(frame)
                    if idx % 50 == 0:
                        sys.__stdout__.write(f"\rEncoding: {idx}/{len(self.frame_buffer)}")
                        sys.__stdout__.flush()
                writer.release()
                print("\n🎬 Video compiled successfully.")

        total_time = time.time() - self.start_time if self.start_time else 0
        new_name = f"{self.trial_folder.name} [{total_time:.2f}s Trial]"
        new_path = self.trial_folder.parent / new_name
        
        sys.stdout = sys.__stdout__ 
        try:
            os.rename(self.trial_folder, new_path)
            print(f"✅ Mission Data Saved to: {new_name}")
        except Exception as e:
            print(f"⚠️ Rename failed: {e}")

    def update_frame_share(self, fL, fR):
        with self.frame_lock:
            self.latest_fL = fL.copy() if fL is not None else None
            self.latest_fR = fR.copy() if fR is not None else None

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

    def save_survey_images(self):
        for _ in range(5):
            self.cap_L.grab(); self.cap_R.grab()
        retL, imgL = self.cap_L.read(); retR, imgR = self.cap_R.read()
        if retL and retR:
            combined = np.hstack((imgL, imgR))
            save_path = str(self.trial_folder / "survey_combined.png") if self.trial_folder else str(BASE_DIR / "survey_combined.png")
            cv2.imwrite(save_path, combined)
            print(f"📸 [SAVE] Survey image saved.")

    def get_cluster(self, x, y):
        pts = [[float(x + i), float(y + j)] for i in [-2, 0, 2] for j in [-2, 0, 2]]
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def match_and_optimize(self):
        if not (self.clicks_L and self.clicks_R): return False
        
        # MOCKED LASER POSITION: We supply dummy coordinates since the hardware isn't connected
        curr_mpos = {'x': START_POS[0], 'y': START_POS[1]} 
        
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
        
        matched, initial_snapshot = {}, {}
        for m_id, (l_idx, r_idx) in enumerate(best_matches.items()):
            p_l, p_r = pts_l[l_idx], pts_r[r_idx]
            matched[m_id] = {'l': self.get_cluster(*p_l), 'r': self.get_cluster(*p_r), 'orig_l': p_l}
            ex, ey = -(p_l[0] - W + p_r[0]), -(((p_l[1] - TARGET_Y_L) + (p_r[1] - TARGET_Y_R)) / 2)
            initial_snapshot[m_id] = {'l': matched[m_id]['l'].copy(), 'r': matched[m_id]['r'].copy(), 'orig_l': p_l, 'mag': np.sqrt(ex**2+ey**2)}
        
        self.lk_targets = matched
        self.spatial_anchors["START"] = {'mpos': curr_mpos, 'snapshot': initial_snapshot}
        
        rem = list(matched.keys())
        last_xy = np.array([W//2, H//2])
        self.path_order = []
        
        while rem:
            node = min(rem, key=lambda i: np.linalg.norm(np.array(matched[i]['orig_l']) - last_xy))
            self.path_order.append(node)
            last_xy = np.array(matched[node]['orig_l'])
            rem.remove(node)
            
        print(f"\n✅ Stereo Matching Complete. Successfully Linked Targets: {len(self.path_order)}")
        return True

    def start(self):
        try:
            print("🚀 MOCK STARTUP: Bypassing Gantry Homing...")
            time.sleep(1.0) # Simulate a short delay
            
            self.save_survey_images()
            
            print("\n--- 🔍 STABLE BURST SURVEY MODE ---")
            while True:
                burst_history_L, burst_history_R = [], []
                
                for _ in range(10):
                    self.cap_L.grab(); self.cap_R.grab()
                    retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
                    
                    if retL and retR:
                        self.update_frame_share(fL, fR)

                        # --- LEFT CAMERA ---
                        boxes_L, masks_L = self.cv_L._get_detections(
                            fL, weed_classes=[0, 2], kale_class=1, 
                            kale_thresh=0.05, weed_conf=0.40, kale_conf=1.0
                        )
                        coords_L = self.cv_L.return_full(fL, precalc_boxes=boxes_L, precalc_masks=masks_L)
                        
                        frame_data_L = []
                        for i, box in enumerate(self.cv_L.filtered_boxes):
                            if i < len(coords_L):
                                frame_data_L.append({'box': box.xyxy[0].cpu().numpy(), 'point': coords_L[i]})
                        burst_history_L.append(frame_data_L)

                        # --- RIGHT CAMERA ---
                        boxes_R, masks_R = self.cv_R._get_detections(
                            fR, weed_classes=[0, 2], kale_class=1, 
                            kale_thresh=0.05, weed_conf=0.40, kale_conf=1.0
                        )
                        coords_R = self.cv_R.return_full(fR, precalc_boxes=boxes_R, precalc_masks=masks_R)
                        
                        frame_data_R = []
                        for i, box in enumerate(self.cv_R.filtered_boxes):
                            if i < len(coords_R):
                                frame_data_R.append({'box': box.xyxy[0].cpu().numpy(), 'point': coords_R[i]})
                        burst_history_R.append(frame_data_R)

                    time.sleep(0.1)                        
                
                if len(burst_history_L) < 3: continue
                
                # Make sure calculate_burst_math exists in your cv_helpers!
                self.clicks_L = self.cv_L.calculate_burst_math(burst_history_L)
                self.clicks_R = self.cv_R.calculate_burst_math(burst_history_R)

                sys.stdout.write(f"\r📊 TARGETS [L:{len(self.clicks_L)} R:{len(self.clicks_R)}] | 'p' FIRE | 's' SAVE | 'q' QUIT")
                sys.stdout.flush()

                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    cmd = sys.stdin.readline().strip().lower()
                    if cmd == 'p': 
                        self.setup_recording()
                        break
                    elif cmd == 's': self.save_survey_images()
                    elif cmd == 'q': return

            # Run the stereo matching logic exactly as it would on the real machine
            if self.match_and_optimize():
                print("🎯 Matching phase passed successfully. Hardware execution bypassed.")
            else:
                print("⚠️ Matching phase failed. Not enough targets or poor stereo correspondence.")

        except KeyboardInterrupt:
            print("\n🛑 Manual Interruption.")
        except Exception as e:
            print(f"\n💥 FATAL CRASH: {e}")
        finally:
            self.finalize_recording()

if __name__ == "__main__":
    HardwareFreeSurveyTester().start()