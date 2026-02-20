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
from motion_helpers import B1LaserController
from cv_helpers import WeedCV

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
YOLO_PT = str(WEIGHTS_DIR / "yolo_weed.pt")
SNIPER_PT = str(WEIGHTS_DIR / "sniper.pt")   
CAMERA_SETTINGS = Path("/home/laser/Documents/Laser_Workspace/SCI_Weeder/Minimal_Demo/camera_config.json")
HARDWARE_CONFIG = BASE_DIR / "hardware_config.json"
RECORDED_DATA_DIR = BASE_DIR / "Recorded_Data"

# --- SYSTEM CONSTANTS ---
IS_WINDOWS = platform.system() == "Windows"
CAM_BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

# --- CONTROL CONSTANTS ---
W, H = 640, 480
TARGET_Y_L, TARGET_Y_R = 240, 240
START_POS = (225, 220)

Kp = 60.0
DEADZONE = 5    
MAX_SPEED, TRAVEL_SPEED = 12000, 12000
DWELL_TIME = .5  
SETTLE_TIME = 0.5  

# --- RECOVERY & MOTION ---
RECOVERY_SPEED = 18000 
RECOVERY_ACCEL = 7000  
NORMAL_ACCEL = 2000

# Robust LK Parameters
LK_PARAMS = dict(
    winSize  = (31, 31),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

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

class B1ProductionMission:
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
        
        try:
            self.laser = B1LaserController(self.port)
        except Exception as e:
            print(f"❌ SERIAL ERROR: {e}"); sys.exit()

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
        self.current_step, self.mode = 0, "SURVEY"
        self.old_gray_L, self.old_gray_R = None, None
        self.tracking_frozen = False
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
            
            # Sleep to maintain constant video speed
            loop_time = time.time() - loop_start
            time.sleep(max(0, interval - loop_time))

    def setup_recording(self):
        if not RECORDED_DATA_DIR.exists():
            RECORDED_DATA_DIR.mkdir()
        
        folder_name = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
        self.trial_folder = RECORDED_DATA_DIR / folder_name
        self.trial_folder.mkdir()

        sys.stdout = Logger(self.trial_folder, self)
        
        # Flush the buffer to ensure the absolute latest frame is grabbed
        for _ in range(3):
            self.cap_L.grab(); self.cap_R.grab()
        retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
        if retL and retR:
            self.update_frame_share(fL, fR)

        self.start_time = time.time()
        self.recording_active = True
        
        # Start background thread
        self.stop_recording_thread.clear()
        self.rec_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.rec_thread.start()
        
        print(f"📁 Trial folder created: {folder_name}")
        print("📹 Steady-rate background recording active.")

    def finalize_recording(self):
        # Stop thread first
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
        """Helper to safely pass the latest frames to the recorder."""
        with self.frame_lock:
            self.latest_fL = fL.copy() if fL is not None else None
            self.latest_fR = fR.copy() if fR is not None else None

    def load_hardware_config(self):
        with open(HARDWARE_CONFIG, "r") as f:
            cfg = json.load(f)
        self.port = cfg["serial"]["grbl_port"]
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

    def wait_for_idle(self, timeout=30.0):
        start_wait = time.time()
        self.laser.serial.reset_input_buffer() 
        while time.time() - start_wait < timeout:
            # We capture frames during wait loops so the video doesn't freeze
            retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
            self.update_frame_share(fL, fR)
            
            status = self.laser.send_raw("?")
            if status and "Idle" in status:
                return True
            time.sleep(0.05) 
        return False

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

    def save_current_as_anchor(self, mpos):
        anchor_snapshot = {}
        for tid, data in self.lk_targets.items():
            xl, yl = np.median(data['l'].reshape(-1,2), axis=0)
            xr, yr = np.median(data['r'].reshape(-1,2), axis=0)
            ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
            anchor_snapshot[tid] = {'l': data['l'].copy(), 'r': data['r'].copy(), 'orig_l': data['orig_l'], 'mag': np.sqrt(ex**2+ey**2)}
        anchor_id = self.path_order[self.current_step]
        self.spatial_anchors[anchor_id] = {'mpos': (mpos['x'], mpos['y']), 'snapshot': anchor_snapshot}

    def match_and_optimize(self):
        if not (self.clicks_L and self.clicks_R): return False
        mpos = self.laser.update_status()
        curr_mpos = (mpos['x'], mpos['y']) if mpos else START_POS
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
        rem = list(matched.keys()); last_xy = np.array([W//2, H//2]); self.path_order = []
        while rem:
            node = min(rem, key=lambda i: np.linalg.norm(np.array(matched[i]['orig_l']) - last_xy))
            self.path_order.append(node); last_xy = np.array(matched[node]['orig_l']); rem.remove(node)
        self.mode = "EXECUTE"; self.current_step = 0
        print(f"🚀 Mission Loaded. Targets: {len(self.path_order)}")
        return True
    
    def execute_precision_strike(self, return_pos, lk_L, lk_R):
        old_mode = self.mode
        self.mode = "PRECISION_STRIKE"
        print(f"⚔️ ENTERING PRECISION STRIKE")
        self.laser.stop()
        time.sleep(0.1) 
        
        SAMPLES = 10
        valid_L, valid_R = [], []
        crop_s = 120  
        
        def get_roi(img, cx, cy, size):
            h, w = img.shape[:2]
            x1, y1 = int(np.clip(cx-size, 0, w)), int(np.clip(cy-size, 0, h))
            x2, y2 = int(np.clip(cx+size, 0, w)), int(np.clip(cy+size, 0, h))
            return img[y1:y2, x1:x2], x1, y1

        for _ in range(SAMPLES):
            retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
            if not (retL and retR): continue
            self.update_frame_share(fL, fR)

            roi_L, offX_L, offY_L = get_roi(fL, lk_L[0], lk_L[1], crop_s)
            roi_R, offX_R, offY_R = get_roi(fR, lk_R[0], lk_R[1], crop_s)

            pts_L = self.cv_L.return_full(roi_L)
            boxes_L = self.cv_L.filtered_boxes
            pts_R = self.cv_R.return_full(roi_R)
            boxes_R = self.cv_R.filtered_boxes

            rel_lk_L = (lk_L[0] - offX_L, lk_L[1] - offY_L)
            for j, box in enumerate(boxes_L):
                b = box.xyxy[0].cpu().numpy()
                if (b[0] <= rel_lk_L[0] <= b[2]) and (b[1] <= rel_lk_L[1] <= b[3]):
                    valid_L.append(np.array([offX_L + pts_L[j][0], offY_L + pts_L[j][1]]))
                    break

            rel_lk_R = (lk_R[0] - offX_R, lk_R[1] - offY_R)
            for j, box in enumerate(boxes_R):
                b = box.xyxy[0].cpu().numpy()
                if (b[0] <= rel_lk_R[0] <= b[2]) and (b[1] <= rel_lk_R[1] <= b[3]):
                    valid_R.append(np.array([offX_R + pts_R[j][0], offY_R + pts_R[j][1]]))
                    break
            time.sleep(0.01)

        if len(valid_L) < 2 or len(valid_R) < 2:
            print(f"⚠️ ISOLATION FAIL. Fallback to LK.")
            final_L, final_R = lk_L, lk_R
        else:
            print(f"✅ ISOLATION SUCCESS.")
            final_L = np.median(np.stack(valid_L), axis=0)
            final_R = np.median(np.stack(valid_R), axis=0)

        pt_L, pt_R = np.array([[final_L[0], final_L[1]]], dtype=np.float32), np.array([[final_R[0], final_R[1]]], dtype=np.float32)
        start_fine = time.time()
        _, fL = self.cap_L.read(); _, fR = self.cap_R.read()
        grayL_old, grayR_old = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)

        while (time.time() - start_fine < 2.0):
            retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
            if not (retL and retR): break
            self.update_frame_share(fL, fR)
            
            grayL, grayR = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
            pt_L, _, _ = cv2.calcOpticalFlowPyrLK(grayL_old, grayL, pt_L, None, **LK_PARAMS)
            pt_R, _, _ = cv2.calcOpticalFlowPyrLK(grayR_old, grayR, pt_R, None, **LK_PARAMS)
            xl, yl = pt_L[0]; xr, yr = pt_R[0]
            ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
            mag = np.sqrt(ex**2 + ey**2)
            if mag < 2.0: break
            self.laser.jog(ex/mag, -ey/mag, feed=500)
            grayL_old, grayR_old = grayL.copy(), grayR.copy()

        self.laser.stop()
        self.mode = "LASER_SPIRAL"
        curr_pos = self.laser.update_status()
        
        # --- FIX: FORCE LASER STATE ---
        self.laser.send_raw("M4") # 1. Arm Dynamic Laser Mode
        
        # Spiral burn logic
        self.laser.spiral_burn(curr_pos['x'], curr_pos['y'], radius=4.0, steps=20, speed=1000)
        
        self.laser.send_raw("M5") # 2. Disarm Laser for safe travel
        # ------------------------------
        
        # Capture frames during return travel (G90 ensures Absolute Mode)
        self.laser.set_acceleration(NORMAL_ACCEL/2)
        self.laser.send_raw(f"G90\nG1 X{return_pos['x']} Y{return_pos['y']} F{RECOVERY_SPEED}")
        self.wait_for_idle(timeout=2.0)
        self.laser.set_acceleration(NORMAL_ACCEL)
        self.mode = old_mode

    def start(self):
        try:
            self.laser.send_raw("$X")  
            time.sleep(0.5)
            print("🤖 Homing Gantry..."); self.laser.home()
            if not self.wait_for_idle(timeout=100): return
            print(f"📍 Moving to Survey Position {START_POS}...")
            self.laser.send_raw(f"G90\nG1 X{START_POS[0]} Y{START_POS[1]} F{TRAVEL_SPEED}")
            if not self.wait_for_idle(): return

            settle_end = time.time() + 3.0
            while time.time() < settle_end:
                self.cap_L.grab(); self.cap_R.grab(); time.sleep(0.01)
            self.save_survey_images()
            print("\n--- 🔍 STABLE BURST SURVEY MODE ---")
            while True:
                burst_L, burst_R = [], []
                for _ in range(10):
                    self.cap_L.grab(); self.cap_R.grab()
                    retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
                    if retL and retR:
                        burst_L.append(fL); burst_R.append(fR)
                        # Ensure the recording thread has access to the most recent frame!
                        self.update_frame_share(fL, fR)

                    time.sleep(0.1)                        
                
                if len(burst_L) < 3: continue
                self.clicks_L = self.cv_L.return_burst_stable(burst_L)
                self.clicks_R = self.cv_R.return_burst_stable(burst_R)

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

            if not self.match_and_optimize(): 
                self.finalize_recording(); return

            self.laser.set_acceleration(NORMAL_ACCEL)

            while True:
                self.frame_count += 1
                if self.frame_count % 500 == 0: self.apply_nuclear_hardware_lock()

                retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
                if not (retL and retR): break
                self.update_frame_share(fL, fR)

                grayL, grayR = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)

                if self.mode == "EXECUTE":
                    if self.current_step >= len(self.path_order):
                        print("🏁 Mission Complete."); break
                    
                    if self.old_gray_L is not None and not self.tracking_frozen:
                        new_lk = {}
                        for tid, data in self.lk_targets.items():
                            pL_n, stL, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_L, grayL, data['l'], None, **LK_PARAMS)
                            pR_n, stR, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_R, grayR, data['r'], None, **LK_PARAMS)
                            if np.sum(stL) >= 5 and np.sum(stR) >= 5:
                                new_lk[tid] = {'l': pL_n, 'r': pR_n, 'orig_l': data['orig_l']}
                        self.lk_targets = new_lk

                    tid = self.path_order[self.current_step]
                    
                    if tid not in self.lk_targets:
                        print(f"⚠️ Target {tid} lost! Returning to Survey Anchor...")
                        # 1. Move gantry back to the original survey position
                        anchor_pos = self.spatial_anchors["START"]["mpos"]
                        self.laser.send_raw(f"G90\nG1 X{anchor_pos[0]} Y{anchor_pos[1]} F{RECOVERY_SPEED}")
                        self.wait_for_idle()
                        
                        # 2. THE FIX: Flush the camera buffer to get rid of frames from the move
                        for _ in range(10): self.cap_L.grab(); self.cap_R.grab()
                        
                        # 3. Capture a fresh baseline so the tracker has a valid "start" point
                        retL, fL_n = self.cap_L.read(); retR, fR_n = self.cap_R.read()
                        if retL and retR:
                            self.old_gray_L = cv2.cvtColor(fL_n, cv2.COLOR_BGR2GRAY)
                            self.old_gray_R = cv2.cvtColor(fR_n, cv2.COLOR_BGR2GRAY)
                        
                        # 4. Re-apply the initial survey coordinates to the tracker
                        snap = self.spatial_anchors["START"]["snapshot"][tid]
                        self.lk_targets[tid] = {'l': snap['l'], 'r': snap['r'], 'orig_l': snap['orig_l']}
                        
                        print(f"🔄 Baseline refreshed at Anchor. Restarting tracking for {tid}.")
                        continue # Restart the loop with the gantry physically and digitally reset

                    t = self.lk_targets[tid]
                    xl, yl = np.median(t['l'].reshape(-1,2), axis=0)
                    xr, yr = np.median(t['r'].reshape(-1,2), axis=0)
                    ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
                    mag = np.sqrt(ex**2 + ey**2)
                    step_size = np.clip(mag * 0.05, 1.0, 3.0)

                    if mag > DEADZONE: 
                        self.laser.jog(ex/mag*step_size, -ey/mag*step_size, np.clip(mag*Kp, 0, MAX_SPEED))
                    else:
                        self.laser.stop()
                        clean_mpos = self.laser.update_status()
                        self.save_current_as_anchor(clean_mpos)
                        self.execute_precision_strike(return_pos=clean_mpos, lk_L=(xl, yl), lk_R=(xr, yr))
                        time.sleep(1.0)
                        self.current_step += 1

                self.old_gray_L, self.old_gray_R = grayL.copy(), grayR.copy()
            self.laser.send_raw(f"G90\nG1 X{10} Y{10} F{TRAVEL_SPEED}")
            

            self.laser.close()
        
        except KeyboardInterrupt:
            print("\n🛑 Manual Interruption.")
        except Exception as e:
            print(f"\n💥 FATAL CRASH: {e}")
        finally:
            self.finalize_recording()

if __name__ == "__main__":
    B1ProductionMission().start()