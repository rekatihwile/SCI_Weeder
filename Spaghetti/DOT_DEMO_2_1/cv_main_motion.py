import cv2
import numpy as np
import time
import sys
import json

from helpers import B1LaserController
from cv_helpers import WeedCV
import subprocess
import time
# --- CONFIGURATION ---
PORT = '/dev/ttyUSB0'
W, H = 640, 480
TARGET_Y_L, TARGET_Y_R = 240, 240
START_POS = (225, 220)
CONFIG_FILE = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/DOT_DEMO_2_1/camera_config.json"

Kp = 60.0
DEADZONE = 5    
MAX_SPEED, TRAVEL_SPEED = 12000, 12000
DWELL_TIME = .5  
SETTLE_TIME = 0.5  

# --- RECOVERY & MOTION ---
RECOVERY_SPEED = 18000 
RECOVERY_ACCEL = 7000  
NORMAL_ACCEL = 2000

YOLO_PT = "/home/laser/Downloads/final_train_feb_1stbest (2).pt"
SNIPER_PT = "/home/laser/Downloads/sniper_jetson_ready.pt"

def hard_reset_camera_hardware(cam_id):
        """Forces the Linux kernel to lock camera registers to manual mode."""
        dev = f"/dev/video{cam_id}"
        print(f"🔒 Locking Hardware Registers for {dev}...")
        # 1 = Manual Exposure, 0 = Manual White Balance
        commands = [
            ['v4l2-ctl', '-d', dev, '-c', 'exposure_auto=1'],
            ['v4l2-ctl', '-d', dev, '-c', 'white_balance_temperature_auto=0'],
            ['v4l2-ctl', '-d', dev, '-c', 'backlight_compensation=0'],
            ['v4l2-ctl', '-d', dev, '-c', 'power_line_frequency=0'] # Prevents 60Hz flicker
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"⚠️ v4l2 command failed for {dev}: {e}")

# Robust LK Parameters
LK_PARAMS = dict(
    winSize  = (31, 31),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

class B1ProductionMission:
    def __init__(self):
        print("🔍 Initializing Hardware & Vision...")
        try:
            self.laser = B1LaserController(PORT)
        except Exception as e:
            print(f"❌ SERIAL ERROR: {e}"); sys.exit()

        self.cv_L = WeedCV(YOLO_PT, SNIPER_PT)
        self.cv_R = WeedCV(YOLO_PT, SNIPER_PT)

        # 1. Open the streams (Firmware starts its "Auto-Hunt")
        self.cap_R, self.cap_L = cv2.VideoCapture(0), cv2.VideoCapture(2)
        
        # 2. Wait for the camera to "Wake Up" (Crucial for Jetson)
        print("⏳ Waiting 2.0s for ISP to stabilize...")
        time.sleep(2.0)

        # 3. Apply the OS-Level Hard Lock
        self.apply_nuclear_hardware_lock()

        # ... (rest of mission variables)

        # ... (rest of mission variables)

        # 4. Initialize Mission Variables
        self.clicks_L, self.clicks_R, self.lk_targets = [], [], {}
        self.path_order, self.spatial_anchors = [], {} 
        self.current_step, self.mode = 0, "SURVEY"
        self.old_gray_L, self.old_gray_R = None, None
        self.dwell_start = None
        self.laser_fired = False 
        self.tracking_frozen = False
        self.last_status_check = 0
        self.recovery_settle_start = None
        self.current_anchor_id = "START"
        self.prev_time = 0

    def apply_camera_settings(self, caps):
        try:
            with open(CONFIG_FILE, 'r') as f:
                s = json.load(f)
            
            # PHASE 1: Hard Register Lock (Kernel Level)
            hard_reset_camera_hardware(0)
            hard_reset_camera_hardware(2)
            
            # Give the hardware 200ms to switch pipelines
            time.sleep(0.2)

            for cap in caps:
                # PHASE 2: Firmware Hammer (OpenCV Level)
                # We loop this 5 times because some UVC cameras ignore 
                # manual writes during the first 5-10 frames of capture.
                for _ in range(5):
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                    cap.set(cv2.CAP_PROP_EXPOSURE, s['exp'])
                    cap.set(cv2.CAP_PROP_GAIN, s['gain'])
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, s['brt'])
                    cap.set(cv2.CAP_PROP_CONTRAST, s['con'])
                    cap.set(cv2.CAP_PROP_SATURATION, s['sat'])
                    cap.set(cv2.CAP_PROP_HUE, s['hue'])
                    cap.set(cv2.CAP_PROP_GAMMA, s['gamma'])
                    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, s['wb'])
                    cap.set(cv2.CAP_PROP_SHARPNESS, s['sharp'])
            
            print("✅ Hardware Lock Complete: Cameras are now fully manual.")
        except Exception as e:
            print(f"❌ Hardware Lock Failed: {e}")

    def apply_nuclear_hardware_lock(self):
        """Forces the Linux Kernel to ignore camera firmware auto-requests."""
        try:
            with open(CONFIG_FILE, 'r') as f:
                s = json.load(f)
            
            for cid in [0, 2]:
                dev = f"/dev/video{cid}"
                print(f"🔒 Nuking Auto-Logic on {dev}...")

                # Force Manual Mode (1=Manual, 3=Auto)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', 'exposure_auto=1'], check=True)
                # Disable Auto Priority (Prevents frame rate drops/brightness shifts)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', 'exposure_auto_priority=0'], check=True)
                # Disable White Balance Auto
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', 'white_balance_temperature_auto=0'], check=True)
                
                # SET THE SPECIFIC VALUES (Using v4l2 instead of cap.set)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', f'exposure_absolute={s["exp"]}'], check=True)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', f'gain={s["gain"]}'], check=True)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', f'brightness={s["brt"]}'], check=True)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', f'contrast={s["con"]}'], check=True)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', f'sharpness={s["sharp"]}'], check=True)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', f'white_balance_temperature={s["wb"]}'], check=True)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', f'backlight_compensation=0'], check=True)
                
            print("✅ Nuclear Lock Complete. The camera is now a dumb sensor.")
        except Exception as e:
            print(f"❌ Nuclear Lock Failed: {e}")

    def get_cluster(self, x, y):
        """Generates a 3x3 grid (9 points) centered on (x,y) for robust tracking."""
        pts = []
        for i in [-2, 0, 2]: # 2-pixel spacing for better texture capture
            for j in [-2, 0, 2]:
                pts.append([float(x + i), float(y + j)])
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def save_current_as_anchor(self, mpos):
        """Saves 9-point clusters for every target to the spatial anchor."""
        anchor_snapshot = {}
        for tid, data in self.lk_targets.items():
            # Calculate error based on the MEDIAN of the 9 points
            xl, yl = np.median(data['l'].reshape(-1,2), axis=0)
            xr, yr = np.median(data['r'].reshape(-1,2), axis=0)
            ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
            mag = np.sqrt(ex**2 + ey**2)
            
            anchor_snapshot[tid] = {
                'l': data['l'].copy(), # Stores the full (9, 1, 2) array
                'r': data['r'].copy(),
                'orig_l': data['orig_l'], 
                'mag': mag 
            }
        anchor_id = self.path_order[self.current_step]
        self.spatial_anchors[anchor_id] = {'mpos': (mpos['x'], mpos['y']), 'snapshot': anchor_snapshot}
        self.current_anchor_id = anchor_id
        return anchor_snapshot

    def match_and_optimize(self):
        print("🔍 Finalizing Mission Map...")
        if not (self.clicks_L and self.clicks_R): return
        mpos = self.laser.update_status()
        curr_mpos = (mpos['x'], mpos['y']) if mpos else START_POS
        pts_l, pts_r = np.array(self.clicks_L), np.array(self.clicks_R)
        
        best_matches, max_matches = {}, 0
        for i, a_l in enumerate(pts_l):
            for j, a_r in enumerate(pts_r):
                dx, dy = a_r[0] - a_l[0], a_r[1] - a_l[1]
                if not (30 < abs(dx) < 350): continue 
                cfg = {}
                shifted_l = pts_l + [dx, dy]
                for l_idx, p_s in enumerate(shifted_l):
                    dists = np.linalg.norm(pts_r - p_s, axis=1)
                    r_idx = np.argmin(dists)
                    if dists[r_idx] < 20: cfg[l_idx] = r_idx
                if len(cfg) > max_matches: max_matches = len(cfg); best_matches = cfg

        matched, initial_snapshot = {}, {}
        for m_id, (l_idx, r_idx) in enumerate(best_matches.items()):
            p_l, p_r = pts_l[l_idx], pts_r[r_idx]
            # Convert single AI points into 9-point clusters
            matched[m_id] = {'l': self.get_cluster(*p_l), 'r': self.get_cluster(*p_r), 'orig_l': p_l}
            
            xl, yl = np.median(matched[m_id]['l'].reshape(-1,2), axis=0)
            xr, yr = np.median(matched[m_id]['r'].reshape(-1,2), axis=0)
            ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
            initial_snapshot[m_id] = {'l': matched[m_id]['l'].copy(), 'r': matched[m_id]['r'].copy(), 'orig_l': p_l, 'mag': np.sqrt(ex**2+ey**2)}

        self.lk_targets = matched
        self.spatial_anchors["START"] = {'mpos': curr_mpos, 'snapshot': initial_snapshot}
        
        rem = list(matched.keys()); curr_xy = np.array([W//2, H//2]); self.path_order = []
        while rem:
            node = min(rem, key=lambda i: np.linalg.norm(np.array(matched[i]['orig_l']) - curr_xy))
            self.path_order.append(node); curr_xy = np.array(matched[node]['orig_l']); rem.remove(node)
        
        try:
            cv2.destroyWindow(self.cv_L.win_main)
            cv2.destroyWindow(self.cv_R.win_main)
        except: pass
        
        self.mode = "EXECUTE"; self.current_step = 0
        print(f"🚀 AI Mission Active. Heavy Vision Shutdown. Targets: {len(self.path_order)}")

    def start(self):
        self.laser.set_acceleration(3000); self.laser.home(); time.sleep(1)
        self.laser.send_raw(f"G90\nG1 X{START_POS[0]} Y{START_POS[1]} F{TRAVEL_SPEED}")
        self.laser.set_acceleration(NORMAL_ACCEL)
        
        while True:
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time != 0 else 0
            self.prev_time = curr_time

            retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
            if not (retL and retR): break
            grayL, grayR = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
            
            if self.mode == "SURVEY":
                self.clicks_L = self.cv_L.return_full(fL)
                self.clicks_R = self.cv_R.return_full(fR)
                self.cv_L.show_debug(fL, self.clicks_L)
                self.cv_R.show_debug(fR, self.clicks_R)

            elif self.mode in ["EXECUTE", "RECOVER", "IDLE"]:
                if self.old_gray_L is not None and not self.tracking_frozen and self.mode != "RECOVER":
                    new_lk = {}
                    for tid, data in self.lk_targets.items():
                        # Track all 9 points per target
                        pL_n, stL, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_L, grayL, data['l'], None, **LK_PARAMS)
                        pR_n, stR, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_R, grayR, data['r'], None, **LK_PARAMS)

                        # Check if at least 5 of the 9 points are still visible
                        if np.sum(stL) >= 5 and np.sum(stR) >= 5:
                            # Verify median back-error
                            pL_b, _, _ = cv2.calcOpticalFlowPyrLK(grayL, self.old_gray_L, pL_n, None, **LK_PARAMS)
                            
                            # Calculate distance between original cluster and back-tracked cluster
                            err = np.linalg.norm(data['l'] - pL_b, axis=2)
                            if np.median(err) < 2.5: # Slightly more lenient for 3x3 grid
                                new_lk[tid] = {'l': pL_n, 'r': pR_n, 'orig_l': data['orig_l']}
                                
                                # Visualize based on Median
                                xl, yl = np.median(pL_n.reshape(-1,2), axis=0)
                                xr, yr = np.median(pR_n.reshape(-1,2), axis=0)
                                is_curr = (self.current_step < len(self.path_order) and tid == self.path_order[self.current_step])
                                clr = (0, 255, 0) if is_curr else (0, 165, 255)
                                cv2.circle(fL, (int(xl), int(yl)), 7, clr, 1)
                                cv2.circle(fR, (int(xr), int(yr)), 7, clr, 1)
                    self.lk_targets = new_lk

                if self.mode == "EXECUTE":
                    if self.current_step >= len(self.path_order):
                        print("🏁 Mission Complete."); self.mode = "IDLE"; continue
                    
                    tid = self.path_order[self.current_step]
                    if tid not in self.lk_targets:
                        # --- Recovery logic (unchanged) ---
                        best_k, best_m = None, float('inf')
                        for k, anc in self.spatial_anchors.items():
                            if tid in anc['snapshot'] and anc['snapshot'][tid]['mag'] < best_m:
                                best_m, best_k = anc['snapshot'][tid]['mag'], k
                        if best_k:
                            self.current_anchor_id = best_k
                            self.laser.set_acceleration(RECOVERY_ACCEL)
                            self.laser.send_raw(f"G90\nG1 X{self.spatial_anchors[best_k]['mpos'][0]} Y{self.spatial_anchors[best_k]['mpos'][1]} F{RECOVERY_SPEED}")
                            self.mode = "RECOVER"; self.recovery_settle_start = None; continue
                        else: self.current_step += 1; continue

                    # 1. Calculate Error from 3x3 Cluster Median
                    t = self.lk_targets[tid]
                    xl, yl = np.median(t['l'].reshape(-1,2), axis=0)
                    xr, yr = np.median(t['r'].reshape(-1,2), axis=0)
                    ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
                    mag = np.sqrt(ex**2 + ey**2)
                    
                    if mag > DEADZONE: 
                        # Continue jogging while tracking is ACTIVE
                        self.laser.jog(ex/mag, -ey/mag, np.clip(mag*Kp, 0, MAX_SPEED))
                    else:
                        # --- TARGET REACHED: FREEZE TRACKING & BURN ---
                        if self.dwell_start is None:
                            # 1. STOP EVERYTHING
                            self.laser.stop()
                            self.tracking_frozen = True  # <--- LK ignores all frames from now on
                            self.dwell_start = time.time()
                            
                            # 2. Get arrival point for return
                            mpos_arrival = self.laser.update_status()
                            sx, sy = (mpos_arrival['x'], mpos_arrival['y']) if mpos_arrival else START_POS
                            
                            print(f"🔥 [LOCK ON] Burn started at {sx, sy}. LK Paused.")
                            
                            # 3. Trigger Atomic Spiral (Stationary 0.5s @ F10 + 1.0s Spiral)
                            # NOTE: This function must finish with a return to sx, sy
                            self.laser.spiral_burn(sx, sy, radius=5.0, steps=20, speed=1500)
                            
                            # 4. Save arrival pose to the spatial anchor
                            self.save_current_as_anchor(mpos_arrival if mpos_arrival else {'x':sx, 'y':sy})

                            # 5. Flush Smoke: Grab frames while stationary at center
                            # We grab 12+ frames to ensure the 'live' buffer is post-smoke
                            for _ in range(15): self.cap_L.grab(); self.cap_R.grab()
                            retL_n, fL_n = self.cap_L.read(); retR_n, fR_n = self.cap_R.read()
                            
                            # 6. RE-SEED: Load the saved 3x3 clusters from the anchor
                            # This aligns the tracker to the weed's current visual position
                            anc = self.spatial_anchors[self.current_anchor_id]
                            for sid, d in anc['snapshot'].items():
                                self.lk_targets[sid] = {
                                    'l': d['l'].copy().reshape(-1,1,2), 
                                    'r': d['r'].copy().reshape(-1,1,2), 
                                    'orig_l': d['orig_l']
                                }
                            
                            # 7. Update reference frames for the next movement
                            self.old_gray_L = cv2.cvtColor(fL_n, cv2.COLOR_BGR2GRAY)
                            self.old_gray_R = cv2.cvtColor(fR_n, cv2.COLOR_BGR2GRAY)
                            
                            # 8. RESUME TRACKING
                            self.tracking_frozen = False
                            self.current_step += 1
                            self.dwell_start = None
                            print(f"✅ Target {tid} Burned. LK Resumed.")

                elif self.mode == "RECOVER":
                    if time.time() - self.last_status_check > 0.1:
                        self.last_status_check = time.time()
                        status = self.laser.send_raw("?") or ""
                        if "Idle" in status:
                            if self.recovery_settle_start is None: self.recovery_settle_start = time.time()
                            for _ in range(5): self.cap_L.grab(); self.cap_R.grab()
                            if time.time() - self.recovery_settle_start > 0.3:
                                _, fL_n = self.cap_L.read(); _, fR_n = self.cap_R.read()
                                anc = self.spatial_anchors[self.current_anchor_id]
                                self.lk_targets = {}
                                for sid, d in anc['snapshot'].items():
                                    # Recovery with full 9-point cluster
                                    self.lk_targets[sid] = {
                                        'l': d['l'].copy().reshape(-1, 1, 2), 
                                        'r': d['r'].copy().reshape(-1, 1, 2), 
                                        'orig_l': d['orig_l']
                                    }
                                self.old_gray_L, self.old_gray_R = cv2.cvtColor(fL_n, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR_n, cv2.COLOR_BGR2GRAY)
                                self.laser.set_acceleration(NORMAL_ACCEL); self.mode = "EXECUTE"

            # Telemetry
            cv2.drawMarker(fL, (320, TARGET_Y_L), (0,0,255), 2, 25, 2)
            cv2.drawMarker(fR, (320, TARGET_Y_R), (0,0,255), 2, 25, 2)
            cv2.putText(fL, f"FPS: {fps:.1f}", (W-120, 30), 0, 0.6, (0, 255, 255), 2)
            cv2.imshow("Left Main", fL); cv2.imshow("Right Main", fR)
            if self.mode != "RECOVER": self.old_gray_L, self.old_gray_R = grayL.copy(), grayR.copy()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'): self.match_and_optimize()
            elif key == ord('c'): self.clicks_L, self.clicks_R, self.lk_targets, self.mode = [], [], {}, "SURVEY"
            elif key == 27: break
        self.laser.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    B1ProductionMission().start()