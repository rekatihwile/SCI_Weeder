import cv2
import numpy as np
import time
import sys
import json
import subprocess
import select
from helpers import B1LaserController
from cv_helpers import WeedCV

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

# Robust LK Parameters
LK_PARAMS = dict(
    winSize  = (31, 31),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

class B1ProductionMission:
    def __init__(self):
        print("🔍 Initializing Hardware & Vision (Headless Mode)...")
        try:
            self.laser = B1LaserController(PORT)
        except Exception as e:
            print(f"❌ SERIAL ERROR: {e}"); sys.exit()

        self.cv_L = WeedCV(YOLO_PT, SNIPER_PT)
        self.cv_R = WeedCV(YOLO_PT, SNIPER_PT)
#SET CAMERA INDICES HERE
        self.cap_R, self.cap_L = cv2.VideoCapture(0), cv2.VideoCapture(2)
        # self.cap_R, self.cap_L = cv2.VideoCapture(4), cv2.VideoCapture(1)
        for cap in [self.cap_L, self.cap_R]:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        time.sleep(2.0)
        self.apply_nuclear_hardware_lock()

        self.clicks_L, self.clicks_R, self.lk_targets = [], [], {}
        self.path_order, self.spatial_anchors = [], {} 
        self.current_step, self.mode = 0, "SURVEY"
        self.old_gray_L, self.old_gray_R = None, None
        self.tracking_frozen = False
        self.current_anchor_id = "START"

    def apply_nuclear_hardware_lock(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                s = json.load(f)
            for cid in [0,2]:   
            # for cid in [1, 4]:
                dev = f"/dev/video{cid}"
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', 'exposure_auto=1'], check=True)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', 'exposure_auto_priority=0'], check=True)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', f'exposure_absolute={s["exp"]}'], check=True)
                subprocess.run(['v4l2-ctl', '-d', dev, '-c', f'gain={s["gain"]}'], check=True)
            print("✅ Nuclear Lock Applied.")
        except Exception as e:
            print(f"❌ Lock Failed: {e}")
            
    def wait_for_idle(self, timeout=30.0):
        start_wait = time.time()
        # Clear the buffer before starting to ensure we aren't reading old data
        self.laser.serial.reset_input_buffer() 
        
        while time.time() - start_wait < timeout:
            status = self.laser.send_raw("?")
            if status:
                # GRBL status is usually inside brackets, e.g., <Idle|WPos:0,0,0...>
                if "Idle" in status:
                    print(f"✅ Machine Idle after {round(time.time()-start_wait, 2)}s")
                    return True
            
            # Short sleep to avoid flooding the serial line
            time.sleep(0.1) 
        return False

    def get_cluster(self, x, y):
        pts = [[float(x + i), float(y + j)] for i in [-2, 0, 2] for j in [-2, 0, 2]]
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def save_current_as_anchor(self, mpos):
        anchor_snapshot = {}
        for tid, data in self.lk_targets.items():
            xl, yl = np.median(data['l'].reshape(-1,2), axis=0)
            xr, yr = np.median(data['r'].reshape(-1,2), axis=0)
            ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
            mag = np.sqrt(ex**2 + ey**2)
            anchor_snapshot[tid] = {'l': data['l'].copy(), 'r': data['r'].copy(), 'orig_l': data['orig_l'], 'mag': mag}
        anchor_id = self.path_order[self.current_step]
        self.spatial_anchors[anchor_id] = {'mpos': (mpos['x'], mpos['y']), 'snapshot': anchor_snapshot}
        self.current_anchor_id = anchor_id

    def match_and_optimize(self):
        if not (self.clicks_L and self.clicks_R): 
            print("\n⚠️ No weeds found. Survey failed."); return False
        
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
            matched[m_id] = {'l': self.get_cluster(*p_l), 'r': self.get_cluster(*p_r), 'orig_l': p_l}
            ex, ey = -(p_l[0] - W + p_r[0]), -(((p_l[1] - TARGET_Y_L) + (p_r[1] - TARGET_Y_R)) / 2)
            initial_snapshot[m_id] = {'l': matched[m_id]['l'].copy(), 'r': matched[m_id]['r'].copy(), 'orig_l': p_l, 'mag': np.sqrt(ex**2+ey**2)}

        self.lk_targets = matched
        self.spatial_anchors["START"] = {'mpos': curr_mpos, 'snapshot': initial_snapshot}
        rem = list(matched.keys()); curr_xy = np.array([W//2, H//2]); self.path_order = []
        while rem:
            node = min(rem, key=lambda i: np.linalg.norm(np.array(matched[i]['orig_l']) - curr_xy))
            self.path_order.append(node); curr_xy = np.array(matched[node]['orig_l']); rem.remove(node)
        
        self.mode = "EXECUTE"; self.current_step = 0
        print(f"\n🚀 Mission Loaded. Targets to neutralize: {len(self.path_order)}")
        return True

    def start(self):
        # --- 1. HOMING & POSITIONING ---
        print("🤖 Homing Gantry...")
        self.laser.home()
        if not self.wait_for_idle(timeout=100):
            print("❌ Homing timed out."); return

        print(f"📍 Moving to Survey Position {START_POS}...")
        self.laser.send_raw(f"G90\nG1 X{START_POS[0]} Y{START_POS[1]} F{TRAVEL_SPEED}")
        if not self.wait_for_idle():
            print("❌ Initial move failed."); return

        # --- 2. SETTLE & SURVEY ---
        print("📷 Gantry stationary. Settling vision buffer (3s)...")
        settle_end = time.time() + 3.0
        while time.time() < settle_end:
            self.cap_L.grab(); self.cap_R.grab()
            time.sleep(0.01)

        print("\n--- 🔍 SURVEY MODE ---")
        while True:
            retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
            if not (retL and retR): continue
            
            self.clicks_L = self.cv_L.return_full(fL)
            self.clicks_R = self.cv_R.return_full(fR)
            print(f"\r📊 Found: L[{len(self.clicks_L)}] R[{len(self.clicks_R)}] | 'p'+Enter to fire, 'q' to quit", end="")
            
            # Use select to check for input without blocking the camera read
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'p': break
                elif cmd == 'q': print("\n👋 Quitting."); return

        if not self.match_and_optimize(): return
        self.laser.set_acceleration(NORMAL_ACCEL)

        # --- 3. EXECUTION MISSION ---
        while True:
            retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
            if not (retL and retR): break
            grayL, grayR = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)

            if self.mode == "EXECUTE":
                if self.current_step >= len(self.path_order):
                    print("🏁 Mission Complete."); break
                
                # Lucas-Kanade Tracking
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
                    self.current_step += 1; continue

                t = self.lk_targets[tid]
                xl, yl = np.median(t['l'].reshape(-1,2), axis=0)
                xr, yr = np.median(t['r'].reshape(-1,2), axis=0)
                ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
                mag = np.sqrt(ex**2 + ey**2)
                
                if mag > DEADZONE: 
                    self.laser.jog(ex/mag, -ey/mag, np.clip(mag*Kp, 0, MAX_SPEED))
                else:
                    # Target Lock: Spiral Burn Sequence
                    self.laser.stop()
                    self.tracking_frozen = True
                    mpos = self.laser.update_status()
                    print(f"\n🔥 Target {tid}: Firing Spiral Burn...")
                    self.laser.spiral_burn(mpos['x'], mpos['y'], radius=5.0, steps=20, speed=1500)
                    self.save_current_as_anchor(mpos)
                    
                    # Smoke Flush
                    for _ in range(15): self.cap_L.grab(); self.cap_R.grab()
                    _, fL_n = self.cap_L.read(); _, fR_n = self.cap_R.read()
                    self.old_gray_L, self.old_gray_R = cv2.cvtColor(fL_n, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR_n, cv2.COLOR_BGR2GRAY)
                    
                    self.tracking_frozen = False
                    self.current_step += 1

            self.old_gray_L, self.old_gray_R = grayL.copy(), grayR.copy()
        
        self.laser.close()

if __name__ == "__main__":
    B1ProductionMission().start()