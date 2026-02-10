import cv2
import numpy as np
import time
import sys
import json
import subprocess
import select
from pathlib import Path

# Local imports
from motion_helpers import B1LaserController
from cv_helpers import WeedCV

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
YOLO_PT = WEIGHTS_DIR / "yolo_weed.pt"
SNIPER_PT = WEIGHTS_DIR / "sniper.pt"   
CAMERA_SETTINGS = BASE_DIR / "camera_config.json"
HARDWARE_CONFIG = BASE_DIR / "hardware_config.json"

# --- FALLBACKS & CONSTANTS ---
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

class B1ProductionMission:
    def __init__(self):
        print("🔍 Loading Dynamic Hardware Configuration...")
        self.load_hardware_config()

        print(f"🤖 Initializing Laser on {self.port}...")
        try:
            self.laser = B1LaserController(self.port)
        except Exception as e:
            print(f"❌ SERIAL ERROR: {e}"); sys.exit()

        self.cv_L = WeedCV(YOLO_PT, SNIPER_PT)
        self.cv_R = WeedCV(YOLO_PT, SNIPER_PT)

        print(f"📸 Opening Cameras: Left={self.left_id}, Right={self.right_id}")
        
        self.cap_L = cv2.VideoCapture(self.left_id)
        self.cap_R = cv2.VideoCapture(self.right_id)



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

    def load_hardware_config(self):
        with open(HARDWARE_CONFIG, "r") as f:
            cfg = json.load(f)

        self.port = cfg["serial"]["grbl_port"]

        self.left_cam = cfg["cameras"]["left"]
        self.right_cam = cfg["cameras"]["right"]

        self.left_id = self.left_cam["index"]
        self.right_id = self.right_cam["index"]

        self.left_node = self.left_cam["node"]
        self.right_node = self.right_cam["node"]


    def apply_nuclear_hardware_lock(self):
        with open(CAMERA_SETTINGS, "r") as f:
            s = json.load(f)

        for dev in [self.left_node, self.right_node]:
            subprocess.run(
                ["v4l2-ctl", "-d", dev, "-c", "exposure_auto=1"], check=True
            )
            subprocess.run(
                ["v4l2-ctl", "-d", dev, "-c", "exposure_auto_priority=0"], check=True
            )
            subprocess.run(
                ["v4l2-ctl", "-d", dev, "-c", f"exposure_absolute={s['exp']}"], check=True
            )
            subprocess.run(
                ["v4l2-ctl", "-d", dev, "-c", f"gain={s['gain']}"], check=True
            )

        print("✅ Nuclear lock applied to correct physical cameras.")


    def wait_for_idle(self, timeout=30.0):
        start_wait = time.time()
        self.laser.serial.reset_input_buffer() 
        
        while time.time() - start_wait < timeout:
            status = self.laser.send_raw("?")
            if status and "Idle" in status:
                print(f"✅ Machine Idle after {round(time.time()-start_wait, 2)}s")
                return True
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
        # ... existing code ...
        self.path_order = []
        last_xy = curr_xy  # Start measuring from the center or current gantry pos

        while rem:
            # Find the 'i' in 'rem' that is closest to 'last_xy'
            node = min(rem, key=lambda i: np.linalg.norm(np.array(matched[i]['orig_l']) - last_xy))
            
            self.path_order.append(node)
            last_xy = np.array(matched[node]['orig_l']) # Update the reference point
            rem.remove(node)
        
        self.mode = "EXECUTE"; self.current_step = 0
        print(f"\n🚀 Mission Loaded. Targets to neutralize: {len(self.path_order)}")
        return True

    def start(self):
        print("🤖 Homing Gantry...")
        self.laser.home()
        if not self.wait_for_idle(timeout=100):
            print("❌ Homing timed out."); return

        print(f"📍 Moving to Survey Position {START_POS}...")
        self.laser.send_raw(f"G90\nG1 X{START_POS[0]} Y{START_POS[1]} F{TRAVEL_SPEED}")
        if not self.wait_for_idle():
            print("❌ Initial move failed."); return

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
            # Swap displayed counts: show right-camera count in the "L" slot and left-camera count in the "R" slot
            print(f"\r📊 Found: L[{len(self.clicks_R)}] R[{len(self.clicks_L)}] | 'p'+Enter to fire, 'q' to quit", end="")
            
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'p': break
                elif cmd == 'q': print("\n👋 Quitting."); return

        if not self.match_and_optimize(): return
        self.laser.set_acceleration(NORMAL_ACCEL)

        while True:
            retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
            if not (retL and retR): break
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
                    self.current_step += 1; continue

                t = self.lk_targets[tid]
                xl, yl = np.median(t['l'].reshape(-1,2), axis=0)
                xr, yr = np.median(t['r'].reshape(-1,2), axis=0)
                ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
                mag = np.sqrt(ex**2 + ey**2)
                
                if mag > DEADZONE: 
                    self.laser.jog(ex/mag, -ey/mag, np.clip(mag*Kp, 0, MAX_SPEED))
                else:
                    self.laser.stop()
                    self.tracking_frozen = True
                    mpos = self.laser.update_status()
                    print(f"\n🔥 Target {tid}: Firing Spiral Burn...")
                    self.laser.spiral_burn(mpos['x'], mpos['y'], radius=5.0, steps=20, speed=1500)
                    self.save_current_as_anchor(mpos)
                    
                    for _ in range(15): self.cap_L.grab(); self.cap_R.grab()
                    _, fL_n = self.cap_L.read(); _, fR_n = self.cap_R.read()
                    self.old_gray_L, self.old_gray_R = cv2.cvtColor(fL_n, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR_n, cv2.COLOR_BGR2GRAY)
                    
                    self.tracking_frozen = False
                    self.current_step += 1

            self.old_gray_L, self.old_gray_R = grayL.copy(), grayR.copy()
        
        self.laser.close()

if __name__ == "__main__":
    B1ProductionMission().start()