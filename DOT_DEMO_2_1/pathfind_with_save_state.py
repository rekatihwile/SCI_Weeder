import cv2
import numpy as np
import time
import sys
from helpers import B1LaserController

# --- CONFIGURATION ---
PORT = '/dev/ttyUSB0'
W, H = 640, 480
TARGET_Y_L, TARGET_Y_R = 242, 240
START_POS = (225, 220)

Kp, Kd = 60.0, 30.0 
DEADZONE = 5    
MAX_SPEED, TRAVEL_SPEED = 12000, 12000
DWELL_TIME = .5  # Time nozzle stays over target
SETTLE_TIME = 0.5  # Seconds to wait for smoke/vibration to clear post-fire

# --- RECOVERY & MOTION ---
RECOVERY_SPEED = 18000 
RECOVERY_ACCEL = 7000  
NORMAL_ACCEL = 2000

class B1ProductionMission:
    def __init__(self):
        print("🔍 Initializing Hardware...")
        try:
            self.laser = B1LaserController(PORT)
        except Exception as e:
            print(f"❌ SERIAL ERROR: {e}"); sys.exit()

        self.cap_R, self.cap_L = cv2.VideoCapture(0), cv2.VideoCapture(2)
        for cap in [self.cap_L, self.cap_R]:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

    def get_cluster(self, x, y):
        pts = [[float(x + i), float(y + j)] for i in [-3, 0, 3] for j in [-3, 0, 3]]
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def save_current_as_anchor(self, mpos):
        """Saves current pose and scores target quality (mag) for optimal recovery later."""
        anchor_snapshot = {}
        for tid, data in self.lk_targets.items():
            xl, yl = np.median(data['l'].reshape(-1,2), axis=0)
            xr, yr = np.median(data['r'].reshape(-1,2), axis=0)
            ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
            mag = np.sqrt(ex**2 + ey**2)
            anchor_snapshot[tid] = {
                'l': data['l'].copy(), 'r': data['r'].copy(),
                'orig_l': data['orig_l'], 'mag': mag 
            }
        anchor_id = self.path_order[self.current_step]
        self.spatial_anchors[anchor_id] = {'mpos': (mpos['x'], mpos['y']), 'snapshot': anchor_snapshot}
        self.current_anchor_id = anchor_id
        return anchor_snapshot

    def match_and_optimize(self):
        print("🔍 Overlap Search: Finding Best Constellation Fit...")
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
            ex, ey = -(p_l[0] - W + p_r[0]), -(((p_l[1] - TARGET_Y_L) + (p_r[1] - TARGET_Y_R)) / 2)
            matched[m_id] = {'l': self.get_cluster(*p_l), 'r': self.get_cluster(*p_r), 'orig_l': p_l}
            initial_snapshot[m_id] = {
                'l': matched[m_id]['l'].copy(), 
                'r': matched[m_id]['r'].copy(), 
                'orig_l': p_l,
                'mag': np.sqrt(ex**2+ey**2)
            }

        self.lk_targets = matched
        self.spatial_anchors["START"] = {'mpos': curr_mpos, 'snapshot': initial_snapshot}
        
        rem = list(matched.keys()); curr_xy = np.array([W//2, H//2]); self.path_order = []
        while rem:
            node = min(rem, key=lambda i: np.linalg.norm(np.array(matched[i]['orig_l']) - curr_xy))
            self.path_order.append(node); curr_xy = np.array(matched[node]['orig_l']); rem.remove(node)
        self.mode = "EXECUTE"; self.current_step = 0
        print(f"🚀 Mission Ready. Path: {self.path_order}")

    def start(self):
        self.laser.set_acceleration(3000); self.laser.home(); time.sleep(1)
        self.laser.send_raw(f"G90\nG1 X{START_POS[0]} Y{START_POS[1]} F{TRAVEL_SPEED}")
        self.laser.set_acceleration(NORMAL_ACCEL)
        cv2.namedWindow("Left"); cv2.namedWindow("Right")
        def h_L(e,x,y,f,p): 
            if e==cv2.EVENT_LBUTTONDOWN and self.mode=="SURVEY": self.clicks_L.append((x,y))
        def h_R(e,x,y,f,p): 
            if e==cv2.EVENT_LBUTTONDOWN and self.mode=="SURVEY": self.clicks_R.append((x,y))
        cv2.setMouseCallback("Left", h_L); cv2.setMouseCallback("Right", h_R)

        while True:
            retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
            if not (retL and retR): break
            grayL, grayR = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
            
            # --- PERMANENT CROSSHAIRS ---
            cv2.drawMarker(fL, (320, TARGET_Y_L), (0,0,255), 2, 25, 2)
            cv2.drawMarker(fR, (320, TARGET_Y_R), (0,0,255), 2, 25, 2)

            if self.mode == "SURVEY":
                cv2.putText(fL, "SURVEY", (10, 30), 0, 0.7, (255, 0, 255), 2)
                for i, (x,y) in enumerate(self.clicks_L):
                    cv2.drawMarker(fL,(x,y),(255,0,255),1,10,1)
                    cv2.putText(fL, str(i), (x+5, y-5), 0, 0.4, (255,0,255), 1)
                for x,y in self.clicks_R: cv2.drawMarker(fR,(x,y),(255,0,255),1,10,1)

            elif self.mode in ["EXECUTE", "RECOVER", "IDLE"]:
                if self.old_gray_L is not None and not self.tracking_frozen and self.mode != "RECOVER":
                    new_lk = {}
                    for tid, data in self.lk_targets.items():
                        pL, stL, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_L, grayL, data['l'], None)
                        pR, stR, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_R, grayR, data['r'], None)
                        if np.sum(stL) > 4 and np.sum(stR) > 4:
                            new_lk[tid] = {'l': pL, 'r': pR, 'orig_l': data['orig_l']}
                            xl, yl = np.median(pL.reshape(-1,2), axis=0); xr, yr = np.median(pR.reshape(-1,2), axis=0)
                            is_c = (self.current_step < len(self.path_order) and tid == self.path_order[self.current_step])
                            clr = (0, 255, 0) if is_c else (0, 165, 255)
                            cv2.circle(fL, (int(xl), int(yl)), 6, clr, 1)
                            cv2.putText(fL, f"ID:{tid}", (int(xl)+8, int(yl)-8), 0, 0.4, clr, 1)
                            cv2.circle(fR, (int(xr), int(yr)), 6, clr, 1)
                    self.lk_targets = new_lk

                if self.mode == "EXECUTE":
                    if self.current_step >= len(self.path_order):
                        print("🏁 Mission Complete."); self.mode = "IDLE"; continue
                    
                    tid = self.path_order[self.current_step]
                    if tid not in self.lk_targets:
                        # --- GOLD POSE RECOVERY ---
                        best_k, best_m = None, float('inf')
                        for k, anc in self.spatial_anchors.items():
                            if tid in anc['snapshot'] and anc['snapshot'][tid]['mag'] < best_m:
                                best_m, best_k = anc['snapshot'][tid]['mag'], k
                        if best_k:
                            self.current_anchor_id = best_k
                            print(f"⏪ [GOLD RECOVERY] ID:{best_k} (Mag:{best_m:.1f})")
                            self.laser.set_acceleration(RECOVERY_ACCEL)
                            self.laser.send_raw(f"G90\nG1 X{self.spatial_anchors[best_k]['mpos'][0]} Y{self.spatial_anchors[best_k]['mpos'][1]} F{RECOVERY_SPEED}")
                            self.mode = "RECOVER"; self.recovery_settle_start = None; continue
                        else: self.current_step += 1; continue

                    t = self.lk_targets[tid]
                    xl, yl = np.median(t['l'].reshape(-1,2), axis=0); xr, yr = np.median(t['r'].reshape(-1,2), axis=0)
                    ex, ey = -(xl - W + xr), -(((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2)
                    mag = np.sqrt(ex**2 + ey**2)
                    if mag > DEADZONE: self.laser.jog(ex/mag, -ey/mag, np.clip(mag*Kp, 0, MAX_SPEED))
                    else:
                        if self.dwell_start is None:
                            self.laser.stop(); self.tracking_frozen = True; self.dwell_start = time.time()
                            mpos = self.laser.update_status(); self.save_current_as_anchor(mpos)
                        
                        if not self.laser_fired: self.laser.fire_low(); self.laser_fired = True
                        
                        # Wait for fire duration + settle time (smoke clearance)
                        if (time.time() - self.dwell_start) >= (DWELL_TIME + SETTLE_TIME):
                            # --- ATOMIC RE-SEED ---
                            for _ in range(5): self.cap_L.grab(); self.cap_R.grab()
                            _, fL_n = self.cap_L.read(); _, fR_n = self.cap_R.read()
                            
                            anc = self.spatial_anchors[self.current_anchor_id]
                            for sid, d in anc['snapshot'].items():
                                self.lk_targets[sid] = {'l': d['l'].copy(), 'r': d['r'].copy(), 'orig_l': d['orig_l']}
                            
                            self.old_gray_L, self.old_gray_R = cv2.cvtColor(fL_n, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR_n, cv2.COLOR_BGR2GRAY)
                            self.tracking_frozen = False; self.current_step += 1; self.dwell_start = None; self.laser_fired = False

                elif self.mode == "RECOVER":
                    # 1. Faster polling (0.1s instead of 0.4s)
                    if time.time() - self.last_status_check > 0.1:
                        self.last_status_check = time.time()
                        
                        # Use a lightweight check instead of a full status update
                        status = self.laser.send_raw("?") or ""
                        if "Idle" in status:
                            if self.recovery_settle_start is None:
                                self.recovery_settle_start = time.time()
                            
                            # 2. OVERLAP: Flush buffers WHILE the gantry is settling
                            self.cap_L.grab()
                            self.cap_R.grab()

                            # 3. Reduced Settle Time
                            if time.time() - self.recovery_settle_start > 0.3:
                                # Final read of fresh frames
                                _, fL_n = self.cap_L.read()
                                _, fR_n = self.cap_R.read()
                                
                                anc = self.spatial_anchors[self.current_anchor_id]
                                self.lk_targets = {}
                                for sid, d in anc['snapshot'].items():
                                    self.lk_targets[sid] = {'l': d['l'].copy(), 'r': d['r'].copy(), 'orig_l': d['orig_l']}
                                
                                self.old_gray_L = cv2.cvtColor(fL_n, cv2.COLOR_BGR2GRAY)
                                self.old_gray_R = cv2.cvtColor(fR_n, cv2.COLOR_BGR2GRAY)
                                
                                self.laser.set_acceleration(NORMAL_ACCEL)
                                self.mode = "EXECUTE"
                                print(f"✅ Fast Re-Sync Complete: {time.time() - self.recovery_settle_start:.2f}s")

            cv2.imshow("Left", fL); cv2.imshow("Right", fR)
            if self.mode != "RECOVER": self.old_gray_L, self.old_gray_R = grayL.copy(), grayR.copy()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'): self.match_and_optimize()
            elif key == ord('c'): self.clicks_L, self.clicks_R, self.lk_targets, self.mode = [], [], {}, "SURVEY"
            elif key == 27: break
        self.laser.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    B1ProductionMission().start()