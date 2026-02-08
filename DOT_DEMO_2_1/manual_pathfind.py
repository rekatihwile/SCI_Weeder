import cv2
import numpy as np
import time
from helpers import B1LaserController

# --- CONFIGURATION ---
PORT = '/dev/ttyUSB0'
W, H = 640, 480
TARGET_X, TARGET_Y = 320, 240
START_POS = (225, 220)
Kp, Kd = 36.0, 45.0
DEADZONE = 4
MAX_SPEED, TRAVEL_SPEED, PLAYBACK_SPEED = 10000, 12000, 5000
DWELL_TIME = 2.0

class B1LaserMission:
    def __init__(self):
        self.laser = B1LaserController(PORT)
        self.cap_R, self.cap_L = cv2.VideoCapture(0), cv2.VideoCapture(2)
        self.clicks_L, self.clicks_R, self.lk_targets = [], [], {}
        self.path_order, self.saved_coords = [], {}
        self.current_step, self.mode = 0, "SURVEY"
        self.old_gray_L, self.old_gray_R, self.last_mag, self.dwell_start = None, None, 0.0, None

    def wait_for_idle(self):
        """Blocks until the gantry status is 'Idle'."""
        print("Waiting for gantry to settle...")
        while True:
            # Check status via helper's ? query
            with self.laser.lock:
                self.laser.serial.reset_input_buffer()
                self.laser.serial.write(b'?')
                line = self.laser.serial.readline().decode().strip()
            if "Idle" in line: break
            time.sleep(0.1)

    def match_and_optimize(self):
        if not (self.clicks_L and self.clicks_R): return
        print("\n!!! SAFETY PROTOCOL: LASER ARMED !!!")
        if input("Wear glasses. Type 'YES': ").strip().upper() != "YES": return

        # Constellation Match
        pts_l, pts_r = np.array(self.clicks_L), np.array(self.clicks_R)
        best_dx, best_dy, min_err = 0, 0, float('inf')
        for dx in range(50, 251, 4):
            for dy in range(-15, 16, 3):
                shifted = pts_l - [dx, dy]
                err = sum(np.min(np.linalg.norm(pts_r - p, axis=1)) for p in shifted)
                if err < min_err: min_err, best_dx, best_dy = err, dx, dy
        
        matched = {}
        for i, p_l in enumerate(pts_l):
            p_l_s = p_l - [best_dx, best_dy]
            dists = np.linalg.norm(pts_r - p_l_s, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < 40:
                pts = [[float(p_l[0]+x), float(p_l[1]+y)] for x in range(-12,12,4) for y in range(-12,12,4)]
                pts_r_cluster = [[float(pts_r[idx][0]+x), float(pts_r[idx][1]+y)] for x in range(-12,12,4) for y in range(-12,12,4)]
                matched[i] = {
                    'l': np.array(pts, dtype=np.float32).reshape(-1,1,2),
                    'r': np.array(pts_r_cluster, dtype=np.float32).reshape(-1,1,2),
                    'orig_l': p_l
                }
        
        self.lk_targets = matched
        remaining = list(matched.keys())
        curr_xy = np.array([W//2, H//2])
        while remaining:
            node = min(remaining, key=lambda i: np.linalg.norm(matched[i]['orig_l'] - curr_xy))
            self.path_order.append(node); curr_xy = matched[node]['orig_l']; remaining.remove(node)

        self.mode = "EXECUTE"; self.current_step = 0

    def start(self):
        self.laser.set_acceleration(2500); self.laser.home(); time.sleep(1)
        self.laser.send_raw(f"G90\nG1 X{START_POS[0]} Y{START_POS[1]} F{TRAVEL_SPEED}")
        cv2.namedWindow("Left"); cv2.namedWindow("Right")
        cv2.setMouseCallback("Left", lambda e,x,y,f,p: self.clicks_L.append((x,y)) if e==cv2.EVENT_LBUTTONDOWN and self.mode=="SURVEY" else None)
        cv2.setMouseCallback("Right", lambda e,x,y,f,p: self.clicks_R.append((x,y)) if e==cv2.EVENT_LBUTTONDOWN and self.mode=="SURVEY" else None)

        while True:
            retL, fL = self.cap_L.read(); retR, fR = self.cap_R.read()
            if not (retL and retR): break
            grayL, grayR = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
            cv2.drawMarker(fL, (TARGET_X, TARGET_Y), (0,0,255), 2, 25, 2)
            cv2.drawMarker(fR, (TARGET_X, TARGET_Y), (0,0,255), 2, 25, 2)

            if self.mode == "SURVEY":
                for x,y in self.clicks_L: cv2.drawMarker(fL,(x,y),(255,0,255),1,10,1)
                for x,y in self.clicks_R: cv2.drawMarker(fR,(x,y),(255,0,255),1,10,1)

            elif self.mode == "EXECUTE":
                # LK Tracking
                if self.old_gray_L is not None:
                    new_lk = {}
                    for tid, data in self.lk_targets.items():
                        pL, stL, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_L, grayL, data['l'], None)
                        pR, stR, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_R, grayR, data['r'], None)
                        if stL[0] and stR[0]:
                            new_lk[tid] = {'l': pL, 'r': pR, 'orig_l': data['orig_l']}
                            xl, yl = np.median(pL.reshape(-1,2), axis=0); xr, yr = np.median(pR.reshape(-1,2), axis=0)
                            is_curr = (self.current_step < len(self.path_order) and tid == self.path_order[self.current_step])
                            color = (0, 255, 0) if is_curr else (0, 165, 255)
                            for img, (cx, cy) in [(fL, (xl, yl)), (fR, (xr, yr))]:
                                cv2.circle(img, (int(cx), int(cy)), 5, color, -1 if is_curr else 1)
                                cv2.putText(img, str(tid), (int(cx), int(cy-15)), 0, 0.5, color, 1)
                    self.lk_targets = new_lk

                if self.current_step < len(self.path_order):
                    tid = self.path_order[self.current_step]
                    if tid in self.lk_targets:
                        t = self.lk_targets[tid]
                        xl, yl = np.median(t['l'].reshape(-1,2), axis=0); xr, yr = np.median(t['r'].reshape(-1,2), axis=0)
                        ex, ey = xl - (W - xr), ((yl + yr) / 2) - TARGET_Y
                        mag = np.sqrt(ex**2 + ey**2)
                        if mag > DEADZONE:
                            uv = np.array([ex/mag, -ey/mag]); feed = np.clip((mag*Kp) + (mag - self.last_mag)*Kd, 0, MAX_SPEED)
                            self.laser.jog_clear(uv[0], uv[1], feed); self.last_mag = mag; self.dwell_start = None
                        else:
                            if self.dwell_start is None:
                                self.laser.stop(); time.sleep(0.2)
                                mpos = self.laser.update_status()
                                if mpos: self.saved_coords[tid] = (mpos['x'], mpos['y'])
                                self.laser.fire_low() # Hard reset fire
                                self.dwell_start = time.time()
                            rem = max(0, DWELL_TIME - (time.time() - self.dwell_start))
                            cv2.putText(fL, f"FIRE ID:{tid} {rem:.1f}s", (180, 50), 0, 0.8, (0, 0, 255), 2)
                            if rem <= 0: self.current_step += 1; self.dwell_start = None
                else:
                    # Transition Sequence
                    print("🏁 EXECUTE DONE. HOME -> START -> BLIND")
                    self.laser.send_raw("G90\nG1 X0 Y0 F12000"); self.wait_for_idle()
                    self.laser.send_raw(f"G1 X{START_POS[0]} Y{START_POS[1]} F12000"); self.wait_for_idle()
                    self.mode = "BLIND"; self.current_step = 0

            elif self.mode == "BLIND":
                if self.current_step < len(self.path_order):
                    tid = self.path_order[self.current_step]
                    if tid in self.saved_coords:
                        tx, ty = self.saved_coords[tid]
                        if self.dwell_start is None:
                            print(f"🙈 Blind Playback ID:{tid}")
                            self.laser.send_raw(f"G90\nG1 X{tx} Y{ty} F{PLAYBACK_SPEED}"); self.wait_for_idle()
                            self.laser.fire_low()
                            self.dwell_start = time.time()
                        rem = max(0, DWELL_TIME - (time.time() - self.dwell_start))
                        cv2.putText(fL, f"BLIND ID:{tid} {rem:.1f}s", (180, 50), 0, 0.8, (0, 0, 255), 2)
                        if rem <= 0: self.current_step += 1; self.dwell_start = None
                else:
                    self.laser.send_raw("G1 X0 Y0 F12000"); self.mode = "SURVEY"

            cv2.imshow("Left", fL); cv2.imshow("Right", fR)
            self.old_gray_L, self.old_gray_R = grayL.copy(), grayR.copy()
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key == ord('p'): self.match_and_optimize()
            elif key == ord('c'): self.clicks_L, self.clicks_R, self.mode = [], [], "SURVEY"

        self.laser.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    B1LaserMission().start()