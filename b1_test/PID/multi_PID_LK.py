import cv2
import numpy as np
import time
import sys
import os
import json

# Add the specific path to system so we can import helpers
sys.path.append('/home/laser/Documents/Laser_Workspace/SCI_Weeder/b1_test')
from SCI_Weeder.b1_test.PID.helpers import LaserHelper 

# === CONFIGURATION ===
CAM_LEFT, CAM_RIGHT = 2, 0  
W, H = 640, 480
CENTER_X, CENTER_Y = W // 2, H // 2
START_X, START_Y = 225, 220
WIN_FEED = "Persistent LK Tracker"
SETTINGS_FILE = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/b1_test/PID/blob_settings.json"

# --- SOFT LIMITS (mm) ---
MIN_X, MAX_X = 50, 400
MIN_Y, MAX_Y = 50, 400

# --- PID & LK PARAMS ---
KP, KI, KD = 0.18, 0.0, 0.01 
PX_TO_MM = 0.08      
JOG_FEED = 3000      
DEADZONE = 6         
SMOOTHING = 0.4      
LOOP_HZ = 20         
MATCH_TOLERANCE = 40 

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

def get_detector_from_json():
    if not os.path.exists(SETTINGS_FILE):
        cfg = {"MinArea": 40, "MaxArea": 1000, "Circularity": 45, "Convexity": 10, "Inertia": 20}
    else:
        with open(SETTINGS_FILE, 'r') as f: cfg = json.load(f)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea, params.maxArea = float(cfg["MinArea"]), float(cfg["MaxArea"])
    return cv2.SimpleBlobDetector_create(params)

def find_global_best_shift(pts_l, pts_r):
    if len(pts_l) == 0 or len(pts_r) == 0: return 0, 999
    best_shift, min_err = 0, float('inf')
    for shift in range(10, 250, 4):
        shifted_l = pts_l - np.array([shift, 0])
        total_err = sum(np.min(np.linalg.norm(pts_r - p_l, axis=1)) for p_l in shifted_l)
        avg_err = total_err / len(pts_l)
        if avg_err < min_err:
            min_err, best_shift = avg_err, shift
    return best_shift, min_err

def main():
    laser = LaserHelper()
    print("🏠 Homing...")
    laser.send_blocking("$H") 
    laser.move_to(START_X, START_Y, speed=7000)
    
    detector = get_detector_from_json()
    pid_x, pid_y = PIDController(KP, KI, KD), PIDController(KP, KI, KD)

    cap_l = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
    cap_r = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)
    for c in [cap_l, cap_r]:
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, W); c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    # --- TRACKING STATE ---
    tracking_active = False
    current_target_id = 0
    dwell_start_time = None
    smooth_dx, smooth_dy = 0, 0
    
    lk_targets = {} # { id: {'l': pt, 'r': pt} }
    prev_gray_l, prev_gray_r = None, None
    last_time = time.time()
    frame_time = 1.0 / LOOP_HZ

    print("\n--- READY ---")
    print("Press [p] to convert all Blobs to LK Tracks and Start.")

    while True:
        loop_start = time.time()
        ret_l, f_l = cap_l.read(); ret_r, f_r = cap_r.read()
        if not ret_l or not ret_r: break

        gray_l = cv2.cvtColor(f_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(f_r, cv2.COLOR_BGR2GRAY)
        dt = max(time.time() - last_time, 0.001)
        last_time = time.time()
        
        mpos = laser.update_status()

        if not tracking_active:
            # DISCOVERY MODE: Just show blobs and wait for 'p'
            k_l, k_r = detector.detect(f_l), detector.detect(f_r)
            if k_l and k_r:
                pts_l, pts_r = np.array([kp.pt for kp in k_l]), np.array([kp.pt for kp in k_r])
                shift, _ = find_global_best_shift(pts_l, pts_r)
                
                shifted_l = pts_l - np.array([shift, 0])
                current_blobs = {}
                for i, p_l_s in enumerate(shifted_l):
                    dists = np.linalg.norm(pts_r - p_l_s, axis=1)
                    idx = np.argmin(dists)
                    if dists[idx] < MATCH_TOLERANCE:
                        p1, p2 = pts_l[i], pts_r[idx]
                        current_blobs[i] = {'l': p1, 'r': p2}
                        cv2.circle(f_l, tuple(p1.astype(int)), 10, (255, 255, 255), 1)
                        cv2.putText(f_l, f"ID:{i}", (int(p1[0]), int(p1[1]-15)), 0, 0.5, (255,255,255), 1)

        else:
            # TARGETING MODE: Use LK Optical Flow
            if prev_gray_l is not None:
                new_targets = {}
                for tid, pts in lk_targets.items():
                    # Update Left
                    p_l_in = np.array([[pts['l']]], dtype=np.float32)
                    new_l, stat_l, _ = cv2.calcOpticalFlowPyrLK(prev_gray_l, gray_l, p_l_in, None, **LK_PARAMS)
                    # Update Right
                    p_r_in = np.array([[pts['r']]], dtype=np.float32)
                    new_r, stat_r, _ = cv2.calcOpticalFlowPyrLK(prev_gray_r, gray_r, p_r_in, None, **LK_PARAMS)

                    if stat_l[0][0] and stat_r[0][0]:
                        new_targets[tid] = {'l': new_l[0][0], 'r': new_r[0][0]}
                        # UI Drawing
                        color = (0, 255, 0) if tid == current_target_id else (0, 150, 255)
                        cv2.circle(f_l, tuple(new_l[0][0].astype(int)), 12, color, 2)
                        cv2.putText(f_l, f"LK:{tid}", (int(new_l[0][0][0]), int(new_l[0][0][1]-15)), 0, 0.5, color, 2)
                
                lk_targets = new_targets

            # PID Execution for current ID
            if current_target_id in lk_targets and mpos:
                p_l, p_r = lk_targets[current_target_id]['l'], lk_targets[current_target_id]['r']
                ex = p_r[0] - (W - p_l[0]) 
                ey = CENTER_Y - ((p_l[1] + p_r[1]) / 2.0)

                raw_dx = pid_x.compute(ex, dt) * PX_TO_MM
                raw_dy = pid_y.compute(ey, dt) * PX_TO_MM
                smooth_dx = (0.4 * raw_dx) + (0.6 * smooth_dx)
                smooth_dy = (0.4 * raw_dy) + (0.6 * smooth_dy)

                if abs(ex) < DEADZONE and abs(ey) < DEADZONE:
                    if dwell_start_time is None: dwell_start_time = time.time()
                    elif (time.time() - dwell_start_time) > 2.0:
                        print(f"🏁 Target {current_target_id} Done.")
                        current_target_id += 1
                        dwell_start_time = None
                else:
                    dwell_start_time = None
                    if not laser.jog_in_flight:
                        safe_x = np.clip(mpos['x'] + smooth_dx, MIN_X, MAX_X)
                        safe_y = np.clip(mpos['y'] + smooth_dy, MIN_Y, MAX_Y)
                        laser.send_jog(safe_x - mpos['x'], safe_y - mpos['y'], feed=JOG_FEED)
            
            elif tracking_active and current_target_id >= len(lk_targets) + current_target_id:
                print("✅ All Targets Processed.")
                tracking_active = False

        prev_gray_l, prev_gray_r = gray_l.copy(), gray_r.copy()
        view = np.hstack((f_l, f_r))
        cv2.imshow(WIN_FEED, view)
        
        elapsed = time.time() - loop_start
        if elapsed < frame_time: time.sleep(frame_time - elapsed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): 
            if not tracking_active:
                # HANDOVER: Copy current blobs into persistent LK targets
                lk_targets = current_blobs.copy()
                tracking_active = True
                current_target_id = 0
                print(f"🚀 LK Persistent Tracking Engaged for {len(lk_targets)} targets.")

    laser.close()
    cap_l.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()