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
WIN_FEED = "Sequential Targeting PID"
SETTINGS_FILE = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/b1_test/PID/blob_settings.json"

# --- SOFT LIMITS (mm) ---
MIN_X, MAX_X = 50, 400
MIN_Y, MAX_Y = 50, 400

# --- PID STABILITY ---
KP, KI, KD = 0.18, 0.0, 0.01 
PX_TO_MM = 0.08      
JOG_FEED = 3000      
DEADZONE = 6         
SMOOTHING = 0.4      
LOOP_HZ = 20         
MATCH_TOLERANCE = 40 

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
        with open(SETTINGS_FILE, 'r') as f:
            cfg = json.load(f)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea, params.maxArea = float(cfg["MinArea"]), float(cfg["MaxArea"])
    params.filterByCircularity = True
    params.minCircularity = max(0.01, cfg["Circularity"] / 100.0)
    params.filterByConvexity = True
    params.minConvexity = max(0.01, cfg["Convexity"] / 100.0)
    params.filterByInertia = True
    params.minInertiaRatio = max(0.01, cfg["Inertia"] / 100.0)
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
    print("🏠 Homing Gantry...")
    laser.send_blocking("$H") 
    laser.move_to(START_X, START_Y, speed=7000)
    
    detector = get_detector_from_json()
    pid_x, pid_y = PIDController(KP, KI, KD), PIDController(KP, KI, KD)

    cap_l = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
    cap_r = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)
    for c in [cap_l, cap_r]:
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, W); c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    # --- STATE MANAGEMENT ---
    tracking_active = False
    current_target_id = 0
    dwell_start_time = None
    smooth_dx, smooth_dy = 0, 0
    last_time = time.time()
    frame_time = 1.0 / LOOP_HZ

    print("\n--- READY ---")
    print("Press [p] to begin SEQUENTIAL TARGETING | [q] Quit")

    while True:
        loop_start = time.time()
        ret_l, f_l = cap_l.read(); ret_r, f_r = cap_r.read()
        if not ret_l or not ret_r: break

        curr_time = time.time()
        dt = max(curr_time - last_time, 0.001)
        last_time = curr_time
        
        mpos = laser.update_status()
        k_l, k_r = detector.detect(f_l), detector.detect(f_r)

        matches = {} # To store ID mapping for current frame

        if k_l and k_r:
            pts_l = np.array([kp.pt for kp in k_l])
            pts_r = np.array([kp.pt for kp in k_r])
            best_shift, _ = find_global_best_shift(pts_l, pts_r)
            
            # --- MATCHING & LABELING ---
            shifted_l = pts_l - np.array([best_shift, 0])
            for i, p_l_s in enumerate(shifted_l):
                dists = np.linalg.norm(pts_r - p_l_s, axis=1)
                idx = np.argmin(dists)
                
                if dists[idx] < MATCH_TOLERANCE:
                    matches[i] = (pts_l[i], pts_r[idx])
                    # Always draw labels
                    p1 = tuple(pts_l[i].astype(int))
                    p2 = tuple(pts_r[idx].astype(int))
                    color = (0, 255, 0) if (tracking_active and current_target_id == i) else (255, 255, 255)
                    cv2.circle(f_l, p1, 15, color, 2)
                    cv2.putText(f_l, f"ID:{i}", (p1[0]-15, p1[1]-20), 0, 0.5, color, 2)
                    cv2.circle(f_r, p2, 15, color, 2)

        # --- SEQUENTIAL PID LOGIC ---
        if tracking_active and mpos:
            if current_target_id in matches:
                p_l, p_r = matches[current_target_id]
                dist_left, dist_right = W - p_l[0], p_r[0]
                error_x = dist_right - dist_left
                error_y = CENTER_Y - ((p_l[1] + p_r[1]) / 2.0)

                # PID & Smoothing
                raw_dx = pid_x.compute(error_x, dt) * PX_TO_MM
                raw_dy = pid_y.compute(error_y, dt) * PX_TO_MM
                smooth_dx = (SMOOTHING * raw_dx) + ((1.0 - SMOOTHING) * smooth_dx)
                smooth_dy = (SMOOTHING * raw_dy) + ((1.0 - SMOOTHING) * smooth_dy)

                # Check if Centered
                if abs(error_x) < DEADZONE and abs(error_y) < DEADZONE:
                    if dwell_start_time is None:
                        dwell_start_time = time.time()
                        print(f"✅ ID:{current_target_id} Locked. Dwelling...")
                    
                    # Dwell Check
                    elif (time.time() - dwell_start_time) > 2.0:
                        print(f"🏁 ID:{current_target_id} Done. Switching to ID:{current_target_id + 1}")
                        current_target_id += 1
                        dwell_start_time = None
                        # Reset PID for new target
                        pid_x.prev_error, pid_y.prev_error = 0, 0
                else:
                    # Move towards target
                    dwell_start_time = None # Reset dwell if we drift out
                    if not laser.jog_in_flight:
                        safe_x = np.clip(mpos['x'] + smooth_dx, MIN_X, MAX_X)
                        safe_y = np.clip(mpos['y'] + smooth_dy, MIN_Y, MAX_Y)
                        dx, dy = safe_x - mpos['x'], safe_y - mpos['y']
                        laser.send_jog(dx, dy, feed=JOG_FEED)
            else:
                # If target ID is missing, but higher IDs might exist
                if current_target_id < len(matches) + current_target_id:
                     # This helps skip a flickering ID but usually we just wait
                     pass

        # UI
        view = np.hstack((f_l, f_r))
        status_txt = f"TARGETING ID:{current_target_id}" if tracking_active else "IDLE: PRESS 'P'"
        cv2.putText(view, status_txt, (20, 40), 0, 0.8, (0, 255, 255), 2)
        cv2.imshow(WIN_FEED, view)
        
        elapsed = time.time() - loop_start
        if elapsed < frame_time: time.sleep(frame_time - elapsed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): 
            tracking_active = True
            current_target_id = 0
            print("🚀 Starting Sequence...")

    laser.close()
    cap_l.release(); cap_r.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()