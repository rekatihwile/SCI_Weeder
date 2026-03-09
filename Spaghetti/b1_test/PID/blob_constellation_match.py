import cv2
import numpy as np
import json
import os

# === CONFIGURATION ===
CAM_LEFT, CAM_RIGHT = 2, 0  # Fixed: Left is 2, Right is 0
W, H = 640, 480
SETTINGS_FILE = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/b1_test/PID/blob_settings.json"
WIN_FEED = "Matched Stereo Output"

# --- STEREO LOGIC PARAMS ---
MIN_SHIFT, MAX_SHIFT = 10, 250
MATCH_TOLERANCE = 25  # Pixels

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            print(f"✅ Loading parameters from {SETTINGS_FILE}")
            return json.load(f)
    print("⚠️ Settings file not found. Using hardcoded defaults.")
    return {"MinArea": 40, "MaxArea": 1000, "Circularity": 45, "Convexity": 10, "Inertia": 20}

def create_fixed_detector(cfg):
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
    best_shift, min_err = 0, float('inf')
    for shift in range(MIN_SHIFT, MAX_SHIFT, 2):
        shifted_l = pts_l - np.array([shift, 0])
        # Find avg distance to nearest neighbor in Right view
        total_err = sum(np.min(np.linalg.norm(pts_r - p_l, axis=1)) for p_l in shifted_l)
        avg_err = total_err / len(pts_l)
        if avg_err < min_err:
            min_err, best_shift = avg_err, shift
    return best_shift, min_err

def main():
    config = load_settings()
    detector = create_fixed_detector(config)

    cap_l = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
    cap_r = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)

    for c in [cap_l, cap_r]:
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        c.set(cv2.CAP_PROP_EXPOSURE, 550)
        c.set(cv2.CAP_PROP_GAIN, 44)
        c.set(cv2.CAP_PROP_BRIGHTNESS, 50)
        c.set(cv2.CAP_PROP_CONTRAST, 40)
        c.set(cv2.CAP_PROP_SATURATION, 63)
        c.set(cv2.CAP_PROP_AUTO_WB, 0)            
        c.set(cv2.CAP_PROP_WB_TEMPERATURE, 4079)

    print("🚀 Matcher Running. Synchronizing ID constellations...")

    while True:
        ret_l, f_l = cap_l.read()
        ret_r, f_r = cap_r.read()
        if not ret_l or not ret_r: break

        k_l, k_r = detector.detect(f_l), detector.detect(f_r)

        if k_l and k_r:
            pts_l = np.array([kp.pt for kp in k_l])
            pts_r = np.array([kp.pt for kp in k_r])

            # 1. Align the constellations
            best_shift, err = find_global_best_shift(pts_l, pts_r)
            
            # 2. Match and Label
            shifted_l = pts_l - np.array([best_shift, 0])
            for i, p_l_s in enumerate(shifted_l):
                dists = np.linalg.norm(pts_r - p_l_s, axis=1)
                idx = np.argmin(dists)
                
                if dists[idx] < MATCH_TOLERANCE:
                    p1 = tuple(pts_l[i].astype(int))
                    p2 = tuple(pts_r[idx].astype(int))
                    
                    # Draw on Left
                    cv2.circle(f_l, p1, 15, (0, 255, 0), 2)
                    cv2.putText(f_l, f"ID:{i}", (p1[0]-15, p1[1]-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw on Right (Using the same ID 'i' for consistency)
                    cv2.circle(f_r, p2, 15, (0, 255, 0), 2)
                    cv2.putText(f_r, f"ID:{i}", (p2[0]-15, p2[1]-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show shift data for debugging
        cv2.putText(f_l, f"Shift: {best_shift}px", (10, 30), 0, 0.6, (255, 255, 255), 2)

        cv2.imshow(WIN_FEED, np.hstack((f_l, f_r)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_l.release(); cap_r.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()