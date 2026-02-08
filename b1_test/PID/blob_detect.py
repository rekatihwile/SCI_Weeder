import cv2
import numpy as np
import json
import os

# === CONFIGURATION ===
CAM_LEFT = 0
CAM_RIGHT = 2
W, H = 640, 480
SETTINGS_FILE = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/b1_test/PID/blob_settings.json"
WIN_CONTROLS = "Blob Controls"
WIN_FEED = "Stereo Blob Detection"

def nothing(x):
    pass

def load_settings():
    """Loads settings from JSON or returns default starting values."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                print(f"📂 Loading settings from {SETTINGS_FILE}")
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading JSON: {e}")
    
    # Absolute defaults if no file exists
    return {
        "MinArea": 100,
        "MaxArea": 2000,
        "Circularity": 10,
        "Convexity": 10,
        "Inertia": 1
    }

def save_settings():
    """Saves current trackbar positions to a JSON file."""
    data = {
        "MinArea": cv2.getTrackbarPos("MinArea", WIN_CONTROLS),
        "MaxArea": cv2.getTrackbarPos("MaxArea", WIN_CONTROLS),
        "Circularity": cv2.getTrackbarPos("Circularity", WIN_CONTROLS),
        "Convexity": cv2.getTrackbarPos("Convexity", WIN_CONTROLS),
        "Inertia": cv2.getTrackbarPos("Inertia", WIN_CONTROLS)
    }
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"💾 Settings saved to {SETTINGS_FILE}")

# --- INITIALIZE UI ---
defaults = load_settings()
cv2.namedWindow(WIN_CONTROLS)

cv2.createTrackbar("MinArea", WIN_CONTROLS, defaults["MinArea"], 5000, nothing)
cv2.createTrackbar("MaxArea", WIN_CONTROLS, defaults["MaxArea"], 10000, nothing)
cv2.createTrackbar("Circularity", WIN_CONTROLS, defaults["Circularity"], 100, nothing) 
cv2.createTrackbar("Convexity", WIN_CONTROLS, defaults["Convexity"], 100, nothing)   
cv2.createTrackbar("Inertia", WIN_CONTROLS, defaults["Inertia"], 100, nothing)     

def get_detector():
    """Builds a detector with strict safety checks to prevent crashes"""
    params = cv2.SimpleBlobDetector_Params()
    
    # Get values
    ma = cv2.getTrackbarPos("MinArea", WIN_CONTROLS)
    MA = cv2.getTrackbarPos("MaxArea", WIN_CONTROLS)
    
    # 1. Area Safety: 0 < minArea < maxArea
    params.filterByArea = True
    params.minArea = float(max(1, ma))
    params.maxArea = float(max(ma + 1, MA))
    
    # 2. Ratio Safety: 0 < value <= 1.0 (OpenCV crashes at 0.0)
    params.filterByCircularity = True
    params.minCircularity = max(0.01, cv2.getTrackbarPos("Circularity", WIN_CONTROLS) / 100.0)
    
    params.filterByConvexity = True
    params.minConvexity = max(0.01, cv2.getTrackbarPos("Convexity", WIN_CONTROLS) / 100.0)
    
    params.filterByInertia = True
    params.minInertiaRatio = max(0.01, cv2.getTrackbarPos("Inertia", WIN_CONTROLS) / 100.0)

    # Thresholding
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = 10

    return cv2.SimpleBlobDetector_create(params)

def main():
    cap_l = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
    cap_r = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)

    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    print("🚀 Tuner Online.")
    print("Commands: [s] Save Settings | [q] Quit")
    
    try:
        while True:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r: break

            detector = get_detector()
            key_l = detector.detect(frame_l)
            key_r = detector.detect(frame_r)

            res_l = cv2.drawKeypoints(frame_l, key_l, np.array([]), (0, 0, 255), 4)
            res_r = cv2.drawKeypoints(frame_r, key_r, np.array([]), (0, 0, 255), 4)

            cv2.imshow(WIN_FEED, np.hstack((res_l, res_r)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_settings()
                
    finally:
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()