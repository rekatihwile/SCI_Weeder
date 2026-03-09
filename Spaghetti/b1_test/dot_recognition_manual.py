import cv2
import numpy as np
import json
import time
import os

# === CONFIG ===
CAM_ID_RIGHT = 2
CAM_ID_LEFT = 0
SETTINGS_FILE = "camera_settings.json"

def nothing(x):
    pass

def load_settings():
    """Load settings from JSON or return defaults."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
    return None

def save_settings(window_name):
    """Save current trackbar positions to JSON."""
    settings = {
        "Exposure": cv2.getTrackbarPos("Exposure", window_name),
        "Gain": cv2.getTrackbarPos("Gain", window_name),
        "Brightness": cv2.getTrackbarPos("Brightness", window_name),
        "Contrast": cv2.getTrackbarPos("Contrast", window_name),
        "Saturation": cv2.getTrackbarPos("Saturation", window_name),
        "WB_Temp": cv2.getTrackbarPos("WB_Temp", window_name),
        "Threshold": cv2.getTrackbarPos("Threshold", window_name),
        "Min_Area": cv2.getTrackbarPos("Min_Area", window_name),
        "Max_Area": cv2.getTrackbarPos("Max_Area", window_name)
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)
    print(f"--- Settings Saved to {SETTINGS_FILE} ---")

def setup_master_ui(window_name):
    """Create UI with values from the saved JSON or defaults."""
    cv2.namedWindow(window_name)
    saved = load_settings()
    
    # helper to get saved value or default
    def get_v(name, default):
        return saved.get(name, default) if saved and name in saved else default

    # Trackbars (Name, Window, Default, Max, Callback)
    cv2.createTrackbar("Exposure", window_name, get_v("Exposure", 150), 1000, nothing)
    cv2.createTrackbar("Gain", window_name, get_v("Gain", 0), 255, nothing)
    cv2.createTrackbar("Brightness", window_name, get_v("Brightness", 128), 255, nothing)
    cv2.createTrackbar("Contrast", window_name, get_v("Contrast", 32), 255, nothing)
    cv2.createTrackbar("Saturation", window_name, get_v("Saturation", 64), 255, nothing)
    cv2.createTrackbar("WB_Temp", window_name, get_v("WB_Temp", 4500), 6500, nothing)
    
    cv2.createTrackbar("Threshold", window_name, get_v("Threshold", 100), 255, nothing)
    cv2.createTrackbar("Min_Area", window_name, get_v("Min_Area", 50), 5000, nothing)
    cv2.createTrackbar("Max_Area", window_name, get_v("Max_Area", 10000), 50000, nothing)

def apply_stereo_settings(caps, window_name):
    """Apply current trackbar values to both camera hardwares."""
    exp = cv2.getTrackbarPos("Exposure", window_name)
    gain = cv2.getTrackbarPos("Gain", window_name)
    brt = cv2.getTrackbarPos("Brightness", window_name)
    con = cv2.getTrackbarPos("Contrast", window_name)
    sat = cv2.getTrackbarPos("Saturation", window_name)
    wb_temp = cv2.getTrackbarPos("WB_Temp", window_name)

    for cap in caps:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # Manual
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        cap.set(cv2.CAP_PROP_GAIN, gain)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brt)
        cap.set(cv2.CAP_PROP_CONTRAST, con)
        cap.set(cv2.CAP_PROP_SATURATION, sat)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temp)

def detect_dots(frame, thresh_val, min_a, max_a):
    """Process frame and return dot coordinates + binary mask."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # Denoise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_a < area < max_a:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dots.append((cx, cy, area))
    return dots, thresh

def main():
    WIN_NAME = "Master Stereo Dashboard"
    
    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)
    
    # Optimization: Set MJPG for bandwidth and baseline res
    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    setup_master_ui(WIN_NAME)
    print("Commands: [s] Save Settings | [q] Quit")

    while True:
        # Update hardware to match sliders
        apply_stereo_settings([cap_l, cap_r], WIN_NAME)

        # Get detection thresholds
        t_val = cv2.getTrackbarPos("Threshold", WIN_NAME)
        min_a = cv2.getTrackbarPos("Min_Area", WIN_NAME)
        max_a = cv2.getTrackbarPos("Max_Area", WIN_NAME)

        ret_l, fl = cap_l.read()
        ret_r, fr = cap_r.read()

        if not ret_l or not ret_r:
            continue

        # Run Dot Recognition
        dots_l, mask_l = detect_dots(fl, t_val, min_a, max_a)
        dots_r, mask_r = detect_dots(fr, t_val, min_a, max_a)

        # Draw Feedback Circles
        for (x, y, a) in dots_l:
            cv2.circle(fl, (x, y), 12, (0, 255, 0), 2)
        for (x, y, a) in dots_r:
            cv2.circle(fr, (x, y), 12, (0, 255, 255), 2)

        # Output Windows
        combined_rgb = np.hstack((fl, fr))
        combined_mask = np.hstack((mask_l, mask_r))
        
        cv2.putText(combined_rgb, f"L: {len(dots_l)} | R: {len(dots_r)}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(WIN_NAME, combined_rgb)
        cv2.imshow("Binary Threshold Mask", combined_mask)

        # Key Listeners
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_settings(WIN_NAME)

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()