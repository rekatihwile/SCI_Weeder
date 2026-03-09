import cv2
import numpy as np
import json
import os

# === CONFIG ===
CAM_ID_LEFT = 0
CAM_ID_RIGHT = 2
SETTINGS_FILE = "cv_settings.json"

# Define the "Dead Spot" zone for the Left/Right cameras
# Format: (x_min, y_min, x_max, y_max)
# You will need to tune these based on your specific camera's dead spot!
LENS_DEAD_ZONE_L = (0, 440, 40, 480) # Example: bottom-left corner
LENS_DEAD_ZONE_R = (0, 440, 40, 480) 

def nothing(x):
    pass

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "Filter_d": 9, "Filter_sigma": 75, "BlockSize": 21, 
        "C_Value": 4, "MinArea": 50, "MinCirc": 65, "UseMask": 1
    }

def save_settings(window_name):
    settings = {
        "Filter_d": cv2.getTrackbarPos("Filter_d", window_name),
        "Filter_sigma": cv2.getTrackbarPos("Filter_sigma", window_name),
        "BlockSize": cv2.getTrackbarPos("BlockSize", window_name),
        "C_Value": cv2.getTrackbarPos("C_Value", window_name),
        "MinArea": cv2.getTrackbarPos("MinArea", window_name),
        "MinCirc": cv2.getTrackbarPos("MinCirc", window_name),
        "UseMask": cv2.getTrackbarPos("UseMask", window_name)
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)
    print("--- Settings Saved ---")

def setup_ui(window_name, defaults):
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Filter_d", window_name, defaults["Filter_d"], 20, nothing)
    cv2.createTrackbar("Filter_sigma", window_name, defaults["Filter_sigma"], 150, nothing)
    cv2.createTrackbar("BlockSize", window_name, defaults["BlockSize"], 101, nothing)
    cv2.createTrackbar("C_Value", window_name, defaults["C_Value"], 30, nothing)
    cv2.createTrackbar("MinArea", window_name, defaults["MinArea"], 500, nothing)
    cv2.createTrackbar("MinCirc", window_name, defaults["MinCirc"], 100, nothing)
    cv2.createTrackbar("UseMask", window_name, defaults["UseMask"], 1, nothing)

def process_frame(frame, params, dead_zone=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- MASKING STEP ---
    # If a dead zone is defined and enabled, we "white out" that spot 
    # so the thresholding ignores it.
    if params['use_mask'] and dead_zone:
        x1, y1, x2, y2 = dead_zone
        # We fill it with white (255) so the Inverse Threshold sees "Background"
        cv2.rectangle(gray, (x1, y1), (x2, y2), (255), -1)

    filtered = cv2.bilateralFilter(gray, params['f_d'], params['f_sigma'], params['f_sigma'])
    
    bs = params['bs'] if params['bs'] % 2 != 0 else params['bs'] + 1
    if bs < 3: bs = 3
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, bs, params['c'])
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > params['min_a']:
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circ = 4 * np.pi * (area / (peri * peri))
            if circ > params['min_c']:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    dots.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), cnt))
    return dots, thresh

def main():
    TUNER_WIN = "Stereo CV Tuner"
    cap_l, cap_r = cv2.VideoCapture(CAM_ID_LEFT), cv2.VideoCapture(CAM_ID_RIGHT)
    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    setup_ui(TUNER_WIN, load_settings())

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r: continue

        params = {
            'f_d': cv2.getTrackbarPos("Filter_d", TUNER_WIN),
            'f_sigma': cv2.getTrackbarPos("Filter_sigma", TUNER_WIN),
            'bs': cv2.getTrackbarPos("BlockSize", TUNER_WIN),
            'c': cv2.getTrackbarPos("C_Value", TUNER_WIN),
            'min_a': cv2.getTrackbarPos("MinArea", TUNER_WIN),
            'min_c': cv2.getTrackbarPos("MinCirc", TUNER_WIN) / 100.0,
            'use_mask': cv2.getTrackbarPos("UseMask", TUNER_WIN)
        }

        # Pass the dead zones to the processor
        dots_l, mask_l = process_frame(frame_l, params, LENS_DEAD_ZONE_L)
        dots_r, mask_r = process_frame(frame_r, params, LENS_DEAD_ZONE_R)

        for (x, y, cnt) in dots_l:
            cv2.drawContours(frame_l, [cnt], -1, (0, 255, 0), 2)
        for (x, y, cnt) in dots_r:
            cv2.drawContours(frame_r, [cnt], -1, (0, 255, 0), 2)

        cv2.imshow(TUNER_WIN, np.hstack((frame_l, frame_r)))
        cv2.imshow("Mask", np.hstack((mask_l, mask_r)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'): save_settings(TUNER_WIN)

    cap_l.release(); cap_r.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()