import cv2
import numpy as np
import time
import sys
import json

# === CONFIG ===
CAM_ID_LEFT = 0
CAM_ID_RIGHT = 2
SYNC_ITERATIONS = 5 
CONFIG_FILE = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/DOT_DEMO_2_1/camera_config.json"

def force_sync_cameras(caps, settings):
    """Pushes trackbar states to the hardware firmware."""
    for _ in range(SYNC_ITERATIONS):
        for cap in caps:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)       
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            cap.set(cv2.CAP_PROP_EXPOSURE, settings['exp'])
            cap.set(cv2.CAP_PROP_GAIN, settings['gain'])
            cap.set(cv2.CAP_PROP_BRIGHTNESS, settings['brt'])
            cap.set(cv2.CAP_PROP_CONTRAST, settings['con'])
            cap.set(cv2.CAP_PROP_SATURATION, settings['sat'])
            cap.set(cv2.CAP_PROP_HUE, settings['hue'])
            cap.set(cv2.CAP_PROP_GAMMA, settings['gamma'])
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, settings['wb'])
            cap.set(cv2.CAP_PROP_SHARPNESS, settings['sharp'])

def main():
    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)
    
    if not cap_l.isOpened() or not cap_r.isOpened():
        print("❌ Error: One or both cameras failed to open."); sys.exit()

    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    time.sleep(1.0)
    win_name = "Color Match Tool"
    cv2.namedWindow(win_name)
    
    # Trackbars
    cv2.createTrackbar("Exposure", win_name, 150, 1000, lambda x: None)
    cv2.createTrackbar("Gain", win_name, 0, 255, lambda x: None)
    cv2.createTrackbar("Brightness", win_name, 128, 255, lambda x: None)
    cv2.createTrackbar("Contrast", win_name, 32, 255, lambda x: None) # ADDED
    cv2.createTrackbar("Hue", win_name, 0, 180, lambda x: None)
    cv2.createTrackbar("Saturation", win_name, 64, 255, lambda x: None)
    cv2.createTrackbar("Gamma", win_name, 100, 300, lambda x: None)
    cv2.createTrackbar("WB_Temp", win_name, 4500, 6500, lambda x: None)
    cv2.createTrackbar("Sharpness", win_name, 100, 255, lambda x: None)

    print("\n--- 🕹️ CONTROLS ---")
    print(" 's' -> Save settings to JSON and Exit")
    print(" 'q' -> Quit without saving\n")

    while True:
        current_settings = {
            'exp': cv2.getTrackbarPos("Exposure", win_name),
            'gain': cv2.getTrackbarPos("Gain", win_name),
            'brt': cv2.getTrackbarPos("Brightness", win_name),
            'con': cv2.getTrackbarPos("Contrast", win_name), # UPDATED
            'hue': cv2.getTrackbarPos("Hue", win_name),
            'gamma': cv2.getTrackbarPos("Gamma", win_name),
            'sat': cv2.getTrackbarPos("Saturation", win_name),
            'wb': cv2.getTrackbarPos("WB_Temp", win_name),
            'sharp': cv2.getTrackbarPos("Sharpness", win_name)
        }

        force_sync_cameras([cap_l, cap_r], current_settings)

        ret_l, fl = cap_l.read()
        ret_r, fr = cap_r.read()

        if ret_l and ret_r:
            combined = np.hstack((fl, fr))
            cv2.imshow(win_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            with open(CONFIG_FILE, 'w') as f:
                json.dump(current_settings, f, indent=4)
            print(f"✅ Settings saved to {CONFIG_FILE}")
            break
        elif key == ord('q'):
            break

    cap_l.release(); cap_r.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()