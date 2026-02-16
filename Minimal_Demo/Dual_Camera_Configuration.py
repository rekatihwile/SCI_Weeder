import cv2
import json
import time
import sys
from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
HW_CFG_PATH = BASE_DIR / "hardware_config.json"
CAM_CFG_PATH = BASE_DIR / "camera_config.json"

IS_WINDOWS = sys.platform.startswith('win')
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

def update_camera(cap, props):
    """Pushes all hardware parameters to the firmware."""
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    cap.set(cv2.CAP_PROP_BRIGHTNESS, props['brightness'])
    cap.set(cv2.CAP_PROP_CONTRAST, props['contrast'])
    cap.set(cv2.CAP_PROP_EXPOSURE, props['exposure'])
    cap.set(cv2.CAP_PROP_GAIN, props['gain'])
    cap.set(cv2.CAP_PROP_SATURATION, props['saturation'])
    cap.set(cv2.CAP_PROP_SHARPNESS, props['sharpness'])
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, props['white_balance'])

def main():
    if not HW_CFG_PATH.exists():
        print("❌ Error: hardware_config.json not found.")
        return

    with open(HW_CFG_PATH, 'r') as f:
        hw = json.load(f)

    cap_l = cv2.VideoCapture(hw['cameras']['left']['index'], BACKEND)
    cap_r = cv2.VideoCapture(hw['cameras']['right']['index'], BACKEND)

    # 1. LOAD PERSISTENT SETTINGS
    if CAM_CFG_PATH.exists():
        with open(CAM_CFG_PATH, 'r') as f:
            config = json.load(f)
        print("📂 Loaded existing configuration from JSON.")
    else:
        # Default fallback
        def_exp = -6 if IS_WINDOWS else 350
        config = {
            "left":  {"brightness": 15, "contrast": 30, "exposure": def_exp, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100},
            "right": {"brightness": 15, "contrast": 30, "exposure": def_exp, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100}
        }

    win_l, win_r = "LEFT_TUNER", "RIGHT_TUNER"
    cv2.namedWindow(win_l); cv2.namedWindow(win_r)

    def setup_sliders(win, side):
        c = config[side]
        # Initialize sliders using the loaded config values
        cv2.createTrackbar("Brightness", win, c["brightness"], 32, lambda x: None)
        cv2.createTrackbar("Contrast",   win, max(0, c["contrast"]-10), 40, lambda x: None)
        
        # Exposure slider initialization (reverse the math for Windows)
        exp_init = int(abs(c["exposure"]) / 13 * 100) if IS_WINDOWS else c["exposure"]
        cv2.createTrackbar("Fine_Expos", win, exp_init, 100 if IS_WINDOWS else 1000, lambda x: None)
        
        cv2.createTrackbar("Gain",       win, c["gain"], 255, lambda x: None)
        cv2.createTrackbar("Saturation", win, c["saturation"], 255, lambda x: None)
        cv2.createTrackbar("WB_Temp",    win, int((c["white_balance"]-2800)/37), 100, lambda x: None)
        cv2.createTrackbar("Sharpness",  win, c["sharpness"], 255, lambda x: None)

    setup_sliders(win_l, "left")
    setup_sliders(win_r, "right")

    last_applied = {"left": None, "right": None}

    while True:
        # 2. READ UI VALUES
        for side, win in [("left", win_l), ("right", win_r)]:
            config[side]["brightness"] = cv2.getTrackbarPos("Brightness", win)
            config[side]["contrast"]   = cv2.getTrackbarPos("Contrast", win) + 10
            raw_ex = cv2.getTrackbarPos("Fine_Expos", win)
            config[side]["exposure"]   = -int((raw_ex/100)*13) if IS_WINDOWS else raw_ex
            config[side]["gain"]       = cv2.getTrackbarPos("Gain", win)
            config[side]["saturation"] = cv2.getTrackbarPos("Saturation", win)
            config[side]["white_balance"] = cv2.getTrackbarPos("WB_Temp", win) * 37 + 2800
            config[side]["sharpness"]  = cv2.getTrackbarPos("Sharpness", win)

        # 3. APPLY TO HARDWARE
        for side, cap in [("left", cap_l), ("right", cap_r)]:
            if config[side] != last_applied[side]:
                update_camera(cap, config[side])
                last_applied[side] = config[side].copy()
                time.sleep(0.02)

        # 4. CAPTURE AND HUD DISPLAY
        ret_l, f_l = cap_l.read()
        ret_r, f_r = cap_r.read()

        if ret_l and ret_r:
            for side, frame in [("left", f_l), ("right", f_r)]:
                c = config[side]
                # Detailed On-Screen HUD
                hud = [
                    f"B:{c['brightness']} C:{c['contrast']} E:{c['exposure']}",
                    f"G:{c['gain']} S:{c['saturation']} WB:{c['white_balance']} SH:{c['sharpness']}"
                ]
                for i, text in enumerate(hud):
                    cv2.putText(frame, text, (15, 35 + (i * 35)), 1, 1.4, (0, 255, 0), 2)
            
            cv2.imshow(win_l, f_l)
            cv2.imshow(win_r, f_r)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            with open(CAM_CFG_PATH, 'w') as f:
                json.dump(config, f, indent=4)
            print("✅ PERSISTED: Configuration saved to JSON.")
        elif key == ord('q') or key == ord('Q'):
            break

    cap_l.release(); cap_r.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()