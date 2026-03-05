import cv2
import json
import time
import sys
from pathlib import Path

# Local imports
from cv_helpers import WeedCV

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
HW_CFG_PATH = BASE_DIR / "hardware_config.json"
CAM_CFG_PATH = BASE_DIR / "camera_config.json"
WEIGHTS_DIR = BASE_DIR / "weights"
YOLO_PT = str(WEIGHTS_DIR / "yolo_w_kale.pt")
SNIPER_PT = str(WEIGHTS_DIR / "sniper.pt")   

IS_WINDOWS = sys.platform.startswith('win')
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

def update_camera(cap, props):
    """Identical to Ablator for perfect settings match."""
    if not props: return
    # Manual Flags
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    
    # Core Parameters
    cap.set(cv2.CAP_PROP_BRIGHTNESS, props.get('brightness', 0))
    cap.set(cv2.CAP_PROP_CONTRAST, props.get('contrast', 0))
    cap.set(cv2.CAP_PROP_EXPOSURE, props.get('exposure', -6))
    cap.set(cv2.CAP_PROP_GAIN, props.get('gain', 0))
    cap.set(cv2.CAP_PROP_SATURATION, props.get('saturation', 64))
    cap.set(cv2.CAP_PROP_SHARPNESS, props.get('sharpness', 100))
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, props.get('white_balance', 4000))

def main():
    if not HW_CFG_PATH.exists(): return
    
    ai_L, ai_R = None, None
    if Path(YOLO_PT).exists() and Path(SNIPER_PT).exists():
        ai_L = WeedCV(YOLO_PT, SNIPER_PT)
        ai_R = WeedCV(YOLO_PT, SNIPER_PT)

    with open(HW_CFG_PATH, 'r') as f:
        hw = json.load(f)

    cap_l = cv2.VideoCapture(hw['cameras']['left']['index'], BACKEND)
    cap_r = cv2.VideoCapture(hw['cameras']['right']['index'], BACKEND)

    # LOCK RESOLUTION TO 640x480 FOR STABILITY & SPEED
    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 

    if CAM_CFG_PATH.exists():
        with open(CAM_CFG_PATH, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "left":  {"brightness": 15, "contrast": 30, "exposure": -6, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100},
            "right": {"brightness": 15, "contrast": 30, "exposure": -6, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100}
        }

    win_l, win_r = "LEFT_TUNER", "RIGHT_TUNER"
    cv2.namedWindow(win_l, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_r, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_l, 640, 480)
    cv2.resizeWindow(win_r, 640, 480)

    def setup_sliders(win, side):
        c = config[side]
        cv2.createTrackbar("Brightness", win, c.get("brightness", 15), 64, lambda x: None)
        cv2.createTrackbar("Contrast",   win, c.get("contrast", 30), 100, lambda x: None)
        cv2.createTrackbar("Exposure",   win, abs(c.get("exposure", -6)), 13, lambda x: None)
        cv2.createTrackbar("Gain",       win, c.get("gain", 0), 255, lambda x: None)
        cv2.createTrackbar("Saturation", win, c.get("saturation", 64), 255, lambda x: None)
        cv2.createTrackbar("WB_Temp",    win, int((c.get("white_balance", 4500)-2800)/37), 100, lambda x: None)

    setup_sliders(win_l, "left")
    setup_sliders(win_r, "right")

    last_applied = {"left": None, "right": None}

    while True:
        for side, win in [("left", win_l), ("right", win_r)]:
            config[side]["brightness"] = cv2.getTrackbarPos("Brightness", win)
            config[side]["contrast"]   = cv2.getTrackbarPos("Contrast", win)
            config[side]["exposure"]   = -cv2.getTrackbarPos("Exposure", win)
            config[side]["gain"]       = cv2.getTrackbarPos("Gain", win)
            config[side]["saturation"] = cv2.getTrackbarPos("Saturation", win)
            config[side]["white_balance"] = cv2.getTrackbarPos("WB_Temp", win) * 37 + 2800

        for side, cap in [("left", cap_l), ("right", cap_r)]:
            if config[side] != last_applied[side]:
                update_camera(cap, config[side])
                last_applied[side] = config[side].copy()

        ret_l, f_l = cap_l.read()
        ret_r, f_r = cap_r.read()
        if ret_l: f_l = cv2.rotate(f_l, cv2.ROTATE_180)  # Rotate if your cameras are mounted upside down
        if ret_r: f_r = cv2.rotate(f_r, cv2.ROTATE_180)

        if ret_l and ret_r:
            cv2.imshow(win_l, f_l)
            cv2.imshow(win_r, f_r)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            with open(CAM_CFG_PATH, 'w') as f:
                json.dump(config, f, indent=4)
            print("✅ Tuner Saved.")
        elif key == ord('q'): break

    cap_l.release(); cap_r.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()