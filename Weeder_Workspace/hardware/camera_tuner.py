from pathlib import Path
import json
import cv2
import sys
import os
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, LEFT_CAMERA_INDEX, RIGHT_CAMERA_INDEX

from hardware.cameras import StereoCameras

try:
    from vision.detectors.ai_detector import _WeedCVCore
    AI_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Could not import AI models: {e}")
    AI_AVAILABLE = False

PREVIEW_WIN = "SCI_Weeder - Stereo Preview"
LEFT_CTRL_WIN = "Left Camera Controls"
RIGHT_CTRL_WIN = "Right Camera Controls"

CONFIG_PATH = BASE_DIR / "params" / "camera_config.json"
PARAMS_DIR = BASE_DIR / "params"
WEIGHTS_DIR = BASE_DIR / "weights"


def pick_existing_path(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]

# Updated to look for your new best_pigweed models first!
YOLO_PT = pick_existing_path(
    PARAMS_DIR / "best_pigweed_145.pt",
    PARAMS_DIR / "yolo_weed.pt",
)

# Updated to look for the v3 targeting weights first!
SNIPER_PT = pick_existing_path(
    PARAMS_DIR / "new_best_targeting_v3.pth",
    PARAMS_DIR / "sniper.pt",
)

if os.name == "nt":
    # Windows Ranges
    TRACKBAR_MAP = {
        "exposure": {"min": -13, "max": 0, "default": -6, "short": "Exposure"},
        "gain": {"min": 0, "max": 100, "default": 0, "short": "Gain"},
        "brightness": {"min": -64, "max": 64, "default": 0, "short": "Brightness"},
        "contrast": {"min": 0, "max": 100, "default": 50, "short": "Contrast"},
        "saturation": {"min": 0, "max": 100, "default": 50, "short": "Saturation"},
        "white_balance": {"min": 2800, "max": 6500, "default": 4600, "short": "W_Balance"},
    }
else:
    # Jetson / Linux V4L2 Ranges (Matched EXACTLY to your hardware!)
    TRACKBAR_MAP = {
        # Max is technically 10000, but capped at 1000 here so the slider isn't too sensitive
        "exposure": {"min": 0, "max": 1000, "default": 166, "short": "Exposure"},
        "gain": {"min": 1, "max": 128, "default": 64, "short": "Gain"},
        "brightness": {"min": -64, "max": 64, "default": 0, "short": "Brightness"},
        "contrast": {"min": 0, "max": 100, "default": 40, "short": "Contrast"},
        "saturation": {"min": 0, "max": 100, "default": 64, "short": "Saturation"},
        "white_balance": {"min": 2800, "max": 6500, "default": 4600, "short": "W_Balance"},
    }


def nothing(_=None):
    pass


def load_existing():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
    else:
        cfg = {"left": {}, "right": {}}

    for side in ["left", "right"]:
        cfg.setdefault(side, {})
        cfg[side].setdefault("auto_exposure", 0)
        cfg[side].setdefault("auto_wb", 0)
    return cfg


def save_settings(settings):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(settings, f, indent=4)
    print(f"[INFO] Settings saved to {CONFIG_PATH}")


def create_multi_window_ui(cfg):
    cv2.namedWindow(LEFT_CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LEFT_CTRL_WIN, 400, 450)
    cv2.moveWindow(LEFT_CTRL_WIN, 50, 50)

    cv2.namedWindow(RIGHT_CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(RIGHT_CTRL_WIN, 400, 450)
    cv2.moveWindow(RIGHT_CTRL_WIN, 850, 50)

    cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_NORMAL)
    cv2.moveWindow(PREVIEW_WIN, 50, 550)

    defaults_left = cfg.get("left", {})
    defaults_right = cfg.get("right", {})

    for key, win_name in [("left", LEFT_CTRL_WIN), ("right", RIGHT_CTRL_WIN)]:
        side_cfg = defaults_left if key == "left" else defaults_right
        prefix = "L_" if key == "left" else "R_"
        
        for param, props in TRACKBAR_MAP.items():
            tb_name = prefix + props["short"]
            tb_max = props["max"] - props["min"]
            saved_val = max(props["min"], min(props["max"], side_cfg.get(param, props["default"])))
            tb_val = int(saved_val - props["min"])
            cv2.createTrackbar(tb_name, win_name, 0, tb_max, nothing)
            cv2.setTrackbarPos(tb_name, win_name, tb_val)

def read_trackbar_settings(existing_cfg):
    manual_val = 1 if os.name != "nt" else 0

    l_exp = int(existing_cfg.get("left", {}).get("auto_exposure", manual_val))
    l_wb  = int(existing_cfg.get("left", {}).get("auto_wb", manual_val))
    r_exp = int(existing_cfg.get("right", {}).get("auto_exposure", manual_val))
    r_wb  = int(existing_cfg.get("right", {}).get("auto_wb", manual_val))

    if os.name != "nt":
        l_exp = 1 if l_exp == 0 else l_exp
        l_wb  = 1 if l_wb == 0 else l_wb
        r_exp = 1 if r_exp == 0 else r_exp
        r_wb  = 1 if r_wb == 0 else r_wb

    settings = {
        "left": {"auto_exposure": l_exp, "auto_wb": l_wb},
        "right": {"auto_exposure": r_exp, "auto_wb": r_wb},
    }

    for key, win_name in [("left", LEFT_CTRL_WIN), ("right", RIGHT_CTRL_WIN)]:
        prefix = "L_" if key == "left" else "R_"
        for param, props in TRACKBAR_MAP.items():
            tb_name = prefix + props["short"]
            tb_val = cv2.getTrackbarPos(tb_name, win_name)
            if tb_val < 0:
                tb_val = props["default"] - props["min"]
            settings[key][param] = tb_val + props["min"]
            
    return settings

def update_live_camera(cap, props, dev_path=None):
    if cap is None or not cap.isOpened():
        return

    auto_exposure = float(props.get("auto_exposure", 1))
    auto_wb = float(props.get("auto_wb", 1))

    if os.name == "nt":
        # Windows: Use standard OpenCV
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
        cap.set(cv2.CAP_PROP_AUTO_WB, auto_wb)
        if "exposure" in props: cap.set(cv2.CAP_PROP_EXPOSURE, float(props["exposure"]))
        if "gain" in props: cap.set(cv2.CAP_PROP_GAIN, float(props["gain"]))
        if "brightness" in props: cap.set(cv2.CAP_PROP_BRIGHTNESS, float(props["brightness"]))
        if "contrast" in props: cap.set(cv2.CAP_PROP_CONTRAST, float(props["contrast"]))
        if "saturation" in props: cap.set(cv2.CAP_PROP_SATURATION, float(props["saturation"]))
        if "white_balance" in props: cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(props["white_balance"]))
    else:
        # Jetson / Linux: Completely bypass OpenCV to stop "cross-talk"
        if dev_path is not None:
            # Print exactly which node is being targeted to the terminal
            print(f"[V4L2 Target] Sending commands strictly to -> {dev_path}")
            
            v4l2_auto_exp = 1 if auto_exposure == 1 else 3
            v4l2_auto_wb = 0 if auto_wb == 1 else 1 

            os.system(f"v4l2-ctl -d {dev_path} -c exposure_auto={v4l2_auto_exp} > /dev/null 2>&1")
            os.system(f"v4l2-ctl -d {dev_path} -c white_balance_temperature_auto={v4l2_auto_wb} > /dev/null 2>&1")

            if "exposure" in props: os.system(f"v4l2-ctl -d {dev_path} -c exposure_absolute={int(props['exposure'])} > /dev/null 2>&1")
            if "gain" in props: os.system(f"v4l2-ctl -d {dev_path} -c gain={int(props['gain'])} > /dev/null 2>&1")
            if "brightness" in props: os.system(f"v4l2-ctl -d {dev_path} -c brightness={int(props['brightness'])} > /dev/null 2>&1")
            if "contrast" in props: os.system(f"v4l2-ctl -d {dev_path} -c contrast={int(props['contrast'])} > /dev/null 2>&1")
            if "saturation" in props: os.system(f"v4l2-ctl -d {dev_path} -c saturation={int(props['saturation'])} > /dev/null 2>&1")
            if "white_balance" in props: os.system(f"v4l2-ctl -d {dev_path} -c white_balance_temperature={int(props['white_balance'])} > /dev/null 2>&1")

def infer_with_markers(detector, frame):
    if detector is None:
        return [], []

    points = detector.detect_points(frame)
    boxes = []
    for box in detector.filtered_boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
        boxes.append(xyxy)
    return points, boxes


def process_frame_visuals(frame, points, boxes, label, settings):
    disp = frame.copy()

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(disp, f"B{i}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for i, (px, py) in enumerate(points, start=1):
        cv2.drawMarker(disp, (int(px), int(py)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.circle(disp, (int(px), int(py)), 5, (0, 255, 255), -1)
        cv2.putText(disp, str(i), (int(px) + 8, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # FIXED: OS-aware mode check
    auto_val = 3 if os.name != "nt" else 1
    is_auto = (settings.get("auto_exposure", -1) == auto_val) or (settings.get("auto_wb", -1) == auto_val)
    mode_text = "AUTO" if is_auto else "MANUAL"
    
    line1 = f"{label} | det={len(points)} | mode={mode_text}"
    line2 = f"B:{settings.get('brightness', 0)} C:{settings.get('contrast', 0)} E:{settings.get('exposure', 0)} G:{settings.get('gain', 0)}"
    line3 = f"S:{settings.get('saturation', 0)} WB:{settings.get('white_balance', 0)}"

    cv2.putText(disp, line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(disp, line2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(disp, line3, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return disp


def add_dashboard_bar(preview):
    _, w = preview.shape[:2]
    bar = np.zeros((90, w, 3), dtype=np.uint8)

    line1 = f"YOLO: {YOLO_PT.name}"
    line2 = f"QPOINT: {SNIPER_PT.name if SNIPER_PT.exists() else 'missing'}"
    line3 = "q = quit | s = save | a = auto | m = manual"

    cv2.putText(bar, line1, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(bar, line2, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(bar, line3, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return np.vstack((preview, bar))


def set_mode(cfg, auto_on):
    # Linux (Jetson) uses 3 for Auto, 1 for Manual. Windows uses 1/0.
    auto_val = 3 if (os.name != "nt") else 1
    manual_val = 1 if (os.name != "nt") else 0
    value = auto_val if auto_on else manual_val

    for side in ["left", "right"]:
        cfg.setdefault(side, {})
        cfg[side]["auto_exposure"] = value
        cfg[side]["auto_wb"] = value


def main():
    print("[INFO] Starting Stereo Camera Tuner...")
    print(f"[INFO] YOLO weights  : {YOLO_PT}")
    print(f"[INFO] QPOINT weights: {SNIPER_PT}")

    if os.name == "nt":
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

    cfg = load_existing()
    create_multi_window_ui(cfg)

    cams = StereoCameras()

    ai_L, ai_R = None, None
    if AI_AVAILABLE:
        if not YOLO_PT.exists():
            print(f"[WARNING] YOLO weights not found: {YOLO_PT}")
        else:
            print("[INFO] Loading WeedCV models...")
            ai_L = _WeedCVCore(yolo_path=YOLO_PT, qpoint_path=SNIPER_PT)
            ai_R = _WeedCVCore(yolo_path=YOLO_PT, qpoint_path=SNIPER_PT)
    else:
        print("[WARNING] AI detector import failed. Running camera-only preview.")

    try:
        cams.open()

        # Step A: Dynamically map dev paths based on your actual config indices
        dev_paths = {
            "left": f"/dev/video{LEFT_CAMERA_INDEX}",
            "right": f"/dev/video{RIGHT_CAMERA_INDEX}"
        }
        
        # Keep the hardware config override fallback just in case
        hw_cfg = BASE_DIR / "params" / "hardware_config.json"
        if hw_cfg.exists():
            with open(hw_cfg, "r") as f:
                hw = json.load(f)
                if "cameras" in hw:
                    if hw["cameras"].get("left") and "device" in hw["cameras"]["left"]:
                        dev_paths["left"] = hw["cameras"]["left"]["device"]
                    if hw["cameras"].get("right") and "device" in hw["cameras"]["right"]:
                        dev_paths["right"] = hw["cameras"]["right"]["device"]


        last_settings = {"left": {}, "right": {}}

        while True:
            live_settings = read_trackbar_settings(cfg)

            # Step B: Pass the dev_paths to the hardware override
            if live_settings["left"] != last_settings["left"]:
                if getattr(cams, "left", None) is not None:
                    update_live_camera(cams.left, live_settings["left"], dev_paths["left"])
                last_settings["left"] = live_settings["left"].copy()

            if live_settings["right"] != last_settings["right"]:
                if getattr(cams, "right", None) is not None:
                    update_live_camera(cams.right, live_settings["right"], dev_paths["right"])
                last_settings["right"] = live_settings["right"].copy()

            try:
                frame_l, frame_r = cams.read_pair()
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                break

            points_l, boxes_l = [], []
            points_r, boxes_r = [], []

            if ai_L is not None and ai_R is not None:
                try:
                    points_l, boxes_l = infer_with_markers(ai_L, frame_l)
                    points_r, boxes_r = infer_with_markers(ai_R, frame_r)
                except Exception as e:
                    print(f"[WARNING] Inference frame error: {e}")

            vis_l = process_frame_visuals(frame_l, points_l, boxes_l, "LEFT", live_settings["left"])
            vis_r = process_frame_visuals(frame_r, points_r, boxes_r, "RIGHT", live_settings["right"])

            preview = np.hstack((vis_l, vis_r))
            final_ui = add_dashboard_bar(preview)

            cv2.imshow(PREVIEW_WIN, final_ui)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                save_settings(live_settings)
                cfg = load_existing()
                print("[INFO] Saved current settings.")
            if key == ord("a"):
                set_mode(cfg, True)
                print("[INFO] AUTO mode enabled for both cameras.")
            if key == ord("m"):
                set_mode(cfg, False)
                print("[INFO] MANUAL mode enabled for both cameras.")

    finally:
        cams.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()