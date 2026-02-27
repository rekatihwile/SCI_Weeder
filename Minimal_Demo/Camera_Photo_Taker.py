# capture_stereo_pairs.py
import cv2
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HW_CFG_PATH = BASE_DIR / "hardware_config.json"
CAM_CFG_PATH = BASE_DIR / "camera_config.json"

# Match your existing backend choice
IS_WINDOWS = sys.platform.startswith("win")
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

W, H = 640, 480

def update_camera(cap, props):
    """Same settings style as your Local_Camera_Config.py tuner."""
    if not props:
        return

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual exposure (OpenCV/V4L2 quirk)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    cap.set(cv2.CAP_PROP_BRIGHTNESS, props.get("brightness", 0))
    cap.set(cv2.CAP_PROP_CONTRAST, props.get("contrast", 0))
    cap.set(cv2.CAP_PROP_EXPOSURE, props.get("exposure", -6))
    cap.set(cv2.CAP_PROP_GAIN, props.get("gain", 0))
    cap.set(cv2.CAP_PROP_SATURATION, props.get("saturation", 64))
    cap.set(cv2.CAP_PROP_SHARPNESS, props.get("sharpness", 100))
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, props.get("white_balance", 4000))

def main():
    if not HW_CFG_PATH.exists():
        print(f"Missing {HW_CFG_PATH}")
        return

    with open(HW_CFG_PATH, "r") as f:
        hw = json.load(f)

    cam_l_idx = hw["cameras"]["left"]["index"]
    cam_r_idx = hw["cameras"]["right"]["index"]

    cam_cfg = None
    if CAM_CFG_PATH.exists():
        with open(CAM_CFG_PATH, "r") as f:
            cam_cfg = json.load(f)

    cap_l = cv2.VideoCapture(cam_l_idx, BACKEND)
    cap_r = cv2.VideoCapture(cam_r_idx, BACKEND)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print(f"Could not open cameras: L={cam_l_idx}, R={cam_r_idx}")
        return

    for cap in (cap_l, cap_r):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    # Apply saved tunings if present
    if cam_cfg:
        update_camera(cap_l, cam_cfg.get("left", {}))
        update_camera(cap_r, cam_cfg.get("right", {}))

    # Warm up
    time.sleep(0.5)
    for _ in range(5):
        cap_l.grab(); cap_r.grab()

    out_dir = BASE_DIR / "calib_pairs"
    out_dir.mkdir(exist_ok=True)

    # Find next index so you don't overwrite existing captures
    existing = sorted(out_dir.glob("left_*.png"))
    idx = 0
    if existing:
        try:
            idx = int(existing[-1].stem.split("_")[-1]) + 1
        except Exception:
            idx = 0

    print("Stereo capture running:")
    print("  SPACE = save pair")
    print("  q     = quit")

    win = "Stereo Preview (L | R)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W * 2, H)

    while True:
        # Flush buffers for "more synchronized" reads
        cap_l.grab(); cap_r.grab()
        ret_l, f_l = cap_l.read()
        ret_r, f_r = cap_r.read()
        if ret_l: f_l = cv2.rotate(f_l, cv2.ROTATE_180)  # Rotate if your cameras are mounted upside down
        if ret_r: f_r = cv2.rotate(f_r, cv2.ROTATE_180)

        if not (ret_l and ret_r):
            print("Frame grab failed.")
            continue

        combined = cv2.hconcat([f_l, f_r])
        cv2.putText(
            combined,
            f"idx={idx} | SPACE save | q quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow(win, combined)

        key = cv2.waitKey(1) & 0xFF

        # SPACE
        if key == 32:
            left_path = out_dir / f"left_{idx:03d}.png"
            right_path = out_dir / f"right_{idx:03d}.png"
            cv2.imwrite(str(left_path), f_l)
            cv2.imwrite(str(right_path), f_r)
            print(f"Saved: {left_path.name}, {right_path.name}")
            idx += 1

        # q
        elif key == ord("q"):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()