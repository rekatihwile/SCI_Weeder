import cv2
import json
import numpy as np
import sys
import time
from pathlib import Path

# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parent
HW_CFG_PATH = BASE_DIR / "hardware_config.json"
CAM_CFG_PATH = BASE_DIR / "camera_config.json"

CALIB_NPZ = BASE_DIR / "stereo_charuco_calib.npz"
RECT_NPZ  = BASE_DIR / "stereo_rectify_maps.npz"

# ===== Options =====
ROTATE_180_LEFT  = True   # set True only if the LEFT camera is physically upside down
ROTATE_180_RIGHT = True   # set True only if the RIGHT camera is physically upside down

W, H = 640, 480

IS_WINDOWS = sys.platform.startswith("win")
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

def update_camera(cap, props):
    if not props:
        return
    # These property names/semantics vary by camera/backend; keep minimal
    if "exposure" in props:
        cap.set(cv2.CAP_PROP_EXPOSURE, props["exposure"])
    if "gain" in props:
        cap.set(cv2.CAP_PROP_GAIN, props["gain"])
    if "brightness" in props:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, props["brightness"])
    if "contrast" in props:
        cap.set(cv2.CAP_PROP_CONTRAST, props["contrast"])

def triangulate_point(uL, vL, uR, vR, K1, D1, K2, D2, R1, P1, R2, P2):
    # clicked pixels -> undistort+rectify points into rectified pixel coord system
    ptsL = np.array([[[uL, vL]]], dtype=np.float32)
    ptsR = np.array([[[uR, vR]]], dtype=np.float32)

    ptsLr = cv2.undistortPoints(ptsL, K1, D1, R=R1, P=P1)  # (1,1,2)
    ptsRr = cv2.undistortPoints(ptsR, K2, D2, R=R2, P=P2)

    xL, yL = float(ptsLr[0, 0, 0]), float(ptsLr[0, 0, 1])
    xR, yR = float(ptsRr[0, 0, 0]), float(ptsRr[0, 0, 1])

    # triangulate in rectified-left frame
    X_h = cv2.triangulatePoints(
        P1, P2,
        np.array([[xL], [yL]], dtype=np.float32),
        np.array([[xR], [yR]], dtype=np.float32),
    )
    X = (X_h[:3] / X_h[3]).reshape(3)  # meters
    return X

def main():
    # --- Load camera indices ---
    with open(HW_CFG_PATH, "r") as f:
        hw = json.load(f)

    cam_l_idx = hw["cameras"]["left"]["index"]
    cam_r_idx = hw["cameras"]["right"]["index"]

    cam_cfg = None
    if CAM_CFG_PATH.exists():
        with open(CAM_CFG_PATH, "r") as f:
            cam_cfg = json.load(f)

    # --- Load calibration ---
    calib = np.load(CALIB_NPZ)
    rect  = np.load(RECT_NPZ)

    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    T      = calib["T"].reshape(3)      # left->right in LEFT camera frame

    R1, P1 = rect["R1"], rect["P1"]
    R2, P2 = rect["R2"], rect["P2"]

    # Rotate T into rectified-left frame so we can shift origin to midpoint correctly
    T_rect = (R1 @ T.reshape(3, 1)).reshape(3)

    # --- Open cameras ---
    capL = cv2.VideoCapture(cam_l_idx, BACKEND)
    capR = cv2.VideoCapture(cam_r_idx, BACKEND)
    if not capL.isOpened() or not capR.isOpened():
        print(f"Could not open cameras L={cam_l_idx}, R={cam_r_idx}")
        return

    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    if cam_cfg:
        update_camera(capL, cam_cfg.get("left", {}))
        update_camera(capR, cam_cfg.get("right", {}))

    time.sleep(0.25)

    state = {
        "L_click": None,
        "R_click": None,
        "L_frozen": None,
        "R_frozen": None,
        "freeze_L": False,
        "freeze_R": False,
        "last_L": None,
        "last_R": None,
    }

    def on_mouse_L(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["last_L"] is None:
                return
            state["L_click"] = (x, y)
            state["L_frozen"] = state["last_L"].copy()
            state["freeze_L"] = True
            print(f"Left click:  ({x}, {y}) [froze LEFT frame]")

    def on_mouse_R(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["last_R"] is None:
                return
            state["R_click"] = (x, y)
            state["R_frozen"] = state["last_R"].copy()
            state["freeze_R"] = True
            print(f"Right click: ({x}, {y}) [froze RIGHT frame]")

    cv2.namedWindow("Left")
    cv2.namedWindow("Right")
    cv2.setMouseCallback("Left", on_mouse_L)
    cv2.setMouseCallback("Right", on_mouse_R)

    print("Live stereo triangulation:")
    print("  - Click LEFT, then click the same physical point in RIGHT.")
    print("  - t = triangulate, r = resume live, c = clear clicks, q = quit")
    print("Note: Image +Y is downward (normal). That doesn't mean the camera is flipped.\n")

    while True:
        # Update frames unless frozen
        if not state["freeze_L"]:
            capL.grab()
            okL, frameL = capL.read()
            if okL:
                if ROTATE_180_LEFT:
                    frameL = cv2.rotate(frameL, cv2.ROTATE_180)
                state["last_L"] = frameL

        if not state["freeze_R"]:
            capR.grab()
            okR, frameR = capR.read()
            if okR:
                if ROTATE_180_RIGHT:
                    frameR = cv2.rotate(frameR, cv2.ROTATE_180)
                state["last_R"] = frameR

        showL = state["L_frozen"] if state["freeze_L"] and state["L_frozen"] is not None else state["last_L"]
        showR = state["R_frozen"] if state["freeze_R"] and state["R_frozen"] is not None else state["last_R"]
        if showL is None or showR is None:
            continue

        visL = showL.copy()
        visR = showR.copy()

        if state["L_click"] is not None:
            cv2.circle(visL, state["L_click"], 6, (0, 255, 255), 2)
        if state["R_click"] is not None:
            cv2.circle(visR, state["R_click"], 6, (0, 255, 255), 2)

        cv2.putText(visL, "LEFT (click to freeze)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(visR, "RIGHT (click to freeze)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Left", visL)
        cv2.imshow("Right", visR)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

        if key == ord("r"):
            state["freeze_L"] = False
            state["freeze_R"] = False
            state["L_frozen"] = None
            state["R_frozen"] = None
            print("Resumed live view (both).")

        if key == ord("c"):
            state["L_click"] = None
            state["R_click"] = None
            print("Cleared clicks.")

        if key == ord("t"):
            if state["L_click"] is None or state["R_click"] is None:
                print("Need both clicks first.")
                continue

            uL, vL = state["L_click"]
            uR, vR = state["R_click"]

            X_rect = triangulate_point(uL, vL, uR, vR, K1, D1, K2, D2, R1, P1, R2, P2)

            # Midpoint-between-cameras origin (same orientation as rectified-left frame)
            X_mid = X_rect - 0.5 * T_rect

            print(f"XYZ_rect(left origin): {X_rect[0]*1000:.2f}, {X_rect[1]*1000:.2f}, {X_rect[2]*1000:.2f} mm")
            print(f"XYZ_mid (mid origin):  {X_mid[0]*1000:.2f}, {X_mid[1]*1000:.2f}, {X_mid[2]*1000:.2f} mm\n")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()