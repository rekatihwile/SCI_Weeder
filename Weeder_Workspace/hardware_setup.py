import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import json
import platform
from pathlib import Path

import cv2
import serial.tools.list_ports


BASE_DIR = Path(__file__).resolve().parent
OUT_CFG = BASE_DIR / "params" / "hardware_config.json"


# Optional Linux hints:
# Put a unique substring from the by-path name here once you know it.
# Example after inspection:
# LEFT_PORT_HINT = "usb-0:2"
# RIGHT_PORT_HINT = "usb-0:3"
LEFT_PORT_HINT = ""
RIGHT_PORT_HINT = ""


def detect_grbl_port():
    ports = serial.tools.list_ports.comports()

    for p in ports:
        desc = (p.description or "").lower()
        dev = (p.device or "").lower()

        if any(k in desc or k in dev for k in ["ch340", "usb", "ttyusb", "ttyacm", "cp210"]):
            return p.device

    return ""


def camera_works(device, backend=None):
    if backend is None:
        cap = cv2.VideoCapture(device)
    else:
        cap = cv2.VideoCapture(device, backend)

    if not cap.isOpened():
        cap.release()
        return False

    ret, _ = cap.read()
    cap.release()
    return bool(ret)


def detect_cameras_windows(max_index=10):
    found = []

    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(i)
        cap.release()

    return found


def detect_cameras_linux():
    """
    Use /dev/v4l/by-path so camera assignment is based on physical USB path,
    not random /dev/videoX ordering.
    """
    by_path_dir = Path("/dev/v4l/by-path")
    entries = []

    if not by_path_dir.exists():
        return entries

    for p in sorted(by_path_dir.glob("*video-index0")):
        try:
            resolved = p.resolve()   # /dev/videoX
            device = str(resolved)
            path_str = str(p)

            if camera_works(device, cv2.CAP_V4L2):
                entries.append({
                    "path": path_str,
                    "device": device,
                    "name": p.name,
                })
        except Exception:
            pass

    return entries


def assign_linux_left_right(cam_entries):
    """
    Prefer explicit path hints if provided.
    Otherwise fall back to sorted by-path ordering.
    """
    if len(cam_entries) < 2:
        return None, None

    left = None
    right = None

    if LEFT_PORT_HINT:
        for cam in cam_entries:
            if LEFT_PORT_HINT in cam["name"] or LEFT_PORT_HINT in cam["path"]:
                left = cam
                break

    if RIGHT_PORT_HINT:
        for cam in cam_entries:
            if RIGHT_PORT_HINT in cam["name"] or RIGHT_PORT_HINT in cam["path"]:
                right = cam
                break

    if left is not None and right is not None and left != right:
        return left, right

    # fallback: stable sorted order by physical path
    ordered = sorted(cam_entries, key=lambda c: c["name"])
    return ordered[0], ordered[1]


def build_config():
    system = platform.system()

    config = {
        "serial": {
            "grbl_port": ""
        },
        "cameras": {
            "left": None,
            "right": None
        }
    }

    print(f"Detecting hardware on {system}...")

    # GRBL
    grbl_port = detect_grbl_port()
    if grbl_port:
        config["serial"]["grbl_port"] = grbl_port
        print(f"[OK] Found GRBL port: {grbl_port}")
    else:
        print("[WARNING] No GRBL port detected.")

    # Cameras
    if system == "Linux":
        cams = detect_cameras_linux()

        if len(cams) < 2:
            print(f"[ERROR] Found {len(cams)} usable Linux cameras by physical path. Need 2.")
        else:
            print("\nDetected Linux cameras by physical path:")
            for i, cam in enumerate(cams):
                print(f"  [{i}] path={cam['path']}")
                print(f"      device={cam['device']}")

            left_cam, right_cam = assign_linux_left_right(cams)

            if left_cam is None or right_cam is None:
                print("[ERROR] Failed to assign left/right cameras.")
            else:
                config["cameras"]["left"] = {
                    "index": int(left_cam["device"].replace("/dev/video", "")),
                    "device": left_cam["device"],
                    "path": left_cam["path"],
                    "node": "LEFT_CAM"
                }
                config["cameras"]["right"] = {
                    "index": int(right_cam["device"].replace("/dev/video", "")),
                    "device": right_cam["device"],
                    "path": right_cam["path"],
                    "node": "RIGHT_CAM"
                }

                print("\n[OK] Assigned LEFT camera:")
                print(f"     path   = {left_cam['path']}")
                print(f"     device = {left_cam['device']}")

                print("[OK] Assigned RIGHT camera:")
                print(f"     path   = {right_cam['path']}")
                print(f"     device = {right_cam['device']}")

                if not LEFT_PORT_HINT or not RIGHT_PORT_HINT:
                    print("\n[INFO] No explicit port hints set.")
                    print("       Using sorted physical USB path order.")
                    print("       If top/bottom is reversed on Jetson, set:")
                    print('       LEFT_PORT_HINT = "..."')
                    print('       RIGHT_PORT_HINT = "..."')

    else:
        cam_indices = detect_cameras_windows()

        if system == "Windows" and len(cam_indices) > 2:
            print(f"[INFO] Detected {len(cam_indices)} cameras. Skipping camera 0.")
            cam_indices = [idx for idx in cam_indices if idx != 0]

        if len(cam_indices) >= 2:
            config["cameras"]["right"] = {
                "index": cam_indices[0],
                "device": None,
                "path": None,
                "node": "RIGHT_CAM"
            }
            config["cameras"]["left"] = {
                "index": cam_indices[1],
                "device": None,
                "path": None,
                "node": "LEFT_CAM"
            }

            print(f"[OK] Left camera index:  {cam_indices[0]}")
            print(f"[OK] Right camera index: {cam_indices[1]}")
            print("[INFO] Windows mode uses index ordering, not guaranteed physical USB port mapping.")
        else:
            print(f"[ERROR] Found {len(cam_indices)} usable cameras. Need 2.")

    return config


def save_config(config):
    with open(OUT_CFG, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nSaved: {OUT_CFG}")


def main():
    config = build_config()
    save_config(config)


if __name__ == "__main__":
    main()