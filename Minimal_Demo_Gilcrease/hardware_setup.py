import os
# Suppress OpenCV C++ internal logging
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import platform
import json
import serial.tools.list_ports
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_CFG = BASE_DIR / "hardware_config.json"

def detect_cameras():
    found = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(i)
            cap.release()
    return found

def main():
    system = platform.system()
    config = {"serial": {"grbl_port": ""}, "cameras": {"left": None, "right": None}}

    print(f"🔍 Detecting hardware on {system}...")

    # 1. Detect Serial Port for GRBL
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = p.description.lower()
        dev = p.device.lower()
        if any(k in desc or k in dev for k in ["ch340", "usb", "ttyusb", "ttyacm", "cp210"]):
            config["serial"]["grbl_port"] = p.device
            print(f"  [OK] Found Serial Port: {p.device}")
            break
    
    if not config["serial"]["grbl_port"]:
        print("  [WARNING] No Serial/GRBL port detected! Check USB cables and drivers.")

    # 2. Detect Cameras
    cam_indices = detect_cameras()
    if system == "Windows" and len(cam_indices) > 2:
        print(f"  [INFO] Detected {len(cam_indices)} cameras. Skipping Index 0 (Integrated).")
        cam_indices = [idx for idx in cam_indices if idx != 0]

    if len(cam_indices) >= 2:
        config["cameras"]["left"] = {"index": cam_indices[0], "node": f"CAM{cam_indices[0]}"}
        config["cameras"]["right"] = {"index": cam_indices[1], "node": f"CAM{cam_indices[1]}"}
        print(f"  [OK] Left Camera Index: {cam_indices[0]}")
        print(f"  [OK] Right Camera Index: {cam_indices[1]}")
    else:
        print(f"  [ERROR] Found {len(cam_indices)} cameras. Need 2 for stereo.")

    with open(OUT_CFG, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\n✅ hardware_config.json updated.")

if __name__ == "__main__":
    main()