#!/usr/bin/env python3
import json
import os
from pathlib import Path
import cv2
import serial.tools.list_ports

BASE_DIR = Path(__file__).resolve().parent
OUT_CFG = BASE_DIR / "hardware_config.json"

def detect_grbl_port():
    for p in serial.tools.list_ports.comports():
        if "/dev/ttyUSB" in (p.device or "") or "/dev/ttyACM" in (p.device or ""):
            return p.device
    return ""

def valid_cam(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return ok

def main():
    cams = []
    by_path = Path("/dev/v4l/by-path")

    for p in sorted(by_path.glob("*-video-index0")):
        node = os.path.realpath(p)
        idx = int(node.replace("/dev/video", ""))
        if valid_cam(idx):
            cams.append({
                "index": idx,
                "by_path": str(p),
                "node": node
            })

    left = right = None
    for c in cams:
        if "4.3" in c["by_path"]:
            right = c   # <-- swapped
        elif "4.4" in c["by_path"]:
            left = c    # <-- swapped


    data = {
        "serial": {"grbl_port": detect_grbl_port()},
        "cameras": {
            "left": left,
            "right": right
        }
    }

    with open(OUT_CFG, "w") as f:
        json.dump(data, f, indent=2)

    print("LEFT :", left)
    print("RIGHT:", right)

if __name__ == "__main__":
    main()
