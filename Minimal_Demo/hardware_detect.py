#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import serial.tools.list_ports

# -----------------------
# SILENCE OPENCV NOISE
# -----------------------
os.environ["OPENCV_LOG_LEVEL"] = "FATAL" # Hides the [WARN] and [ERROR] messages

BASE_DIR = Path(__file__).resolve().parent
OUT_CFG = BASE_DIR / "hardware_config.json"

def detect_grbl_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        dev = p.device or ""
        if "/dev/ttyUSB" in dev or "/dev/ttyACM" in dev:
            return dev
    return ""

def _is_valid_cam(idx: int) -> bool:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return bool(ret)

def get_camera_mapping() -> List[Dict]:
    """
    Returns a list of cameras sorted by their PHYSICAL USB port path.
    On most Jetson/Linux boards, the alphabetical order of 'by-path' 
    corresponds to the physical stack of USB ports.
    """
    by_path_dir = Path("/dev/v4l/by-path")
    if not by_path_dir.exists():
        return []

    found_cams = []
    # Sort paths alphabetically. The 'top' port usually has a lower index/path string.
    for path_link in sorted(by_path_dir.iterdir(), reverse=True):
        if "-video-index0" in path_link.name:  # Only look at primary video nodes
            dev_node = os.path.realpath(str(path_link))
            try:
                idx = int(dev_node.replace("/dev/video", ""))
                if _is_valid_cam(idx):
                    found_cams.append({
                        "index": idx,
                        "usb_path": path_link.name,
                        "node": dev_node
                    })
            except:
                continue
    return found_cams


def main():
    print("🔍 Probing hardware for specific port addresses...")
    
    grbl_port = detect_grbl_port()
    cams = get_camera_mapping()
    
    # Create a lookup dictionary of {usb_path: video_index}
    # Example: {"4.4": 0, "4.3": 2}
    lookup = {c["usb_path"]: c["index"] for c in cams}

    # Explicitly check for your specific physical port fingerprints
    # '4.4' is your top port, '4.3' is your bottom port
    left_cam = -1
    right_cam = -1

    for path, idx in lookup.items():
        if "4.4" in path:
            left_cam = idx
        elif "4.3" in path:
            right_cam = idx

    data = {
        "serial": {"grbl_port": grbl_port},
        "cameras": {
            "left": left_cam, 
            "right": right_cam
        },
    }

    with open(OUT_CFG, "w") as f:
        json.dump(data, f, indent=2)

    print("\n--- FINAL ASSIGNMENT ---")
    print(f"✅ LEFT (Top Port 4.4):    {'NOT FOUND' if left_cam == -1 else '/dev/video' + str(left_cam)}")
    print(f"✅ RIGHT (Bottom Port 4.3): {'NOT FOUND' if right_cam == -1 else '/dev/video' + str(right_cam)}")
    print(f"✅ SERIAL: {grbl_port if grbl_port else 'NOT FOUND'}")


if __name__ == "__main__":
    main()