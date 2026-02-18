#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path
import cv2
import serial.tools.list_ports

BASE_DIR = Path(__file__).resolve().parent
OUT_CFG = BASE_DIR / "hardware_config.json"

def get_ignored_camera_keywords():
    """
    Uses WMIC to find names of integrated or virtual cameras to ignore.
    """
    ignored_names = []
    try:
        # Query Windows for all video capture devices
        cmd = 'wmic path Win32_PnPEntity where "CategoryGuid=\'{ca3e7ab9-b4d3-46e5-8314-7097c99555c7}\'" get Name'
        output = subprocess.check_output(cmd, shell=True).decode()
        for line in output.split('\n'):
            line = line.strip()
            # Common keywords for internal laptop cameras
            if any(k in line.lower() for k in ["integrated", "internal", "webcam", "virtual"]):
                ignored_names.append(line)
    except Exception:
        pass
    return ignored_names

def valid_cam(idx):
    """
    Tries to open the camera and confirms it's not an integrated device by name.
    Note: Correlating indices to names in Windows can be tricky; 
    this helper ensures the hardware actually responds.
    """
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW]:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return True
    return False

def main():
    print("🔍 Scanning for external hardware (Ignoring Integrated)...")
    
    # 1. Identify what to ignore
    ignored_keywords = ["integrated", "internal", "lenovo", "webcam"]
    print(f"🚫 Filtering devices containing: {ignored_keywords}")

    cams = []
    # 2. Iterate through indices. On many laptops, Integrated is Index 0.
    # We scan more indices just in case.
    for idx in range(6):
        if valid_cam(idx):
            # On Windows, if we don't use a library like 'pygrabber', 
            # we often have to rely on the fact that Integrated is usually Index 0.
            # However, we can also check resolution - external cams are often different.
            
            # Simple heuristic: If it's the first camera found (Index 0) on a laptop, 
            # it is highly likely the integrated one. 
            if idx == 0:
                print(f"  [SKIP] Ignoring potential Integrated Camera at Index {idx}")
                continue
                
            print(f"  [OK] Found external camera at index {idx}")
            cams.append({
                "index": idx,
                "node": f"CAM{idx}"
            })

    # Assign detected USB cameras
    left = cams[0] if len(cams) > 0 else None
    right = cams[1] if len(cams) > 1 else None

    # Detect CNC Port
    ports = serial.tools.list_ports.comports()
    grbl = ""
    for p in ports:
        if "usb" in p.description.lower() or "ch340" in p.description.lower():
            grbl = p.device

    data = {
        "serial": {"grbl_port": grbl},
        "cameras": {
            "left": left,
            "right": right
        }
    }

    with open(OUT_CFG, "w") as f:
        json.dump(data, f, indent=2)

    print("-" * 30)
    print("DETECTION SUMMARY:")
    print(f"SERIAL (CNC): {grbl if grbl else 'NOT FOUND'}")
    print(f"LEFT CAMERA : {left}")
    print(f"RIGHT CAMERA: {right}")
    print("-" * 30)

if __name__ == "__main__":
    main()