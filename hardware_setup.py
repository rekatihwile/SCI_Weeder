import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import serial.tools.list_ports


BASE_DIR = Path(__file__).resolve().parent
OUT_CFG = BASE_DIR / "params" / "hardware" / "hardware_config.json"

LEFT_PORT_HINT = ""
RIGHT_PORT_HINT = ""

SCOUT_CAN_CHANNEL = "can0"
SCOUT_CAN_BITRATE = 500000


def _run_privileged(cmd, *, input_text=None):
    """Try command directly first, then fallback to sudo -n for unattended runs."""
    direct = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    if direct.returncode == 0:
        return direct, "direct"

    if direct.returncode != 0 and any(k in (direct.stderr or "").lower() for k in ["permission denied", "permission problem", "operation not permitted"]):
        sudo_cmd = ["sudo", "-n", *cmd]
        sudo_res = subprocess.run(sudo_cmd, input=input_text, capture_output=True, text=True)
        return sudo_res, "sudo"

    return direct, "direct"


def detect_scout(channel=None, bitrate=None):
    """Check if the Scout SocketCAN interface exists and report its state."""
    channel = channel or SCOUT_CAN_CHANNEL
    bitrate = bitrate or SCOUT_CAN_BITRATE

    # Check if the interface exists at all
    net_path = Path(f"/sys/class/net/{channel}")
    if not net_path.exists():
        return {"found": False, "channel": channel, "state": "absent"}

    # Read operational state
    try:
        state = (net_path / "operstate").read_text().strip()
    except Exception:
        state = "unknown"

    # Read actual bitrate from sysfs if available
    detected_bitrate = None
    try:
        bitrate_path = net_path / "statistics" / "bitrate"
        if not bitrate_path.exists():
            # Try the CAN-specific location
            bitrate_path = Path(f"/sys/class/net/{channel}/can_bitrate")
        if bitrate_path.exists():
            detected_bitrate = int(bitrate_path.read_text().strip())
    except Exception:
        pass

    is_up = state == "up"
    return {
        "found": True,
        "channel": channel,
        "state": state,
        "is_up": is_up,
        "bitrate": detected_bitrate or bitrate,
    }


def detect_grbl_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        dev = (p.device or "").lower()
        if any(k in desc or k in dev for k in ["ch340", "usb", "ttyusb", "ttyacm", "cp210"]):
            return p.device
    return ""


def camera_works(device, backend=None):
    cap = cv2.VideoCapture(device, backend) if backend is not None else cv2.VideoCapture(device)
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
    by_path_dir = Path("/dev/v4l/by-path")
    entries = []
    if not by_path_dir.exists():
        return entries

    for p in sorted(by_path_dir.glob("*video-index0")):
        try:
            resolved = p.resolve()
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

def reset_cameras_uhubctl(hub_loc="1-4", ports=(4,), delay=3):
    print(f"\n=== UHUBCTL CAMERA RESET (hub {hub_loc}) ===")
    print(f"[RESET] Cycling entire hub {hub_loc}...")
    result, mode = _run_privileged([
        "uhubctl", "-l", hub_loc, "-a", "cycle", "-d", str(delay)
    ])
    if result.returncode != 0:
        print(f"[RESET] Hub cycle FAILED: {result.stderr.strip()}")
        if mode == "sudo":
            print("[RESET] Hint: allow passwordless sudo for uhubctl to run unattended.")
        else:
            print("[RESET] Hint: run once with sudo or configure udev permissions for uhubctl.")
        return False

    # Poll until cameras are actually readable by OpenCV (max 15s)
    print("[RESET] Waiting for cameras to become readable...")
    for i in range(15):
        time.sleep(1)
        cams = detect_cameras_linux()
        print(f"[RESET] ...{i+1}s: {len(cams)} camera(s) ready")
        if len(cams) >= 2:
            break

    print("[RESET] Camera port reset complete.")
    return True


def nuclear_reset_usb_hub(hub_port="1-4"):
    # Try surgical uhubctl reset first
    if subprocess.run(["which", "uhubctl"], capture_output=True).returncode == 0:
        if reset_cameras_uhubctl():
            return True
        print("[RESET] uhubctl path failed; falling back to sysfs unbind/bind reset...")

    # Fallback: full hub unbind/rebind
    print(f"\n=== NUCLEAR USB RESET (hub {hub_port}) ===")
    for action, wait_after in [("unbind", 5.0), ("bind", 3.0)]:
        path = f"/sys/bus/usb/drivers/usb/{action}"
        print(f"[NUCLEAR] {action} -> {hub_port}")
        result, mode = _run_privileged(["tee", path], input_text=hub_port)
        if result.returncode != 0:
            print(f"[NUCLEAR] {action} FAILED (rc={result.returncode}): {result.stderr.strip()}")
            if mode == "sudo":
                print("[NUCLEAR] Hint: allow passwordless sudo for sysfs usb bind/unbind writes.")
            else:
                print("[NUCLEAR] Hint: run once with sudo or configure permission for sysfs usb bind/unbind.")
            return False
        time.sleep(wait_after)
    print("[NUCLEAR] USB hub reset complete.")
    return True

def assign_linux_left_right(cam_entries):
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

    ordered = sorted(cam_entries, key=lambda c: c["name"])
    return ordered[0], ordered[1]


def detect_display():
    system = platform.system()
    display_env = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or ""

    if system == "Windows":
        return {
            "has_display": True,
            "headless": False,
            "ui_mode": "window",
            "display_backend": "windows",
        }

    has_display = bool(display_env)
    return {
        "has_display": has_display,
        "headless": not has_display,
        "ui_mode": "window" if has_display else "headless",
        "display_backend": display_env,
    }


def build_config():
    system = platform.system()
    config = {
        "serial": {"grbl_port": ""},
        "cameras": {"left": None, "right": None},
        "scout": {"found": False, "channel": SCOUT_CAN_CHANNEL},
        "runtime": detect_display(),
        "detected_stereo_pair": False,
    }

    print(f"Detecting hardware on {system}...")

    # --- Scout detection (Linux / SocketCAN only) ---
    if system == "Linux":
        scout_info = detect_scout()
        config["scout"] = scout_info
        if scout_info["found"]:
            status = "UP" if scout_info.get("is_up") else scout_info["state"].upper()
            print(f"[OK] Scout CAN interface {scout_info['channel']!r} found (state={status}, bitrate={scout_info['bitrate']})")
        else:
            print(f"[WARNING] Scout CAN interface {scout_info['channel']!r} not found (no SocketCAN device).")
            config["scout"] = {"found": False, "channel": SCOUT_CAN_CHANNEL}
    else:
        config["scout"] = {"found": False, "channel": SCOUT_CAN_CHANNEL}

    grbl_port = detect_grbl_port()
    if grbl_port:
        config["serial"]["grbl_port"] = grbl_port
        print(f"[OK] Found GRBL port: {grbl_port}")
    else:
        print("[WARNING] No GRBL port detected.")

    runtime = config["runtime"]
    print(f"[INFO] has_display = {runtime['has_display']}")
    print(f"[INFO] ui_mode     = {runtime['ui_mode']}")
    if runtime["display_backend"]:
        print(f"[INFO] display    = {runtime['display_backend']}")

    if system == "Linux":
        cams = detect_cameras_linux()
        if len(cams) < 2:
            print(f"[WARNING] Found {len(cams)} usable Linux cameras. Attempting nuclear USB reset...")
            if nuclear_reset_usb_hub():
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
                    "node": "LEFT_CAM",
                }
                config["cameras"]["right"] = {
                    "index": int(right_cam["device"].replace("/dev/video", "")),
                    "device": right_cam["device"],
                    "path": right_cam["path"],
                    "node": "RIGHT_CAM",
                }
                print("\n[OK] Assigned LEFT camera:")
                print(f"     path   = {left_cam['path']}")
                print(f"     device = {left_cam['device']}")
                print("[OK] Assigned RIGHT camera:")
                print(f"     path   = {right_cam['path']}")
                print(f"     device = {right_cam['device']}")
                config["detected_stereo_pair"] = True
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
                "node": "RIGHT_CAM",
            }
            config["cameras"]["left"] = {
                "index": cam_indices[1],
                "device": None,
                "path": None,
                "node": "LEFT_CAM",
            }
            print(f"[OK] Right camera index: {cam_indices[0]}")
            print(f"[OK] Left camera index:  {cam_indices[1]}")
            config["detected_stereo_pair"] = True
        else:
            print(f"[ERROR] Found {len(cam_indices)} usable cameras. Need 2.")

    return config


def save_config(config):
    # If detection failed to find both cameras, preserve the last known good
    # camera mapping so runtime imports and recovery tools keep working.
    if (
        (config.get("cameras", {}).get("left") is None or config.get("cameras", {}).get("right") is None)
        and OUT_CFG.exists()
    ):
        try:
            with open(OUT_CFG, "r") as f:
                prev = json.load(f)
            prev_left = (prev or {}).get("cameras", {}).get("left")
            prev_right = (prev or {}).get("cameras", {}).get("right")
            if prev_left is not None and prev_right is not None:
                print("[SAVE] Camera detection incomplete; preserving previous left/right camera config.")
                config.setdefault("cameras", {})["left"] = prev_left
                config.setdefault("cameras", {})["right"] = prev_right
        except Exception:
            pass

    with open(OUT_CFG, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved: {OUT_CFG}")


def main():
    config = build_config()
    save_config(config)
    if not bool(config.get("detected_stereo_pair", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
