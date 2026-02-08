import cv2
import subprocess

def get_usb_devices():
    """Prints all USB devices currently recognized by the system."""
    print("--- USB Device List ---")
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Could not run lsusb: {e}")

def scan_camera_indices():
    """Scans indices 0-10 to find active OpenCV camera IDs."""
    print("\n--- Scanning Camera Indices ---")
    available_cameras = []
    
    # We check 0 through 10 as most USB hubs won't exceed this
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            # Some virtual nodes (like metadata ports) open but don't return frames
            if ret:
                print(f"[FOUND] Index {i}: Camera is ACTIVE and returning frames.")
                available_cameras.append(i)
            else:
                print(f"[INFO]  Index {i}: Device opened, but NO FRAME received (likely metadata).")
            cap.release()
        else:
            pass # Index is empty

    if not available_cameras:
        print("❌ NO ACTIVE CAMERAS FOUND. Check your USB connections.")
    else:
        print(f"\n✅ Scan complete. Use these indices in your config: {available_cameras}")

if __name__ == "__main__":
    get_usb_devices()
    scan_camera_indices()