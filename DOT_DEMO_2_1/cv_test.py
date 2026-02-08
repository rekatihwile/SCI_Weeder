import cv2
import sys
from cv_helpers import WeedCV

# --- CONFIG ---
YOLO_PATH = "/home/laser/Downloads/final_train_feb_1stbest (2).pt"

MNET_PATH = "/home/laser/Downloads/sniper_jetson_ready.pt"
CAMERA_ID = 2  # Set to 0, 1, or 2 depending on your Jetson/USB setup

def main():
    # 1. Init Helper
    helper = WeedCV(YOLO_PATH, MNET_PATH)
    
    # 2. Init Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {CAMERA_ID}")
        sys.exit()

    # Set resolution (matches your laser script config)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Reduce buffer size for lower latency on Jetson
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"✅ Camera {CAMERA_ID} started.")
    print("Controls: 'q' to Quit, 'space' to Toggle Freeze/Live")

    live_mode = True

    while True:
        if live_mode:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to grab frame")
                break
        
        # 3. Process & Display
        # return_full performs: YOLO -> Filter -> Crop (20% pad) -> MobileNet -> Pixels
        target_coords = helper.return_full(frame)
        
        # show_debug handles the 3 resizable windows: Main, Binary Mask, and Masked RGB
        helper.show_debug(frame, target_coords)

        # 4. Input handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord(' '):  # Spacebar toggles live/freeze
            live_mode = not live_mode
            status = "LIVE" if live_mode else "FROZEN"
            print(f"📺 Feed is now {status}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()