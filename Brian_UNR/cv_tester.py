import os
import sys
import glob
import cv2
import numpy as np

# Add workspace to path so cv_helpers can be imported
sys.path.append("/home/laser/Documents/Laser_Workspace/SCI_Weeder")
from cv_helpers import WeedCV

# HEIC support requires pillow-heif (install via: pip install pillow-heif)
try:
    import pillow_heif
    from PIL import Image
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("WARNING: pillow_heif not installed. HEIC files will be skipped.")
    print("To fix: pip install pillow-heif")

def read_image(path):
    """Reads JPGs normally and HEICs via pillow_heif."""
    if path.lower().endswith('.heic'):
        if not HEIC_SUPPORT:
            return None
        heif_file = pillow_heif.read_heif(path)
        img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv2.imread(path)

def main():
    # --- CONFIGURATION ---
    img_dir = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/Minimal_Demo_Gilcrease/pigsweed_test"
    
    # UPDATE THESE PATHS to your actual model files
    YOLO_PATH = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/Minimal_Demo_Gilcrease/weights/pigweed-yolo.pt" 
    MOBILENET_PATH = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/Minimal_Demo_Gilcrease/weights/sniper.pt"
    
    # Initialize your WeedCV class
    weed_cv = WeedCV(yolo_path=YOLO_PATH, mobilenet_path=MOBILENET_PATH)

    # Gather images
    extensions = ['*.jpg', '*.jpeg', '*.HEIC', '*.heic']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    
    image_paths = sorted(image_paths)
    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    # State variables
    idx = 0
    last_idx = -1
    show_overlays = True
    coords = []
    current_img = None

    print("\n--- Controls ---")
    print(" 'd': Next Image")
    print(" 'a': Previous Image")
    print(" 'h': Toggle Overlays (Fast)")
    print(" 'q': Quit\n")

    while True:
        img_path = image_paths[idx]
        
        # Only read and run inference if we moved to a new image
        if idx != last_idx:
            current_img = read_image(img_path)
            
            if current_img is None:
                print(f"Failed to read {os.path.basename(img_path)}. Skipping...")
                idx = (idx + 1) % len(image_paths)
                last_idx = idx
                continue
            
            print(f"[{idx+1}/{len(image_paths)}] Inferencing: {os.path.basename(img_path)}")
            # This populates weed_cv.filtered_boxes and returns the target points
            coords = weed_cv.return_full(current_img)
            last_idx = idx

        # Display Logic
        if show_overlays:
            # Uses your built-in debug HUD
            weed_cv.show_debug(current_img, coords)
        else:
            # Show raw image using the same window properties
            cv2.namedWindow(weed_cv.win_main, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(weed_cv.win_main, 800, int(current_img.shape[0] * (800 / current_img.shape[1])))
            cv2.imshow(weed_cv.win_main, current_img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            idx = (idx + 1) % len(image_paths)
        elif key == ord('a'):
            idx = (idx - 1) % len(image_paths)
        elif key == ord('h'):
            show_overlays = not show_overlays

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()