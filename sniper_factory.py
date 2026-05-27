import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = r"C:\Users\yebri\Downloads\yolo26n_seg_5-25_new_data_20260526_062530.pt"
SOURCE_DIRS = [
    r"C:\Users\yebri\Box\Laser Weeding Training\training_photos_dand_kale_5-23\left"
]
OUTPUT_DIR = "sniper_test_5-25"
DANDELION_CLASS_ID = 0  # Class 0 is 'dandelion' as per model names

def run_sniper_factory():
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Create output dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    image_count = 0
    sniper_count = 0

    for src_dir in SOURCE_DIRS:
        if not os.path.exists(src_dir):
            print(f"Warning: Directory not found: {src_dir}")
            continue
            
        print(f"\nProcessing directory: {src_dir}")
        # Search for common image formats
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(glob.glob(os.path.join(src_dir, ext)))
        
        # Tag to differentiate left/right in filenames
        dir_tag = "left" if "left" in src_dir.lower() else "right"

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue
            
            image_count += 1
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Run inference
            results = model(img, verbose=False, conf=0.25)[0]
            
            # Check if we have masks and boxes
            if results.masks is None or results.boxes is None:
                continue
            
            # Process detections
            for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
                cls_id = int(box.cls[0])
                if cls_id != DANDELION_CLASS_ID:
                    continue
                
                # 1. Get and resize mask to full image size
                # results.masks.data is typically (N, H_small, W_sqmall)
                m = mask.data[0].cpu().numpy().astype(np.uint8) * 255
                m_resized = cv2.resize(m, (img.shape[1], img.shape[0]))
                
                # 2. Apply mask to original image (isolate plant on black background)
                masked_img = cv2.bitwise_and(img, img, mask=m_resized)
                
                # 3. Crop to bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Add slight padding to the crop (optional, set to 0 for tight crop)
                padding = 5
                h_img, w_img = img.shape[:2]
                x1_p = max(0, x1 - padding)
                y1_p = max(0, y1 - padding)
                x2_p = min(w_img, x2 + padding)
                y2_p = min(h_img, y2 + padding)
                
                crop = masked_img[y1_p:y2_p, x1_p:x2_p]
                
                if crop.size == 0:
                    continue
                
                # 4. Save the snippet
                output_name = f"{dir_tag}_{base_name}_dand_{i}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                cv2.imwrite(output_path, crop)
                sniper_count += 1
                
            if image_count % 20 == 0:
                print(f"Processed {image_count} images, saved {sniper_count} snippets...")

    print(f"\n--- MISSION COMPLETE ---")
    print(f"Images Analyzed: {image_count}")
    print(f"Dandelion Snippets Saved: {sniper_count}")
    print(f"Results folder: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    run_sniper_factory()