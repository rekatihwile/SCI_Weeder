import os
import cv2
import glob
import numpy as np

# Apply torchvision.ops.nms fallback patch before importing Ultralytics/YOLO.
# This avoids Jetson crashes when torchvision C++ ops are unavailable.
import runpy

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_NMS_PATCH_PATH = os.path.join(_ROOT_DIR, "bringup", "_nms_patch.py")
if os.path.exists(_NMS_PATCH_PATH):
    runpy.run_path(_NMS_PATCH_PATH, run_name="__nms_patch__")
else:
    print(f"[WARN] NMS patch not found: {_NMS_PATCH_PATH}")

from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = r"/home/eli/LaserWeeder_CleanRuntime/params/cv_weights/yolo26n_seg_5-25_new_data_20260526_062530.pt"
TEST_DIR = r"/home/eli/LaserWeeder_CleanRuntime/training_photos/left"
OUTPUT_DIR = r"/home/eli/LaserWeeder_CleanRuntime/dev_tools/cache/eval_annotated"

# Colors (B, G, R)
COLOR_GT = (0, 255, 0)      # Green for Ground Truth
CLASS_COLORS = [
    (255, 255, 0),  # Cyan-ish for Class 0 (Dandelion)
    (255, 0, 255),  # Magenta for Class 1 (Kale)
    (0, 255, 255),  # Yellow for Class 2
    (255, 128, 0),  # Orange for Class 3
]

def create_yaml_for_validation(classes):
    """YOLO needs a yaml file to run official validation metrics. We generate one dynamically."""
    yaml_path = os.path.join(TEST_DIR, "temp_eval.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {TEST_DIR}\n")
        f.write("train: images\n")
        f.write("val: images\n") # We validate directly on the images folder
        f.write("names:\n")
        for cls_id, cls_name in classes.items():
            f.write(f"  {cls_id}: {cls_name}\n")
    return yaml_path

def read_gt_boxes(txt_path, img_w, img_h):
    """Reads the normalized ground truth polygons/boxes and returns absolute bounding boxes."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            
            # Extract x, y coordinates (polygon or box)
            pts_x = [parts[i] * img_w for i in range(1, len(parts), 2)]
            pts_y = [parts[i] * img_h for i in range(2, len(parts), 2)]
            
            if not pts_x: continue
            
            # Get bounding box from the points
            x1, y1 = int(min(pts_x)), int(min(pts_y))
            x2, y2 = int(max(pts_x)), int(max(pts_y))
            boxes.append((class_id, x1, y1, x2, y2))
            
    return boxes

def draw_label(img, text, pos, color, thickness=2, font_scale=0.6):
    """Draws text with a background for readability."""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rect
    cv2.rectangle(img, (x, y - th - 5), (x + tw, y + 5), color, -1)
    # Draw text (white or black depending on color brightness)
    text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    return th + 10 # Return height for offset

def run_evaluation_and_export():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    classes = model.names
    print(f"Model classes detected: {classes}")
    
    # --- PHASE 1: Accuracy Metrics ---
    print("\n--- PHASE 1: Calculating Accuracy Metrics ---")
    yaml_path = create_yaml_for_validation(classes)
    
    try:
        metrics = model.val(data=yaml_path, split='val', plots=False)
        print("\nMetrics calculation complete!")
    except Exception as e:
        print(f"Could not run built-in validation. Error: {e}")
        
    # --- PHASE 2: Headless export of annotated images ---
    print("\n--- PHASE 2: Saving Annotated Images (Headless) ---")
    
    image_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")) + \
                         glob.glob(os.path.join(TEST_DIR, "*.png")))
    
    if not image_paths:
        print(f"No images found in {TEST_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    show_mode = 0  # 0: Both, 1: GT Only, 2: Pred Only, 3: None
    show_text = True
    show_clutter = True
    mode_names = {0: "BOTH", 1: "GT ONLY", 2: "PRED ONLY", 3: "NONE"}

    saved_count = 0
    total_detections = 0
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue
            
        h, w, _ = img.shape
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(TEST_DIR, "labels", base_name + ".txt")
        
        gt_boxes = read_gt_boxes(txt_path, w, h)
        results = model(img, verbose=False)[0]
        total_detections += len(results.boxes)
        
        disp_img = img.copy()
        
        # Track label positions to prevent overlap
        # Key: (x_bucket, y_bucket), Value: offset
        occupied_top = {} 
        occupied_bottom = {}

        def get_offset(x, y, tracker, direction=1):
            bucket = (x // 50, y // 20)
            offset = tracker.get(bucket, 0)
            tracker[bucket] = offset + (direction * 25)
            return offset

        # Draw Ground Truth (Green)
        if show_mode in [0, 1]:
            for cls_id, x1, y1, x2, y2 in gt_boxes:
                if show_clutter:
                    cv2.rectangle(disp_img, (x1, y1), (x2, y2), COLOR_GT, 2)
                if show_text:
                    label = f"GT: {classes.get(cls_id, str(cls_id))}"
                    offset = get_offset(x1, y1, occupied_top, direction=-1)
                    draw_label(disp_img, label, (x1, max(20, y1 + offset - 5)), COLOR_GT)

        # Draw Predictions
        if show_mode in [0, 2]:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                
                if show_clutter:
                    cv2.rectangle(disp_img, (x1, y1), (x2, y2), color, 2)
                
                if show_text:
                    label = f"{classes.get(cls_id, str(cls_id))} {conf:.2f}"
                    offset = get_offset(x1, y2, occupied_bottom, direction=1)
                    draw_label(disp_img, label, (x1, min(h-5, y2 + offset + 20)), color)
                
            if results.masks is not None and show_clutter:
                for i, mask in enumerate(results.masks.xy):
                    cls_id = int(results.boxes.cls[i])
                    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                    pts = np.array(mask, np.int32)
                    cv2.polylines(disp_img, [pts], isClosed=True, color=color, thickness=2)
                    # Semi-transparent overlay
                    overlay = disp_img.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.3, disp_img, 0.7, 0, disp_img)

        # Overlay Info Text
        progress_text = f"[{idx + 1}/{len(image_paths)}] Mode: {mode_names[show_mode]} | Text: {'ON' if show_text else 'OFF'} | Clutter: {'ON' if show_clutter else 'OFF'}"
        cv2.putText(disp_img, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(disp_img, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(disp_img, os.path.basename(img_path), (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out_name = os.path.splitext(os.path.basename(img_path))[0] + "_annotated.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        ok = cv2.imwrite(out_path, disp_img)
        if ok:
            saved_count += 1
            print(f"Saved [{idx + 1}/{len(image_paths)}]: {out_path}")
        else:
            print(f"[WARN] Failed to save: {out_path}")

    print(f"\nDone. Saved {saved_count}/{len(image_paths)} annotated image(s) to: {OUTPUT_DIR}")
    print(f"Total detections found: {total_detections}")


def run_evaluation_and_viewer():
    """Backward-compatible alias for older launch commands."""
    run_evaluation_and_export()

if __name__ == "__main__":
    run_evaluation_and_export()