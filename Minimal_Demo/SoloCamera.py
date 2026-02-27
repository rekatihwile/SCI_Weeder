import cv2
import numpy as np
import torch
from ultralytics import YOLO

class ClassAwareWeedCV:
    def __init__(self, yolo_path, mobilenet_path, conf=0.50, safe_iom_thresh=0.20):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading models on: {self.device}")
        
        self.yolo = YOLO(yolo_path)
        
        # Load Sniper Keypoint Model
        self.keypoint_model = torch.jit.load(mobilenet_path, map_location=self.device).eval()
        if self.device == 'cuda': 
            self.keypoint_model.cuda()
        
        self.conf = conf
        self.safe_iom_thresh = safe_iom_thresh
        
        # UPDATE THESE IDs TO MATCH YOUR ROBOFLOW DATASET
        self.weed_classes = [0, 2] # e.g., 0: Dandelion, 1: Lamb's Quarters
        self.crop_classes = [1]    # e.g., 2: Kale

        self.valid_weeds = []
        self.protected_crops = []
        self.win_main = "SCI_Weeder Live Targeting"

    def calculate_iom(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1, a2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / float(min(a1, a2)) if min(a1, a2) > 0 else 0

    def process_frame(self, frame):
        results = self.yolo(frame, verbose=False, conf=self.conf)
        
        self.valid_weeds = []
        self.protected_crops = []
        weed_coords = []

        if not results or not results[0].boxes:
            return weed_coords

        raw_boxes = results[0].boxes
        
        # 1. Separate detections into Weeds and Crops
        temp_weeds = []
        for i in range(len(raw_boxes)):
            cls_id = int(raw_boxes[i].cls[0].item())
            if cls_id in self.crop_classes:
                self.protected_crops.append(raw_boxes[i])
            elif cls_id in self.weed_classes:
                temp_weeds.append(raw_boxes[i])

        # 2. Cross-Class Exclusion Filter
        for weed in temp_weeds:
            w_box = weed.xyxy[0].cpu().numpy()
            is_safe_to_shoot = True
            
            for crop in self.protected_crops:
                c_box = crop.xyxy[0].cpu().numpy()
                overlap = self.calculate_iom(w_box, c_box)
                
                if overlap > self.safe_iom_thresh:
                    is_safe_to_shoot = False
                    break # Unsafe, skip checking other crops
            
            if is_safe_to_shoot:
                self.valid_weeds.append(weed)

        # 3. Process Keypoints ONLY for safe weeds
        for box in self.valid_weeds:
            b = box.xyxy[0].cpu().numpy().astype(int)
            w, h = b[2]-b[0], b[3]-b[1]
            px, py = int(w*0.2), int(h*0.2)
            x1, y1, x2, y2 = max(0,b[0]-px), max(0,b[1]-py), min(frame.shape[1],b[2]+px), min(frame.shape[0],b[3]+py)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # Keypoint Inference
            img = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
            img_t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).to(self.device)/255.0
            
            with torch.no_grad():
                out = self.keypoint_model(img_t).cpu().numpy().flatten()
            
            weed_coords.append((int(x1 + out[0]*(x2-x1)), int(y1 + out[1]*(y2-y1))))

        return weed_coords

    def render_debug(self, frame, coords):
        canvas = frame.copy()
        
        # Draw Protected Crops (Green)
        for crop in self.protected_crops:
            b = crop.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(canvas, "SAFE CROP", (b[0], max(0, b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Target Weeds (Red with Crosshairs)
        for weed in self.valid_weeds:
            b = weed.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

        for (x, y) in coords:
            cv2.circle(canvas, (x, y), 8, (0, 0, 255), -1)
            cv2.drawMarker(canvas, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        
        # HUD
        cv2.rectangle(canvas, (0, 0), (300, 70), (0, 0, 0), -1)
        cv2.putText(canvas, f"TARGET WEEDS: {len(self.valid_weeds)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(canvas, f"PROTECTED KALE: {len(self.protected_crops)}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.win_main, canvas)

if __name__ == "__main__":
    # Initialize the tester (Update paths to your weights)
    vision_system = ClassAwareWeedCV(yolo_path="weights/yolo_w_kale.pt", mobilenet_path="weights/sniper.pt")
    
    # Use 0 for default webcam, or provide an image/video path
    cap = cv2.VideoCapture(0)
    
    print("Starting Live Feed. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
            
        # 1. Process the frame through the new logic
        target_coords = vision_system.process_frame(frame)
        
        # 2. Render the results live
        vision_system.render_debug(frame, target_coords)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()