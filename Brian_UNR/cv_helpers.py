import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

class WeedCV:
    def __init__(self, yolo_path, mobilenet_path, conf=0.30, iom_thresh=0.80):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO(yolo_path)
        self.keypoint_model = torch.jit.load(mobilenet_path, map_location=self.device).eval()
        if self.device == 'cuda': self.keypoint_model.cuda()
        
        self.conf, self.iom_thresh = conf, iom_thresh
        self.input_size = 640 
        self.filtered_boxes = []
        self.win_main = "AI Targeter"

    def calculate_iom(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1, a2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / float(min(a1, a2)) if min(a1, a2) > 0 else 0

    def _get_detections(self, frame):
        results = self.yolo(frame, verbose=False, conf=self.conf)
        if not results or not results[0].boxes:
            self.filtered_boxes = []
            return [], None
        raw = results[0].boxes
        idx = sorted(range(len(raw)), key=lambda i: raw[i].conf, reverse=True)
        final = []
        for i in idx:
            bi = raw[i].xyxy[0].cpu().numpy()
            if all(self.calculate_iom(bi, raw[j].xyxy[0].cpu().numpy()) <= self.iom_thresh for j in final):
                final.append(i)
        self.filtered_boxes = [raw[i] for i in final]
        return self.filtered_boxes, results[0].masks

    def return_full(self, frame, precalc_boxes=None, precalc_masks=None):
        # Only run YOLO if we didn't already pass in the boxes
        if precalc_boxes is None:
            boxes, masks = self._get_detections(frame)
        else:
            boxes, masks = precalc_boxes, precalc_masks
            
        coords = []
        if not boxes: return coords

        # Fast Mask Logic
        m_out = np.zeros(frame.shape[:2], dtype=np.uint8)
        if masks is not None:
            for i in range(len(boxes)):
                m = cv2.resize(masks[i].data[0].cpu().numpy(), (frame.shape[1], frame.shape[0]))
                m_out = cv2.bitwise_or(m_out, (m * 255).astype(np.uint8))
        else:
            for b in boxes:
                coords_b = b.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(m_out, (coords_b[0], coords_b[1]), (coords_b[2], coords_b[3]), 255, -1)
        
        m_frame = cv2.bitwise_and(frame, frame, mask=m_out)
        for box in boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            # Sniper Padding 20%
            w, h = b[2]-b[0], b[3]-b[1]
            px, py = int(w*0.2), int(h*0.2)
            x1, y1, x2, y2 = max(0,b[0]-px), max(0,b[1]-py), min(frame.shape[1],b[2]+px), min(frame.shape[0],b[3]+py)
            crop = m_frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # Keypoint Inference
            img = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
            img_t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).to(self.device)/255.0
            with torch.no_grad():
                out = self.keypoint_model(img_t).cpu().numpy().flatten()
            coords.append((int(x1 + out[0]*(x2-x1)), int(y1 + out[1]*(y2-y1))))
        return coords

    def show_debug(self, frame, coords):
        cv2.namedWindow(self.win_main, cv2.WINDOW_NORMAL)
        canvas = frame.copy()
        for box in self.filtered_boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), (255, 100, 0), 2)
        for (x, y) in coords:
            cv2.circle(canvas, (x, y), 10, (0, 0, 255), -1)
            cv2.drawMarker(canvas, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        
        # HUD for Weed Count
        cv2.rectangle(canvas, (0, 0), (220, 50), (0, 255, 0) if coords else (0, 0, 255), -1)
        cv2.putText(canvas, f"WEEDS: {len(coords)}", (15, 35), 0, 1.0, (255, 255, 255), 3)
        cv2.resizeWindow(self.win_main, 800, int(canvas.shape[0] * (800 / canvas.shape[1])))
        cv2.imshow(self.win_main, canvas)


    #this is new burst funtion that works
    def calculate_burst_math(self, accumulated_detections):
        """
        Takes a rolling history of pre-processed frame data and finds the stable medians.
        accumulated_detections: List of lists containing dicts: [{'box': [x1,y1,x2,y2], 'point': (x,y)}]
        """
        plant_groups = [] 

        # 1. Group by spatial proximity (Box Centers)
        for frame_data in accumulated_detections:
            for det in frame_data:
                cx, cy = (det['box'][0] + det['box'][2])/2, (det['box'][1] + det['box'][3])/2
                
                match_found = False
                for group in plant_groups:
                    group_center = np.mean([p['center'] for p in group], axis=0)
                    if np.linalg.norm(np.array([cx, cy]) - group_center) < 30: # 30px radius
                        group.append({'center': [cx, cy], 'point': det['point']})
                        match_found = True
                        break
                
                if not match_found:
                    plant_groups.append([{'center': [cx, cy], 'point': det['point']}])

        # 2. Filter and Average
        stable_coords = []
        for group in plant_groups:
            # Must appear in at least 1 frame (Increase this to 2 or 3 for stricter filtering)
            if len(group) >= 1: 
                pts = np.array([g['point'] for g in group])
                median_x = int(np.median(pts[:, 0]))
                median_y = int(np.median(pts[:, 1]))
                stable_coords.append((median_x, median_y))

        return stable_coords

    
    
    def _get_detections(self, frame, weed_classes=[0, 2], kale_class=1, kale_thresh=0.05, weed_conf=0.50, kale_conf=0.7):
        # 1. Ask YOLO for everything above the absolute minimum threshold
        base_conf = min(weed_conf, kale_conf)
        results = self.yolo(frame, verbose=False, conf=base_conf)
        
        if not results or not results[0].boxes:
            self.filtered_boxes = []
            self.kale_boxes = []
            return [], None

        raw_boxes = results[0].boxes
        raw_masks = results[0].masks

        # 2. Separate by class AND apply independent confidence thresholds
        weed_idx, kale_idx = [], []
        for i, box in enumerate(raw_boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if cls_id in weed_classes and conf >= weed_conf:
                weed_idx.append(i)
            elif cls_id == kale_class and conf >= kale_conf:
                kale_idx.append(i)

        self.kale_boxes = [raw_boxes[i] for i in kale_idx]

        # 3. Class-blind merge for weeds
        weed_idx = sorted(weed_idx, key=lambda i: raw_boxes[i].conf, reverse=True)
        merged_weeds = []
        for i in weed_idx:
            bi = raw_boxes[i].xyxy[0].cpu().numpy()
            if all(self.calculate_iom(bi, raw_boxes[j].xyxy[0].cpu().numpy()) <= self.iom_thresh for j in merged_weeds):
                merged_weeds.append(i)

        self.raw_weed_boxes = [raw_boxes[i] for i in merged_weeds]

        # 4. The "No-Strike" Filter
        safe_weeds = []
        for w_idx in merged_weeds:
            bw = raw_boxes[w_idx].xyxy[0].cpu().numpy()
            is_safe = True
            for k_idx in kale_idx:
                bk = raw_boxes[k_idx].xyxy[0].cpu().numpy()
                if self.calculate_iom(bw, bk) > kale_thresh:
                    is_safe = False
                    break 
            
            if is_safe:
                safe_weeds.append(w_idx)

        self.filtered_boxes = [raw_boxes[i] for i in safe_weeds]

        filtered_masks = None
        if raw_masks is not None and len(safe_weeds) > 0:
            filtered_masks = [raw_masks[i] for i in safe_weeds]

        return self.filtered_boxes, filtered_masks
    

    #ths is old burst we don't use anymore but keeping it here for reference
    def return_burst_stable(self, frames):
        """
        frames: List of 5 captured images.
        Returns: List of (x, y) coordinates that survived the median filter.
        """
        all_detections = [] # List of dicts: {'box': [x1,y1,x2,y2], 'point': (x,y)}
        
        # 1. Collect all raw detections from the 5 frames
        for frame in frames:
            # 1. Run YOLO exactly ONCE
            boxes, masks = self._get_detections(frame)
            
            # 2. Pass the results directly into return_full so it skips YOLO
            frame_coords = self.return_full(frame, precalc_boxes=boxes, precalc_masks=masks) 
            
            frame_data = []
            for i, box in enumerate(self.filtered_boxes):
                if i < len(frame_coords):
                    frame_data.append({
                        'box': box.xyxy[0].cpu().numpy(),
                        'point': frame_coords[i]
                    })
            all_detections.append(frame_data)

        # 2. Group detections across frames by Box Center proximity
        # (Since the camera is stationary, a plant won't move more than a few pixels)
        plant_groups = [] # List of lists of points

        for frame_data in all_detections:
            for det in frame_data:
                cx, cy = (det['box'][0] + det['box'][2])/2, (det['box'][1] + det['box'][3])/2
                
                match_found = False
                for group in plant_groups:
                    # Check if this detection is close to an existing group center
                    group_center = np.mean([p['center'] for p in group], axis=0)
                    if np.linalg.norm(np.array([cx, cy]) - group_center) < 30: # 30px radius
                        group.append({'center': [cx, cy], 'point': det['point']})
                        match_found = True
                        break
                
                if not match_found:
                    plant_groups.append([{'center': [cx, cy], 'point': det['point']}])

        # 3. Filter and Average
        stable_coords = []
        for group in plant_groups:
            # Only keep plants seen 
            if len(group) >= 1:
                pts = np.array([g['point'] for g in group])
                
                # Take Median of X and Y separately (Discarding Min/Max outliers)
                median_x = int(np.median(pts[:, 0]))
                median_y = int(np.median(pts[:, 1]))
                stable_coords.append((median_x, median_y))

        return stable_coords