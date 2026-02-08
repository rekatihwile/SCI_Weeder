import cv2
import numpy as np
import torch
from ultralytics import YOLO

class WeedCV:
    def __init__(self, yolo_path, mobilenet_path, conf=0.60, iom_thresh=0.80):
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

    def return_full(self, frame):
        boxes, masks = self._get_detections(frame)
        coords = []
        # Fast Mask Logic
        m_out = np.zeros(frame.shape[:2], dtype=np.uint8)
        if masks is not None:
            for i in range(len(self.filtered_boxes)):
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