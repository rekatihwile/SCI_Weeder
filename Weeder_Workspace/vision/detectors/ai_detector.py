from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ultralytics import YOLO

from config import (
    BASE_DIR,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    AI_CONFIDENCE,
    AI_IOM_THRESHOLD,
    AI_BURST_SIZE,
    AI_MIN_STABLE_VIEWS,
)

# --- NEW MODEL ARCHITECTURE ---
class MeristemPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v3_small(weights=None).features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(self.encoder(x))


class _WeedCVCore:
    def __init__(self, yolo_path, qpoint_path=None, conf=AI_CONFIDENCE, iom_thresh=AI_IOM_THRESHOLD):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_path = Path(yolo_path)
        self.qpoint_path = Path(qpoint_path) if qpoint_path is not None else None

        # IMPORTANT: task='segment' must be passed!
        self.yolo = YOLO(str(self.yolo_path), task='segment')
        self.qpoint_model = None
        self.filtered_boxes = []

        self.conf = conf
        self.iom_thresh = iom_thresh
        self.TRAIN_SIZE = 224

        self._load_qpoint_model()

    def _load_qpoint_model(self):
        if self.qpoint_path is None or not self.qpoint_path.exists():
            print("[WARN] No qpoint model found. Falling back to box centers.")
            return

        try:
            self.qpoint_model = MeristemPredictor().to(self.device)
            self.qpoint_model.load_state_dict(
                torch.load(str(self.qpoint_path), map_location=self.device, weights_only=True)
            )
            self.qpoint_model.half().eval()
            print(f"[INFO] Using qpoint model: {self.qpoint_path}")
        except Exception as exc:
            print(f"[WARN] Could not load qpoint model ({self.qpoint_path}): {exc}")
            print("[WARN] Falling back to box centers.")
            self.qpoint_model = None

    def _iom(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        amin = min(a1, a2)
        return inter / float(amin) if amin > 0 else 0.0

    def _get_filtered_results(self, frame):
        # imgsz=1280 and retina_masks=True guarantees YOLO output matches the high-res frame perfectly
        results = self.yolo(frame, imgsz=1280, verbose=False, conf=self.conf, retina_masks=True)
        if not results or len(results[0].boxes) == 0 or results[0].masks is None:
            self.filtered_boxes = []
            return [], []

        raw_boxes = results[0].boxes
        raw_masks = results[0].masks.data.half()
        
        order = sorted(range(len(raw_boxes)), key=lambda i: float(raw_boxes[i].conf[0]), reverse=True)
        
        keep = []
        for i in order:
            bi = raw_boxes[i].xyxy[0].cpu().numpy()
            if all(self._iom(bi, raw_boxes[j].xyxy[0].cpu().numpy()) <= self.iom_thresh for j in keep):
                keep.append(i)
                
        self.filtered_boxes = [raw_boxes[i] for i in keep]
        filtered_masks = [raw_masks[i] for i in keep]
        return self.filtered_boxes, filtered_masks

    def detect_points(self, frame):
        boxes, masks = self._get_filtered_results(frame)
        if not boxes:
            return []

        if self.qpoint_model is None:
            return [(int((b.xyxy[0][0] + b.xyxy[0][2]) / 2), int((b.xyxy[0][1] + b.xyxy[0][3]) / 2)) for b in boxes]

        norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1).half()
        norm_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1).half()
        
        img_t = torch.from_numpy(frame).to(self.device)
        img_t = img_t[:, :, [2, 1, 0]].permute(2, 0, 1).half() / 255.0 
        img_t = (img_t - norm_mean) / norm_std 
        
        all_tensors, all_masks, all_metadata = [], [], []
        
        for i in range(len(boxes)):
            b = boxes[i].xyxy[0].int()
            x1, y1 = max(0, b[0].item()), max(0, b[1].item())
            x2, y2 = min(frame.shape[1], b[2].item()), min(frame.shape[0], b[3].item())
            
            if x2 <= x1 or y2 <= y1: continue
            
            crop_img = img_t[:, y1:y2, x1:x2].unsqueeze(0) 
            crop_mask = masks[i][y1:y2, x1:x2].unsqueeze(0).unsqueeze(0) 
            
            ch, cw = y2 - y1, x2 - x1
            scale = self.TRAIN_SIZE / max(ch, cw)
            nw, nh = int(cw * scale), int(ch * scale)
            dx, dy = (self.TRAIN_SIZE - nw) // 2, (self.TRAIN_SIZE - nh) // 2
            
            crop_img_res = F.interpolate(crop_img, size=(nh, nw), mode='bilinear', align_corners=False)
            crop_mask_res = F.interpolate(crop_mask, size=(nh, nw), mode='nearest')
            
            pad_left, pad_right = dx, self.TRAIN_SIZE - nw - dx
            pad_top, pad_bottom = dy, self.TRAIN_SIZE - nh - dy
            
            final_img = F.pad(crop_img_res, (pad_left, pad_right, pad_top, pad_bottom), value=0)
            final_mask = F.pad(crop_mask_res, (pad_left, pad_right, pad_top, pad_bottom), value=0)
            
            all_tensors.append(final_img.squeeze(0))
            all_masks.append(final_mask.squeeze(0).squeeze(0)) 
            all_metadata.append({'x1': x1, 'y1': y1, 'scale': scale, 'dx': dx, 'dy': dy})

        if not all_tensors: return []

        batch_t = torch.stack(all_tensors) 
        batch_m = torch.stack(all_masks)   
        
        with torch.no_grad():
            heatmaps = self.qpoint_model(batch_t).squeeze(1) 
            
        masked_heatmaps = heatmaps * batch_m 
        masked_heatmaps_cpu = masked_heatmaps.cpu().float().numpy()
            
        coords = []
        for i, meta in enumerate(all_metadata):
            masked = masked_heatmaps_cpu[i]
            
            # Use minMaxLoc to strictly target the hottest plant tissue pixel
            _, max_conf, _, max_loc = cv2.minMaxLoc(masked)
            
            if max_conf < 0.05:
                lx, ly = float(self.TRAIN_SIZE / 2.0), float(self.TRAIN_SIZE / 2.0)
            else:
                lx, ly = float(max_loc[0]), float(max_loc[1])
                
            gx = int((lx - meta['dx']) / meta['scale']) + meta['x1']
            gy = int((ly - meta['dy']) / meta['scale']) + meta['y1']
            
            gx = max(0, min(frame.shape[1] - 1, gx))
            gy = max(0, min(frame.shape[0] - 1, gy))
            coords.append((gx, gy))

        return coords

    def return_burst_stable(self, frames, min_stable_views=AI_MIN_STABLE_VIEWS, group_radius_px=30):
        all_detections = []

        for frame in frames:
            coords = self.detect_points(frame)
            frame_data = []

            for i, box in enumerate(self.filtered_boxes):
                if i >= len(coords):
                    continue
                b = box.xyxy[0].cpu().numpy()
                cx = 0.5 * (b[0] + b[2])
                cy = 0.5 * (b[1] + b[3])
                frame_data.append({
                    "center": np.array([cx, cy], dtype=float),
                    "point": coords[i],
                })

            all_detections.append(frame_data)

        groups = []

        for frame_data in all_detections:
            for det in frame_data:
                best_group = None
                best_dist = float("inf")

                for gi, g in enumerate(groups):
                    d = np.linalg.norm(det["center"] - g["center_mean"])
                    if d < group_radius_px and d < best_dist:
                        best_dist = d
                        best_group = gi

                if best_group is None:
                    groups.append({
                        "centers": [det["center"]],
                        "points": [det["point"]],
                        "views": 1,
                        "center_mean": det["center"].copy(),
                    })
                else:
                    g = groups[best_group]
                    g["centers"].append(det["center"])
                    g["points"].append(det["point"])
                    g["views"] += 1
                    g["center_mean"] = np.mean(g["centers"], axis=0)

        stable_points = []
        for g in groups:
            if g["views"] >= min_stable_views:
                pts = np.array(g["points"], dtype=float)
                mean_pt = np.mean(pts, axis=0)
                stable_points.append((int(round(mean_pt[0])), int(round(mean_pt[1]))))

        stable_points.sort(key=lambda p: p[0])
        return stable_points


class AIDetector:
    def __init__(
        self,
        display_scale=1.5,
        burst_size=AI_BURST_SIZE,
        min_stable_views=AI_MIN_STABLE_VIEWS,
        yolo_path=None,
        qpoint_path=None,
        conf=AI_CONFIDENCE,
        iom_thresh=AI_IOM_THRESHOLD,
    ):
        params_dir = BASE_DIR / "params"

        self.yolo_path = Path(yolo_path) if yolo_path is not None else params_dir / "best_pigweed_145.pt"
        if not self.yolo_path.is_absolute():
            self.yolo_path = params_dir / self.yolo_path

        # Updated to default to your new V3 model!
        self.qpoint_path = Path(qpoint_path) if qpoint_path is not None else params_dir / "new_best_targeting_v3.pth"
        if not self.qpoint_path.is_absolute():
            self.qpoint_path = params_dir / self.qpoint_path

        self.display_scale = display_scale
        self.burst_size = burst_size
        self.min_stable_views = min_stable_views
        self.window_name = "AI Detector - Stereo Pair"

        self.cv_left = _WeedCVCore(
            self.yolo_path,
            self.qpoint_path,
            conf=conf,
            iom_thresh=iom_thresh,
        )
        self.cv_right = _WeedCVCore(
            self.yolo_path,
            self.qpoint_path,
            conf=conf,
            iom_thresh=iom_thresh,
        )

    def refine_live(self, cameras):
        left_points, right_points = self.detect_live(cameras)

        if not left_points or not right_points:
            return None, None

        target_xy = (FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0)

        def dist2(p):
            dx = p[0] - target_xy[0]
            dy = p[1] - target_xy[1]
            return dx * dx + dy * dy

        left_pt = min(left_points, key=dist2)
        right_pt = min(right_points, key=dist2)

        return left_pt, right_pt

    def _collect_burst(self, cameras):
        left_frames = []
        right_frames = []
        for _ in range(self.burst_size):
            left_frame, right_frame = cameras.read_pair()
            left_frames.append(left_frame)
            right_frames.append(right_frame)
        return left_frames, right_frames

    def _draw_panel(self, left_frame, right_frame, left_points, right_points):
        h, w = left_frame.shape[:2]
        disp_w = int(w * self.display_scale)
        disp_h = int(h * self.display_scale)

        left_disp = cv2.resize(left_frame.copy(), (disp_w, disp_h))
        right_disp = cv2.resize(right_frame.copy(), (disp_w, disp_h))

        sx = self.display_scale
        sy = self.display_scale

        for pt in left_points:
            x, y = int(pt[0] * sx), int(pt[1] * sy)
            cv2.circle(left_disp, (x, y), 5, (0, 0, 255), -1)

        for pt in right_points:
            x, y = int(pt[0] * sx), int(pt[1] * sy)
            cv2.circle(right_disp, (x, y), 5, (0, 0, 255), -1)

        panel = np.hstack([left_disp, right_disp])
        return panel

    def detect_live(self, cameras):
        left_frame, right_frame = cameras.read_pair()
        left_points = self.cv_left.detect_points(left_frame)
        right_points = self.cv_right.detect_points(right_frame)
        return left_points, right_points

    def detect_stable_live(self, cameras):
        left_frames, right_frames = self._collect_burst(cameras)
        stable_left = self.cv_left.return_burst_stable(
            left_frames,
            min_stable_views=self.min_stable_views,
        )
        stable_right = self.cv_right.return_burst_stable(
            right_frames,
            min_stable_views=self.min_stable_views,
        )
        return stable_left, stable_right

    def show_live(self, cameras):
        while True:
            left_frame, right_frame = cameras.read_pair()
            left_points = self.cv_left.detect_points(left_frame)
            right_points = self.cv_right.detect_points(right_frame)

            panel = self._draw_panel(left_frame, right_frame, left_points, right_points)
            cv2.imshow(self.window_name, panel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyWindow(self.window_name)