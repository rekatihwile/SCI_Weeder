from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import BASE_DIR, FRAME_WIDTH, FRAME_HEIGHT


class _WeedCVCore:
    def __init__(self, yolo_path, qpoint_path=None, conf=0.60, iom_thresh=0.80):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO(str(yolo_path))
        self.qpoint_model = None

        if qpoint_path is not None and Path(qpoint_path).exists():
            self.qpoint_model = torch.jit.load(str(qpoint_path), map_location=self.device).eval()
            if self.device == "cuda":
                self.qpoint_model.cuda()

        self.conf = conf
        self.iom_thresh = iom_thresh
        self.filtered_boxes = []

    def _iom(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        amin = min(a1, a2)
        if amin <= 0:
            return 0.0
        return inter / float(amin)

    def _get_filtered_boxes(self, frame):
        results = self.yolo(frame, verbose=False, conf=self.conf)
        if not results or len(results[0].boxes) == 0:
            self.filtered_boxes = []
            return []

        raw_boxes = results[0].boxes
        order = sorted(range(len(raw_boxes)), key=lambda i: float(raw_boxes[i].conf[0]), reverse=True)

        keep = []
        for i in order:
            bi = raw_boxes[i].xyxy[0].cpu().numpy()
            if all(self._iom(bi, raw_boxes[j].xyxy[0].cpu().numpy()) <= self.iom_thresh for j in keep):
                keep.append(i)

        self.filtered_boxes = [raw_boxes[i] for i in keep]
        return self.filtered_boxes

    def _predict_point_in_crop(self, crop):
        if self.qpoint_model is None:
            h, w = crop.shape[:2]
            return 0.5 * w, 0.5 * h

        img = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            out = self.qpoint_model(img_t).detach().cpu().numpy().flatten()

        return float(out[0]) * crop.shape[1], float(out[1]) * crop.shape[0]

    def detect_points(self, frame):
        boxes = self._get_filtered_boxes(frame)
        coords = []

        for box in boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = b.tolist()

            w = x2 - x1
            h = y2 - y1
            px = int(0.2 * w)
            py = int(0.2 * h)

            cx1 = max(0, x1 - px)
            cy1 = max(0, y1 - py)
            cx2 = min(frame.shape[1], x2 + px)
            cy2 = min(frame.shape[0], y2 + py)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            dx, dy = self._predict_point_in_crop(crop)
            x = int(round(cx1 + dx))
            y = int(round(cy1 + dy))

            x = max(0, min(frame.shape[1] - 1, x))
            y = max(0, min(frame.shape[0] - 1, y))
            coords.append((x, y))

        return coords

    def return_burst_stable(self, frames, min_stable_views=3, group_radius_px=30):
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
                matched = False
                for group in groups:
                    group_center = np.mean([item["center"] for item in group], axis=0)
                    if np.linalg.norm(det["center"] - group_center) < group_radius_px:
                        group.append(det)
                        matched = True
                        break
                if not matched:
                    groups.append([det])

        stable = []
        for group in groups:
            if len(group) < min_stable_views:
                continue
            pts = np.array([item["point"] for item in group], dtype=float)
            stable.append((int(np.median(pts[:, 0])), int(np.median(pts[:, 1]))))

        stable.sort(key=lambda p: (p[1], p[0]))
        return stable


class AIDetector:
    def __init__(
        self,
        display_scale=1.5,
        burst_size=5,
        min_stable_views=3,
        yolo_path=None,
        qpoint_path=None,
        conf=0.60,
        iom_thresh=0.80,
    ):
        params_dir = BASE_DIR / "params"
        self.yolo_path = Path(yolo_path) if yolo_path is not None else params_dir / "yolo_w_kale.pt"
        self.qpoint_path = Path(qpoint_path) if qpoint_path is not None else params_dir / "sniper.pt"

        self.display_scale = display_scale
        self.burst_size = burst_size
        self.min_stable_views = min_stable_views
        self.window_name = "AI Detector - Stereo Pair"

        self.cv_left = _WeedCVCore(self.yolo_path, self.qpoint_path, conf=conf, iom_thresh=iom_thresh)
        self.cv_right = _WeedCVCore(self.yolo_path, self.qpoint_path, conf=conf, iom_thresh=iom_thresh)


    
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
        sx = disp_w / w
        sy = disp_h / h

        left_disp = cv2.resize(left_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        right_disp = cv2.resize(right_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.hstack((left_disp, right_disp))

        cv2.line(canvas, (disp_w, 0), (disp_w, disp_h), (255, 255, 255), 2)
        cv2.putText(canvas, "LEFT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(canvas, "RIGHT", (disp_w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        for i, (x, y) in enumerate(left_points, start=1):
            xd = int(x * sx)
            yd = int(y * sy)
            cv2.circle(canvas, (xd, yd), 7, (0, 0, 255), -1)
            cv2.putText(canvas, str(i), (xd + 8, yd - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for i, (x, y) in enumerate(right_points, start=1):
            xd = int(x * sx) + disp_w
            yd = int(y * sy)
            cv2.circle(canvas, (xd, yd), 7, (0, 255, 0), -1)
            cv2.putText(canvas, str(i), (xd + 8, yd - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        mode_text = "YOLO + QPOINT" if self.qpoint_path.exists() else "YOLO ONLY (sniper.pt missing)"
        status_1 = f"L points: {len(left_points)}    R points: {len(right_points)}"
        status_2 = f"{mode_text}    Enter = accept | r = rescan | q = quit"

        cv2.rectangle(canvas, (0, disp_h - 70), (canvas.shape[1], disp_h), (0, 0, 0), -1)
        cv2.putText(canvas, status_1, (20, disp_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, status_2, (20, disp_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return canvas

    def detect_live(self, cameras):
        print("\n=== AI DETECT ===")
        if self.qpoint_path.exists():
            print(f"Using YOLO:   {self.yolo_path}")
            print(f"Using Qpoint: {self.qpoint_path}")
        else:
            print(f"Using YOLO:   {self.yolo_path}")
            print("Qpoint model not found. Falling back to box centers.")
        print("Enter = accept | r = rescan | q = quit")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while True:
            left_frames, right_frames = self._collect_burst(cameras)

            left_points = self.cv_left.return_burst_stable(
                left_frames,
                min_stable_views=self.min_stable_views,
            )
            right_points = self.cv_right.return_burst_stable(
                right_frames,
                min_stable_views=self.min_stable_views,
            )

            canvas = self._draw_panel(left_frames[-1], right_frames[-1], left_points, right_points)
            cv2.imshow(self.window_name, canvas)

            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10):
                cv2.destroyWindow(self.window_name)
                return left_points, right_points
            if key == ord("r"):
                continue
            if key == ord("q"):
                cv2.destroyWindow(self.window_name)
                return [], []
