from pathlib import Path

import cv2
import numpy as np
import torch
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


class _WeedCVCore:
    def __init__(self, yolo_path, qpoint_path=None, conf=AI_CONFIDENCE, iom_thresh=AI_IOM_THRESHOLD):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_path = Path(yolo_path)
        self.qpoint_path = Path(qpoint_path) if qpoint_path is not None else None

        self.yolo = YOLO(str(self.yolo_path))
        self.qpoint_model = None
        self.filtered_boxes = []
        self.qpoint_is_heatmap = False  # updated by _probe_output_is_heatmap after load

        self.conf = conf
        self.iom_thresh = iom_thresh

        self._load_qpoint_model()

    def _load_qpoint_model(self):
        if self.qpoint_path is None or not self.qpoint_path.exists():
            print("[WARN] No qpoint model found. Falling back to box centers.")
            return

        name = self.qpoint_path.name

        # --- Strategy 1: TorchScript (torch.jit.save / torch.jit.trace) ---
        try:
            model = torch.jit.load(str(self.qpoint_path), map_location=self.device).eval()
            if self.device == "cuda":
                model.cuda()
            self.qpoint_model = model
            self.qpoint_is_heatmap = False
            print(f"[INFO] Loaded qpoint model (TorchScript): {name}")
            return
        except Exception:
            pass   # not a TorchScript file — try next strategy

        # --- Strategy 2: Full model object (torch.save(model, path)) ---
        try:
            # weights_only=False required to unpickle the full model object.
            # Suppress the FutureWarning about weights_only on older PyTorch.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = torch.load(str(self.qpoint_path),
                                   map_location=self.device,
                                   weights_only=False)
            if hasattr(model, "eval"):
                model = model.eval()
            if self.device == "cuda" and hasattr(model, "cuda"):
                model = model.cuda()
            self.qpoint_model = model
            # Probe output shape on a dummy input to decide coord vs heatmap
            self.qpoint_is_heatmap = self._probe_output_is_heatmap()
            mode = "heatmap" if self.qpoint_is_heatmap else "coord"
            print(f"[INFO] Loaded qpoint model (torch.load, {mode}): {name}")
            return
        except Exception as exc:
            print(f"[WARN] Could not load qpoint model ({name}): {exc}")
            print("[WARN] Falling back to box centers.")
            self.qpoint_model = None

    def _probe_output_is_heatmap(self) -> bool:
        """
        Run a blank 640×640 input through the model and inspect the output
        shape.  If the output has spatial dimensions (H×W > 4) we treat it
        as a heatmap; otherwise we assume it returns normalised (x, y)
        coordinates.
        """
        try:
            dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.qpoint_model(dummy)
            # Any dimension > 4 values is almost certainly spatial
            return any(d > 4 for d in out.shape)
        except Exception:
            return False   # can't tell — default to coord mode

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
        order = sorted(
            range(len(raw_boxes)),
            key=lambda i: float(raw_boxes[i].conf[0]),
            reverse=True,
        )

        keep = []
        for i in order:
            bi = raw_boxes[i].xyxy[0].cpu().numpy()
            if all(self._iom(bi, raw_boxes[j].xyxy[0].cpu().numpy()) <= self.iom_thresh for j in keep):
                keep.append(i)

        self.filtered_boxes = [raw_boxes[i] for i in keep]
        return self.filtered_boxes

    def _prep_crop_input(self, crop):
        img = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img).to(self.device)

    def _predict_point_in_crop(self, crop):
        h, w = crop.shape[:2]

        if self.qpoint_model is None:
            return 0.5 * w, 0.5 * h

        img_t = self._prep_crop_input(crop)

        with torch.no_grad():
            raw = self.qpoint_model(img_t)

        out = raw.detach().cpu()

        # ----------------------------------------------------------------
        # Heatmap output  (e.g. shape 1×1×H×W or 1×H×W)
        # The predicted point is the argmax of the spatial heatmap, then
        # mapped back to crop pixel coordinates.
        # ----------------------------------------------------------------
        if self.qpoint_is_heatmap or out.dim() >= 3:
            hm = out.squeeze()           # drop batch + channel dims → (H, W)
            if hm.dim() == 3:            # still (C, H, W) — collapse channels
                hm = hm.max(dim=0).values
            if hm.dim() != 2:
                return 0.5 * w, 0.5 * h
            hm_np = hm.numpy()
            flat_idx         = int(np.argmax(hm_np))
            peak_y, peak_x   = np.unravel_index(flat_idx, hm_np.shape)
            return (float(peak_x) / hm_np.shape[1] * w,
                    float(peak_y) / hm_np.shape[0] * h)

        # ----------------------------------------------------------------
        # Coordinate output  (flat vector [x_norm, y_norm])
        # ----------------------------------------------------------------
        flat = out.numpy().reshape(-1)
        if flat.size < 2:
            return 0.5 * w, 0.5 * h

        return float(flat[0]) * w, float(flat[1]) * h

    def detect_with_visuals(self, frame):
        """
        Run detection and return bounding boxes + keypoints together.

        Returns a list of dicts, one per detected plant:
            {
                "box":      (x1, y1, x2, y2),   # YOLO bbox in frame pixels
                "conf":     float,               # YOLO confidence score
                "keypoint": (x, y),              # Predicted stem point in frame pixels
            }
        """
        boxes = self._get_filtered_boxes(frame)
        results = []

        for box in boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = b.tolist()
            conf = float(box.conf[0])

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
            kx = int(round(cx1 + dx))
            ky = int(round(cy1 + dy))
            kx = max(0, min(frame.shape[1] - 1, kx))
            ky = max(0, min(frame.shape[0] - 1, ky))

            cls_id   = int(box.cls[0])
            cls_name = self.yolo.names.get(cls_id, str(cls_id))

            results.append({
                "box":      (x1, y1, x2, y2),
                "conf":     conf,
                "keypoint": (kx, ky),
                "cls_id":   cls_id,
                "cls_name": cls_name,
            })

        return results

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

        self.qpoint_path = Path(qpoint_path) if qpoint_path is not None else params_dir / "sniper.pt"
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