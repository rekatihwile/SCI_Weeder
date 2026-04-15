from pathlib import Path
from collections import Counter
import warnings
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ultralytics import YOLO

# Suppress PyTorch internals deprecation noise that isn't actionable.
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

from config import (
    BASE_DIR,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    AI_CONFIDENCE,
    AI_IOM_THRESHOLD,
    AI_BURST_SIZE,
    AI_MIN_STABLE_VIEWS,
    AI_TARGET_CLASS,
    DEFAULT_QPOINT_MODEL,
    MODEL_MAP,
    QPOINT_DEBUG,
)


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

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _box_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


def _resolve_classes(target_class, override=None):
    spec = override if override is not None else target_class
    if spec is None:
        return None
    if isinstance(spec, int):
        return [spec]
    return list(spec)


class _WeedCVCore:
    def __init__(self, yolo_path, qpoint_path=None, conf=AI_CONFIDENCE,
                 iom_thresh=AI_IOM_THRESHOLD, target_class=AI_TARGET_CLASS,
                 verbose=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_path = Path(yolo_path)
        self.qpoint_path = Path(qpoint_path) if qpoint_path is not None else None
        self.verbose = verbose

        self.target_class = target_class

        self.yolo = YOLO(str(self.yolo_path), task='segment')

        if verbose:
            if self.yolo.names:
                print(f"[INFO] Model classes: {self.yolo.names}")
            resolved = _resolve_classes(self.target_class)
            if resolved is not None:
                names = [self.yolo.names.get(c, "???") for c in resolved]
                print(f"[CV] Filtering to class(es) {resolved} = {names}")
            else:
                print("[CV] No class filter — detecting all classes.")

        self.qpoint_model = None
        self.filtered_boxes = []

        self.conf = conf
        self.iom_thresh = iom_thresh
        self.TRAIN_SIZE = 224

        self._load_qpoint_model()

    def _load_qpoint_model(self):
        if self.qpoint_path is None or not self.qpoint_path.exists():
            if self.verbose:
                print("[WARN] No qpoint model found. Falling back to box centers.")
            return

        try:
            self.qpoint_model = MeristemPredictor().to(self.device)
            self.qpoint_model.load_state_dict(
                torch.load(str(self.qpoint_path), map_location=self.device, weights_only=True)
            )
            self.qpoint_model.half().eval()
            if self.verbose:
                print(f"[INFO] Using qpoint model: {self.qpoint_path.name}")
        except Exception as exc:
            if self.verbose:
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

    def _extract_qpoint(self, heatmap_224, mask_224):
        heat = heatmap_224.astype(np.float32)
        mask = (mask_224 > 0.5).astype(np.uint8)

        if mask.sum() == 0:
            return None, 0.0

        masked = heat * mask
        masked = cv2.GaussianBlur(masked, (0, 0), 1.2)
        # Re-apply after blur: Gaussian spreads values across the mask boundary.
        # Without this, minMaxLoc and the weighted centroid can land outside
        # the segmented plant region.
        masked = masked * mask

        _, peak_conf, _, peak_loc = cv2.minMaxLoc(masked)

        if peak_conf <= 1e-6:
            Mmask = cv2.moments(mask.astype(np.float32))
            if Mmask["m00"] > 1e-6:
                x = Mmask["m10"] / Mmask["m00"]
                y = Mmask["m01"] / Mmask["m00"]
                return (float(x), float(y)), 0.0
            return None, 0.0

        thresh = max(0.25 * peak_conf, 0.08)
        hot = (masked >= thresh).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(hot)
        peak_x, peak_y = peak_loc

        if num_labels > 1:
            peak_label = labels[peak_y, peak_x]
            if peak_label != 0:
                hot = (labels == peak_label).astype(np.uint8)

        weighted = masked * hot

        M = cv2.moments(weighted)
        if M["m00"] > 1e-6:
            x = M["m10"] / M["m00"]
            y = M["m01"] / M["m00"]
            return (float(x), float(y)), float(peak_conf)

        Mmask = cv2.moments(mask.astype(np.float32))
        if Mmask["m00"] > 1e-6:
            x = Mmask["m10"] / Mmask["m00"]
            y = Mmask["m01"] / Mmask["m00"]
            return (float(x), float(y)), float(peak_conf)

        return (float(peak_x), float(peak_y)), float(peak_conf)

    def _get_filtered_results(self, frame, classes_override=None):
        classes_arg = _resolve_classes(self.target_class, classes_override)

        results = self.yolo(
            frame,
            imgsz=1280,
            verbose=False,
            conf=self.conf,
            retina_masks=True,
            classes=classes_arg,
        )

        if not results or len(results[0].boxes) == 0 or results[0].masks is None:
            self.filtered_boxes = []
            return [], []

        raw_boxes = results[0].boxes
        raw_masks = results[0].masks.data.half()

        order = sorted(
            range(len(raw_boxes)),
            key=lambda i: float(raw_boxes[i].conf[0]),
            reverse=True
        )

        keep = []
        for i in order:
            bi = raw_boxes[i].xyxy[0].cpu().numpy()
            if all(self._iom(bi, raw_boxes[j].xyxy[0].cpu().numpy()) <= self.iom_thresh for j in keep):
                keep.append(i)

        self.filtered_boxes = [raw_boxes[i] for i in keep]
        filtered_masks = [raw_masks[i] for i in keep]
        return self.filtered_boxes, filtered_masks

    def detect_points(self, frame, classes_override=None):
        boxes, masks = self._get_filtered_results(frame, classes_override=classes_override)
        if not boxes:
            return []

        if self.qpoint_model is None:
            return [
                (
                    int((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                    int((b.xyxy[0][1] + b.xyxy[0][3]) / 2)
                )
                for b in boxes
            ]

        norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1).half()
        norm_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1).half()

        img_t = torch.from_numpy(frame).to(self.device)
        img_t = img_t[:, :, [2, 1, 0]].permute(2, 0, 1).half() / 255.0
        img_t = (img_t - norm_mean) / norm_std

        all_tensors = []
        all_masks = []
        all_metadata = []

        for i in range(len(boxes)):
            b = boxes[i].xyxy[0].int()
            x1, y1 = max(0, b[0].item()), max(0, b[1].item())
            x2, y2 = min(frame.shape[1], b[2].item()), min(frame.shape[0], b[3].item())

            if x2 <= x1 or y2 <= y1:
                continue

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
            all_metadata.append({
                'x1': x1,
                'y1': y1,
                'scale': scale,
                'dx': dx,
                'dy': dy
            })

        if not all_tensors:
            return []

        batch_t = torch.stack(all_tensors)
        batch_m = torch.stack(all_masks)

        with torch.no_grad():
            heatmaps = self.qpoint_model(batch_t).squeeze(1)

        heatmaps_cpu = heatmaps.detach().cpu().float().numpy()
        batch_masks_cpu = batch_m.detach().cpu().float().numpy()

        coords = []
        for i, meta in enumerate(all_metadata):
            pt_224, conf = self._extract_qpoint(heatmaps_cpu[i], batch_masks_cpu[i])

            if pt_224 is None:
                continue

            lx, ly = pt_224

            gx = int(round((lx - meta['dx']) / meta['scale'])) + meta['x1']
            gy = int(round((ly - meta['dy']) / meta['scale'])) + meta['y1']

            gx = max(0, min(frame.shape[1] - 1, gx))
            gy = max(0, min(frame.shape[0] - 1, gy))

            if QPOINT_DEBUG:
                print(
                    f"  [qpoint] det {i}: conf={conf:.3f}, "
                    f"224=({lx:.1f},{ly:.1f}) -> global=({gx},{gy})"
                )

            coords.append((gx, gy))

        return coords

    def return_burst_stable(
        self,
        frames,
        min_stable_views=AI_MIN_STABLE_VIEWS,
        group_iou_thresh=0.25,
        group_radius_px=None,
        classes_override=None,
    ):
        all_detections = []

        for frame in frames:
            coords = self.detect_points(frame, classes_override=classes_override)
            frame_data = []

            for i, box in enumerate(self.filtered_boxes):
                if i >= len(coords):
                    continue
                b = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                frame_data.append({
                    "box":   (x1, y1, x2, y2),
                    "point": coords[i],
                    "cls":   int(box.cls[0].cpu().item()),
                    "conf":  float(box.conf[0].cpu().item()),
                })

            all_detections.append(frame_data)

        groups = []

        for frame_data in all_detections:
            for det in frame_data:
                best_group = None
                best_iou = 0.0

                for gi, g in enumerate(groups):
                    iou = _box_iou(det["box"], g["box_mean"])
                    if iou >= group_iou_thresh and iou > best_iou:
                        best_iou = iou
                        best_group = gi

                if best_group is None:
                    groups.append({
                        "boxes":     [det["box"]],
                        "points":    [det["point"]],
                        "cls_votes": [det["cls"]],
                        "confs":     [det["conf"]],
                        "views":     1,
                        "box_mean":  det["box"],
                    })
                else:
                    g = groups[best_group]
                    g["boxes"].append(det["box"])
                    g["points"].append(det["point"])
                    g["cls_votes"].append(det["cls"])
                    g["confs"].append(det["conf"])
                    g["views"] += 1
                    arr = np.array(g["boxes"])
                    g["box_mean"] = tuple(np.mean(arr, axis=0).tolist())

        stable = []
        for g in groups:
            if g["views"] < min_stable_views:
                continue

            pts = np.array(g["points"], dtype=float)
            mean_pt = np.mean(pts, axis=0)
            arr = np.array(g["boxes"])
            mean_box = tuple(np.mean(arr, axis=0).tolist())
            modal_cls = Counter(g["cls_votes"]).most_common(1)[0][0]
            mean_conf = float(np.mean(g["confs"]))

            stable.append({
                "point": (int(round(mean_pt[0])), int(round(mean_pt[1]))),
                "box":   mean_box,
                "views": g["views"],
                "cls":   modal_cls,
                "conf":  mean_conf,
            })

        stable.sort(key=lambda d: d["box"][0])
        return stable


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
        target_class=AI_TARGET_CLASS,
    ):
        params_dir = BASE_DIR / "params"

        self.yolo_path = Path(yolo_path) if yolo_path is not None else params_dir / "26_plastic_nano.pt"
        if not self.yolo_path.is_absolute():
            self.yolo_path = params_dir / self.yolo_path

        # Resolve qpoint path: explicit arg > DEFAULT_QPOINT_MODEL config > None (disabled)
        if qpoint_path is not None:
            self.qpoint_path = Path(qpoint_path)
            if not self.qpoint_path.is_absolute():
                self.qpoint_path = params_dir / self.qpoint_path
        elif DEFAULT_QPOINT_MODEL is not None:
            filename = MODEL_MAP.get(DEFAULT_QPOINT_MODEL)
            self.qpoint_path = params_dir / filename if filename else None
        else:
            self.qpoint_path = None

        self.display_scale = display_scale
        self.burst_size = burst_size
        self.min_stable_views = min_stable_views
        self.target_class = target_class
        self.window_name = "AI Detector - Stereo Pair"

        core_kwargs = dict(conf=conf, iom_thresh=iom_thresh, target_class=target_class)
        self.cv_left  = _WeedCVCore(self.yolo_path, self.qpoint_path, verbose=True,  **core_kwargs)
        self.cv_right = _WeedCVCore(self.yolo_path, self.qpoint_path, verbose=False, **core_kwargs)

    def _collect_burst(self, cameras):
        left_frames, right_frames = [], []
        for _ in range(self.burst_size):
            lf, rf = cameras.read_pair()
            left_frames.append(lf)
            right_frames.append(rf)
        return left_frames, right_frames

    def _draw_panel(self, left_frame, right_frame, left_points, right_points):
        h, w = left_frame.shape[:2]
        disp_w = int(w * self.display_scale)
        disp_h = int(h * self.display_scale)
        sx = sy = self.display_scale

        left_disp = cv2.resize(left_frame.copy(), (disp_w, disp_h))
        right_disp = cv2.resize(right_frame.copy(), (disp_w, disp_h))

        for pt in left_points:
            cv2.circle(left_disp, (int(pt[0] * sx), int(pt[1] * sy)), 5, (0, 0, 255), -1)
        for pt in right_points:
            cv2.circle(right_disp, (int(pt[0] * sx), int(pt[1] * sy)), 5, (0, 0, 255), -1)

        return np.hstack([left_disp, right_disp])

    def refine_live(self, cameras):
        left_points, right_points = self.detect_live(cameras)
        if not left_points or not right_points:
            return None, None

        target_xy = (FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0)

        def dist2(p):
            return (p[0] - target_xy[0]) ** 2 + (p[1] - target_xy[1]) ** 2

        return min(left_points, key=dist2), min(right_points, key=dist2)

    def detect_live(self, cameras):
        lf, rf = cameras.read_pair()
        return self.cv_left.detect_points(lf), self.cv_right.detect_points(rf)

    def detect_stable_live(self, cameras, classes_override=None):
        left_frames, right_frames = self._collect_burst(cameras)
        stable_left = self.cv_left.return_burst_stable(
            left_frames,
            min_stable_views=self.min_stable_views,
            classes_override=classes_override
        )
        stable_right = self.cv_right.return_burst_stable(
            right_frames,
            min_stable_views=self.min_stable_views,
            classes_override=classes_override
        )
        return stable_left, stable_right

    def show_live(self, cameras):
        while True:
            lf, rf = cameras.read_pair()
            lp = self.cv_left.detect_points(lf)
            rp = self.cv_right.detect_points(rf)
            cv2.imshow(self.window_name, self._draw_panel(lf, rf, lp, rp))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyWindow(self.window_name)