from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import time
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
    AI_CLASS_CONFIDENCE,
    AI_IOM_THRESHOLD,
    AI_TARGET_CLASS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_ENGINE,
    DEFAULT_QPOINT_MODEL,
    MODEL_MAP,
    CV_WEIGHTS_DIR,
    QPOINT_DEBUG,
    YOLO_BACKEND,
    USE_TENSORRT_ENGINE,
    YOLO_DEVICE,
    YOLO_HALF,
    YOLO_WARMUP,
    YOLO_WARMUP_IMGSZ,
    YOLO_WARMUP_ITERS,
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


def _box_center(box):
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def _point_dist(p1, p2):
    return float(np.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))


def _adaptive_group_radius(box, base_radius=None):
    bw = max(1.0, float(box[2]) - float(box[0]))
    bh = max(1.0, float(box[3]) - float(box[1]))
    return max(float(base_radius or 0.0), 0.30 * min(bw, bh), 10.0)


def _resolve_classes(target_class, override=None):
    spec = override if override is not None else target_class
    if spec is None:
        return None
    if isinstance(spec, int):
        return [spec]
    return list(spec)


def _resolve_weight_path(model_name_or_path):
    if model_name_or_path is None:
        return None
    filename = MODEL_MAP.get(model_name_or_path, model_name_or_path)
    path = Path(filename)
    if not path.is_absolute():
        path = CV_WEIGHTS_DIR / path
    return path


def _normalise_point_mode(point_mode, default="qpoint"):
    mode = (point_mode or default or "box_center").strip().lower()
    aliases = {
        "center": "box_center",
        "bbox_center": "box_center",
        "heatmap": "qpoint",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("box_center", "qpoint"):
        raise ValueError(f"Unknown point mode: {point_mode}")
    return mode


def _uses_qpoint(point_mode):
    return _normalise_point_mode(point_mode) == "qpoint"


def _is_cuda_device(device):
    if isinstance(device, int):
        return True
    if device is None:
        return torch.cuda.is_available()
    return str(device).lower().startswith("cuda")


def _resolve_yolo_device(device_spec):
    if device_spec is None or str(device_spec).lower() in ("", "auto"):
        return 0 if torch.cuda.is_available() else "cpu"
    if _is_cuda_device(device_spec) and not torch.cuda.is_available():
        print(f"[YOLO] Requested device {device_spec!r}, but CUDA is unavailable; using CPU.")
        return "cpu"
    return device_spec


def _select_yolo_model_path(explicit_path=None):
    if explicit_path is not None:
        path = _resolve_weight_path(explicit_path)
        backend = "engine" if path and path.suffix.lower() == ".engine" else "pt"
        return path, backend

    backend_cfg = str(YOLO_BACKEND or "auto").lower()
    if backend_cfg not in ("pt", "engine", "auto"):
        raise ValueError(f"Unknown YOLO_BACKEND: {YOLO_BACKEND}")

    pt_path = _resolve_weight_path(DEFAULT_MODEL)
    engine_path = _resolve_weight_path(DEFAULT_MODEL_ENGINE)
    want_engine = backend_cfg == "engine" or (backend_cfg == "auto" and USE_TENSORRT_ENGINE)

    if want_engine and engine_path is not None and engine_path.exists():
        return engine_path, "engine"

    if want_engine:
        print(f"[YOLO] TensorRT engine requested but not found: {engine_path}. Falling back to .pt.")

    return pt_path, "pt"


class _WeedCVCore:
    def __init__(self, yolo_path, qpoint_path=None, conf=AI_CONFIDENCE,
                 iom_thresh=AI_IOM_THRESHOLD, target_class=AI_TARGET_CLASS,
                 class_conf=None, verbose=True, yolo_backend=None,
                 yolo_device=YOLO_DEVICE, yolo_half=YOLO_HALF):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_device = _resolve_yolo_device(yolo_device)
        self.yolo_half = bool(yolo_half and _is_cuda_device(self.yolo_device))
        self.yolo_path = Path(yolo_path)
        self.yolo_backend = yolo_backend or ("engine" if self.yolo_path.suffix.lower() == ".engine" else "pt")
        self.qpoint_path = Path(qpoint_path) if qpoint_path is not None else None
        self.verbose = verbose

        self.target_class = target_class
        # Per-class confidence thresholds: {class_id: conf}. Falls back to self.conf.
        self.class_conf = dict(class_conf) if class_conf is not None else dict(AI_CLASS_CONFIDENCE)

        self.yolo = YOLO(str(self.yolo_path), task='segment')

        if verbose:
            print(
                f"[YOLO] backend={self.yolo_backend} model={self.yolo_path} "
                f"device={self.yolo_device} half={self.yolo_half}"
            )
            if self.yolo.names:
                print(f"[INFO] Model classes: {self.yolo.names}")
            resolved = _resolve_classes(self.target_class)
            if resolved is not None:
                names = [self.yolo.names.get(c, "???") for c in resolved]
                print(f"[CV] Filtering to class(es) {resolved} = {names}")
            else:
                print("[CV] No class filter — detecting all classes.")
            if self.class_conf:
                named = {self.yolo.names.get(k, k): v for k, v in self.class_conf.items()}
                print(f"[CV] Per-class conf overrides: {named}")

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

    def _run_qpoints_batch(self, frame, boxes, masks):
        """Batch qpoint inference. Returns [(gx, gy, box_idx, peak_conf), ...]."""
        norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1).half()
        norm_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1).half()

        img_t = torch.from_numpy(frame).to(self.device)
        img_t = img_t[:, :, [2, 1, 0]].permute(2, 0, 1).half() / 255.0
        img_t = (img_t - norm_mean) / norm_std

        all_tensors, all_masks_t, all_metadata = [], [], []

        for i in range(len(boxes)):
            b = boxes[i].xyxy[0].int()
            x1, y1 = max(0, b[0].item()), max(0, b[1].item())
            x2, y2 = min(frame.shape[1], b[2].item()), min(frame.shape[0], b[3].item())
            if x2 <= x1 or y2 <= y1:
                continue

            crop_img  = img_t[:, y1:y2, x1:x2].unsqueeze(0)
            crop_mask = masks[i][y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)

            ch, cw = y2 - y1, x2 - x1
            scale = self.TRAIN_SIZE / max(ch, cw)
            nw, nh = int(cw * scale), int(ch * scale)
            dx, dy = (self.TRAIN_SIZE - nw) // 2, (self.TRAIN_SIZE - nh) // 2

            crop_img_res  = F.interpolate(crop_img,  size=(nh, nw), mode='bilinear', align_corners=False)
            crop_mask_res = F.interpolate(crop_mask, size=(nh, nw), mode='nearest')

            pad_l, pad_r = dx, self.TRAIN_SIZE - nw - dx
            pad_t, pad_b = dy, self.TRAIN_SIZE - nh - dy

            final_img  = F.pad(crop_img_res,  (pad_l, pad_r, pad_t, pad_b), value=0)
            final_mask = F.pad(crop_mask_res, (pad_l, pad_r, pad_t, pad_b), value=0)

            all_tensors.append(final_img.squeeze(0))
            all_masks_t.append(final_mask.squeeze(0).squeeze(0))
            all_metadata.append({'box_idx': i, 'x1': x1, 'y1': y1, 'scale': scale, 'dx': dx, 'dy': dy})

        if not all_tensors:
            return []

        batch_t = torch.stack(all_tensors)
        batch_m = torch.stack(all_masks_t)

        with torch.no_grad():
            heatmaps = self.qpoint_model(batch_t).squeeze(1)

        heatmaps_cpu    = heatmaps.detach().cpu().float().numpy()
        batch_masks_cpu = batch_m.detach().cpu().float().numpy()

        results = []
        for i, meta in enumerate(all_metadata):
            pt_224, peak_conf = self._extract_qpoint(heatmaps_cpu[i], batch_masks_cpu[i])
            if pt_224 is None:
                continue
            lx, ly = pt_224
            gx = int(round((lx - meta['dx']) / meta['scale'])) + meta['x1']
            gy = int(round((ly - meta['dy']) / meta['scale'])) + meta['y1']
            gx = max(0, min(frame.shape[1] - 1, gx))
            gy = max(0, min(frame.shape[0] - 1, gy))
            results.append((gx, gy, meta['box_idx'], peak_conf))

        return results

    def _effective_yolo_conf(self, conf_override=None):
        base = conf_override if conf_override is not None else self.conf
        if self.class_conf:
            # Always run YOLO at the lowest per-class threshold so per-class
            # post-filtering can act on all candidate detections.
            return min(base, min(self.class_conf.values()))
        return base

    def _resolve_imgsz(self, frame_or_frames, imgsz):
        if imgsz is not None:
            return imgsz
        frame = frame_or_frames[0] if isinstance(frame_or_frames, (list, tuple)) else frame_or_frames
        h, w = frame.shape[:2]
        return (h, w)

    def _yolo_predict_kwargs(self):
        kwargs = {}
        if self.yolo_device is not None:
            kwargs["device"] = self.yolo_device
        if self.yolo_half:
            kwargs["half"] = True
        return kwargs

    def _filter_yolo_result(self, result, conf_override=None):
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if result is None or boxes is None or len(boxes) == 0 or masks is None:
            return [], []

        raw_boxes = boxes
        raw_masks = masks.data.half()

        # IoM suppression (keep highest-confidence non-overlapping boxes)
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

        # Per-class confidence filter (applied after IoM so we deduplicate first).
        # Per-class always takes precedence; conf_override (or self.conf) is the fallback
        # for classes that have no per-class entry.
        if self.class_conf:
            fallback = conf_override if conf_override is not None else self.conf
            keep = [
                i for i in keep
                if float(raw_boxes[i].conf[0].cpu().item())
                   >= self.class_conf.get(int(raw_boxes[i].cls[0].cpu().item()), fallback)
            ]

        filtered_boxes = [raw_boxes[i] for i in keep]
        filtered_masks = [raw_masks[i] for i in keep]
        return filtered_boxes, filtered_masks

    def _get_filtered_results(self, frame, classes_override=None, conf_override=None, imgsz=1280):
        classes_arg = _resolve_classes(self.target_class, classes_override)
        run_imgsz = self._resolve_imgsz(frame, imgsz)

        # Run YOLO at the lowest effective threshold so per-class filtering
        # can happen after IoM suppression rather than before it.
        results = self.yolo(
            frame,
            imgsz=run_imgsz,
            verbose=False,
            conf=self._effective_yolo_conf(conf_override),
            retina_masks=True,
            classes=classes_arg,
            **self._yolo_predict_kwargs(),
        )

        result = results[0] if results else None
        self.filtered_boxes, filtered_masks = self._filter_yolo_result(
            result,
            conf_override=conf_override,
        )
        return self.filtered_boxes, filtered_masks

    def _get_filtered_results_batch(self, frames, classes_override=None, conf_override=None, imgsz=1280):
        if not frames:
            self.filtered_boxes = []
            return []

        classes_arg = _resolve_classes(self.target_class, classes_override)
        run_imgsz = self._resolve_imgsz(frames, imgsz)
        results = self.yolo(
            list(frames),
            imgsz=run_imgsz,
            verbose=False,
            conf=self._effective_yolo_conf(conf_override),
            retina_masks=True,
            classes=classes_arg,
            **self._yolo_predict_kwargs(),
        )
        if results is None:
            results = []
        if not isinstance(results, list):
            results = [results]

        batched = [
            self._filter_yolo_result(result, conf_override=conf_override)
            for result in results
        ]
        while len(batched) < len(frames):
            batched.append(([], []))

        self.filtered_boxes = batched[-1][0] if batched else []
        return batched

    def count_at_conf(self, frame, conf_override, classes_override=None):
        """Fast single-frame detection count at a given confidence (no qpoints, no IoM).
        Used for sensitivity analysis after a survey burst."""
        classes_arg = _resolve_classes(self.target_class, classes_override)
        results = self.yolo(
            frame, imgsz=1280, verbose=False,
            conf=conf_override, retina_masks=False, classes=classes_arg,
            **self._yolo_predict_kwargs(),
        )
        if not results or len(results[0].boxes) == 0:
            return 0
        return int(len(results[0].boxes))

    def warmup(self, imgsz=YOLO_WARMUP_IMGSZ, iters=YOLO_WARMUP_ITERS):
        iters = max(0, int(iters or 0))
        if iters == 0:
            return 0.0
        if isinstance(imgsz, (tuple, list)):
            h, w = int(imgsz[0]), int(imgsz[1])
            run_imgsz = (h, w)
        else:
            h = w = int(imgsz or 640)
            run_imgsz = int(imgsz or 640)
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(iters):
            self.yolo(
                dummy,
                imgsz=run_imgsz,
                verbose=False,
                conf=self._effective_yolo_conf(None),
                retina_masks=True,
                classes=_resolve_classes(self.target_class),
                **self._yolo_predict_kwargs(),
            )
        return time.perf_counter() - t0

    def detect_points(self, frame, classes_override=None, point_mode=None):
        boxes, masks = self._get_filtered_results(frame, classes_override=classes_override)
        if not boxes:
            return []

        mode = _normalise_point_mode(point_mode, default="qpoint")
        if self.qpoint_model is None or mode == "box_center":
            return [
                (
                    int((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                    int((b.xyxy[0][1] + b.xyxy[0][3]) / 2)
                )
                for b in boxes
            ]

        coords = []
        for i, (gx, gy, box_idx, peak_conf) in enumerate(self._run_qpoints_batch(frame, boxes, masks)):
            if QPOINT_DEBUG:
                box = boxes[box_idx]
                yolo_conf = float(box.conf[0].cpu().item())
                cls_id    = int(box.cls[0].cpu().item())
                cls_name  = self.yolo.names.get(cls_id, str(cls_id))
                print(
                    f"  [qpoint] det {i}: {cls_name} "
                    f"yolo={yolo_conf:.2f} | "
                    f"heatmap_peak={peak_conf:.4f} | "
                    f"px=({gx},{gy})"
                )
            coords.append((gx, gy))
        return coords

    def detect_rich_points(self, frame, classes_override=None, point_mode=None):
        """Returns [{"point": (gx,gy), "cls": cls_id, "conf": yolo_conf}, ...]."""
        boxes, masks = self._get_filtered_results(frame, classes_override=classes_override)
        if not boxes:
            return []

        mode = _normalise_point_mode(point_mode, default="qpoint")
        if self.qpoint_model is None or mode == "box_center":
            return [
                {
                    "point": (
                        int((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                        int((b.xyxy[0][1] + b.xyxy[0][3]) / 2)
                    ),
                    "cls":  int(b.cls[0].cpu().item()),
                    "conf": float(b.conf[0].cpu().item()),
                }
                for b in boxes
            ]

        return [
            {
                "point": (gx, gy),
                "cls":   int(boxes[box_idx].cls[0].cpu().item()),
                "conf":  float(boxes[box_idx].conf[0].cpu().item()),
            }
            for gx, gy, box_idx, _ in self._run_qpoints_batch(frame, boxes, masks)
        ]

    def draw_detections(self, frame, boxes=None, points=None):
        """Draw YOLO boxes + qpoint markers on frame. Uses self.filtered_boxes if boxes is None."""
        if boxes is None:
            boxes = self.filtered_boxes
        if points is None:
            points = []
        out = frame.copy()
        for i, b in enumerate(boxes):
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            conf = float(b.conf[0].cpu().item())
            cls_id = int(b.cls[0].cpu().item())
            cls_name = self.yolo.names.get(cls_id, str(cls_id))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), (0, 220, 0), -1)
            cv2.putText(out, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if i < len(points):
                px, py = int(points[i][0]), int(points[i][1])
                cv2.circle(out, (px, py), 7, (0, 0, 255), -1)
                cv2.circle(out, (px, py), 10, (255, 255, 255), 1)
        return out

    def draw_stable_detections(self, frame, stable_points):
        """Draw burst-stable cluster results (mean box + mean qpoint + view count)."""
        out = frame.copy()
        for s in stable_points:
            x1, y1, x2, y2 = (int(v) for v in s["box"])
            conf = s.get("conf", 0.0)
            cls_id = s.get("cls", 0)
            cls_name = self.yolo.names.get(cls_id, str(cls_id))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{cls_name} {conf:.2f} [{s['views']}v]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), (0, 200, 255), -1)
            cv2.putText(out, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            px, py = int(s["point"][0]), int(s["point"][1])
            cv2.circle(out, (px, py), 7, (0, 0, 255), -1)
            cv2.circle(out, (px, py), 10, (255, 255, 255), 1)
        return out

    def refine_one_heatmap(self, crop_pt):
        """Run heatmap on ONE detection using the last burst merged image.
        crop_pt: (x, y) box-center in crop space.
        Returns refined (x, y) in crop space, or None if unavailable."""
        if self.qpoint_model is None:
            return None
        merged    = getattr(self, '_last_burst_merged', None)
        last_boxes = getattr(self, '_last_boxes', None)
        last_masks = getattr(self, '_last_masks', None)
        if merged is None or not last_boxes or last_masks is None:
            return None

        px, py = float(crop_pt[0]), float(crop_pt[1])
        best_i, best_area = None, float('inf')
        for i, box in enumerate(last_boxes):
            b = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    best_i = i

        if best_i is None:
            return None

        qpoints = self._run_qpoints_batch(merged, [last_boxes[best_i]], [last_masks[best_i]])
        if qpoints:
            gx, gy, _, _ = qpoints[0]
            return (gx, gy)
        return None

    def snap_meristem_on_crop(self, crop_img):
        """Run qpoint model directly on a tight crop — no YOLO needed.
        Treats the entire crop as the plant region (used after LK settles).
        Returns (x, y) in crop-image coordinates, or None on failure."""
        if self.qpoint_model is None or crop_img is None or crop_img.size == 0:
            return None
        h, w = crop_img.shape[:2]
        if h < 4 or w < 4:
            return None

        norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1).half()
        norm_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1).half()

        img_t = torch.from_numpy(crop_img).to(self.device)
        img_t = img_t[:, :, [2, 1, 0]].permute(2, 0, 1).half() / 255.0
        img_t = (img_t - norm_mean) / norm_std

        scale = self.TRAIN_SIZE / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        dx = (self.TRAIN_SIZE - nw) // 2
        dy = (self.TRAIN_SIZE - nh) // 2
        pad_l = dx
        pad_r = self.TRAIN_SIZE - nw - dx
        pad_t = dy
        pad_b = self.TRAIN_SIZE - nh - dy

        resized = F.interpolate(img_t.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False)
        final_img = F.pad(resized, (pad_l, pad_r, pad_t, pad_b), value=0)
        full_mask = np.ones((self.TRAIN_SIZE, self.TRAIN_SIZE), dtype=np.float32)

        with torch.no_grad():
            heatmap = self.qpoint_model(final_img).squeeze(0).squeeze(0)

        heatmap_np = heatmap.detach().cpu().float().numpy()
        pt_224, _ = self._extract_qpoint(heatmap_np, full_mask)
        if pt_224 is None:
            return None

        gx = int(round((pt_224[0] - dx) / scale))
        gy = int(round((pt_224[1] - dy) / scale))
        return (max(0, min(w - 1, gx)), max(0, min(h - 1, gy)))

    def return_burst_stable(
        self,
        frames,
        min_stable_views=2,
        group_iou_thresh=0.25,
        group_radius_px=None,
        classes_override=None,
        debug_label=None,
        imgsz=1280,
        heatmap_final=True,
        point_mode=None,
    ):
        t_total = time.perf_counter()
        mode = _normalise_point_mode(point_mode, default="qpoint")
        self.last_burst_timing = {}

        def _debug(msg):
            if debug_label:
                print(f"{debug_label} {msg}", flush=True)

        if not frames:
            self._last_burst_merged = None
            _debug("empty burst: 0 frame(s)")
            return []

        t_yolo = time.perf_counter()
        batched_results = self._get_filtered_results_batch(
            frames,
            classes_override=classes_override,
            imgsz=imgsz,
        )
        yolo_dt = time.perf_counter() - t_yolo
        box_counts = [len(boxes) for boxes, _ in batched_results]
        run_imgsz = self._resolve_imgsz(frames, imgsz)
        _debug(
            f"YOLO batch {len(frames)} frame(s) imgsz={run_imgsz} "
            f"point_mode={mode}: boxes={box_counts} in {yolo_dt:.2f}s"
        )

        all_detections = []
        last_boxes = None
        last_masks = None
        for boxes, masks in batched_results:
            last_boxes, last_masks = boxes, masks
            frame_data = []
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                frame_data.append({
                    "box":   (x1, y1, x2, y2),
                    "point": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    "cls":   int(box.cls[0].cpu().item()),
                    "conf":  float(box.conf[0].cpu().item()),
                })
            all_detections.append(frame_data)

        t_group = time.perf_counter()
        groups = []

        def _refresh_group(g):
            arr = np.array(g["boxes"], dtype=float)
            pts = np.array(g["points"], dtype=float)
            g["box_mean"] = tuple(np.mean(arr, axis=0).tolist())
            g["point_mean"] = tuple(np.mean(pts, axis=0).tolist())

        def _group_modal_cls(g):
            return Counter(g["cls_votes"]).most_common(1)[0][0]

        def _group_match_score(det, g):
            iou = _box_iou(det["box"], g["box_mean"])
            radius = max(
                _adaptive_group_radius(det["box"], group_radius_px),
                _adaptive_group_radius(g["box_mean"], group_radius_px),
            )
            dist = _point_dist(det["point"], g["point_mean"])
            same_cls = det["cls"] == _group_modal_cls(g)
            near_enough = dist <= radius
            overlap_enough = iou >= group_iou_thresh
            if not (near_enough or overlap_enough):
                return None

            dist_score = max(0.0, 1.0 - dist / max(1.0, radius))
            cls_bonus = 0.05 if same_cls else 0.0
            return max(iou, dist_score) + cls_bonus

        def _groups_look_duplicate(g1, g2):
            iou = _box_iou(g1["box_mean"], g2["box_mean"])
            radius = max(
                _adaptive_group_radius(g1["box_mean"], group_radius_px),
                _adaptive_group_radius(g2["box_mean"], group_radius_px),
            )
            dist = _point_dist(g1["point_mean"], g2["point_mean"])
            cls1 = _group_modal_cls(g1)
            cls2 = _group_modal_cls(g2)
            if iou >= max(0.15, 0.60 * group_iou_thresh):
                return True
            if cls1 == cls2 and dist <= 0.90 * radius:
                return True
            return dist <= 0.55 * radius

        def _merge_groups(dst, src):
            dst["boxes"].extend(src["boxes"])
            dst["points"].extend(src["points"])
            dst["cls_votes"].extend(src["cls_votes"])
            dst["confs"].extend(src["confs"])
            dst["views"] += src["views"]
            _refresh_group(dst)

        for frame_data in all_detections:
            for det in frame_data:
                best_group = None
                best_score = -1.0

                for gi, g in enumerate(groups):
                    score = _group_match_score(det, g)
                    if score is not None and score > best_score:
                        best_score = score
                        best_group = gi

                if best_group is None:
                    group = {
                        "boxes":     [det["box"]],
                        "points":    [det["point"]],
                        "cls_votes": [det["cls"]],
                        "confs":     [det["conf"]],
                        "views":     1,
                        "box_mean":  det["box"],
                        "point_mean": det["point"],
                    }
                    groups.append(group)
                else:
                    g = groups[best_group]
                    g["boxes"].append(det["box"])
                    g["points"].append(det["point"])
                    g["cls_votes"].append(det["cls"])
                    g["confs"].append(det["conf"])
                    g["views"] += 1
                    _refresh_group(g)

        # Second pass: collapse any leftover duplicate groups for the same plant.
        dedup_groups = groups[:]
        changed = True
        while changed and len(dedup_groups) > 1:
            changed = False
            next_groups = []
            used = [False] * len(dedup_groups)
            for i, base in enumerate(dedup_groups):
                if used[i]:
                    continue
                for j in range(i + 1, len(dedup_groups)):
                    if used[j]:
                        continue
                    other = dedup_groups[j]
                    if _groups_look_duplicate(base, other):
                        _merge_groups(base, other)
                        used[j] = True
                        changed = True
                next_groups.append(base)
            dedup_groups = next_groups
        groups = dedup_groups

        stable = []
        for g in groups:
            if g["views"] < min_stable_views:
                continue

            mean_pt = np.array(g["point_mean"], dtype=float)
            mean_box = g["box_mean"]
            modal_cls = _group_modal_cls(g)
            mean_conf = float(np.mean(g["confs"]))

            stable.append({
                "point": (int(round(mean_pt[0])), int(round(mean_pt[1]))),
                "box":   mean_box,
                "views": g["views"],
                "cls":   modal_cls,
                "conf":  mean_conf,
                "point_source": "box_center",
            })

        stable.sort(key=lambda d: d["box"][0])
        group_dt = time.perf_counter() - t_group
        _debug(
            f"grouped {sum(len(d) for d in all_detections)} detection(s) "
            f"into {len(stable)} stable target(s) in {group_dt:.2f}s"
        )

        # Merge all burst frames into one image (averaging reduces noise, sharpens plant signal).
        t_merge = time.perf_counter()
        need_qpoint_image = heatmap_final and _uses_qpoint(mode) and self.qpoint_model is not None
        if need_qpoint_image and len(frames) > 1:
            merged = np.mean(np.stack(frames), axis=0).astype(np.uint8)
        else:
            merged = frames[-1].copy()
        self._last_burst_merged = merged
        self._last_boxes = last_boxes
        self._last_masks = last_masks
        merge_dt = time.perf_counter() - t_merge
        _debug(f"prepared final burst image from {len(frames)} frame(s) in {merge_dt:.2f}s")

        # One heatmap pass on the merged image using last-frame masks (scene is stationary).
        # Skip when heatmap_final=False — caller will run heatmap on the one winning detection.
        qpoint_dt = 0.0
        if (
            heatmap_final
            and _uses_qpoint(mode)
            and self.qpoint_model is not None
            and stable
            and last_boxes
            and last_masks
        ):
            t_qpoint = time.perf_counter()
            qpoints = self._run_qpoints_batch(merged, last_boxes, last_masks)
            qpoint_dt = time.perf_counter() - t_qpoint
            _debug(
                f"qpoint pass: {len(qpoints)} point(s) "
                f"in {qpoint_dt:.2f}s"
            )
            qpoint_map = {box_idx: (gx, gy) for gx, gy, box_idx, _ in qpoints}
            for s in stable:
                best_i, best_iou = None, group_iou_thresh
                for i, box in enumerate(last_boxes):
                    b = box.xyxy[0].cpu().numpy()
                    b_tuple = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                    iou = _box_iou(s["box"], b_tuple)
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i
                if best_i is not None and best_i in qpoint_map:
                    s["point"] = qpoint_map[best_i]
                    s["point_source"] = "qpoint"

        total_dt = time.perf_counter() - t_total
        self.last_burst_timing = {
            "yolo_time_s": round(yolo_dt, 6),
            "grouping_time_s": round(group_dt, 6),
            "merge_time_s": round(merge_dt, 6),
            "qpoint_time_s": round(qpoint_dt, 6),
            "total_time_s": round(total_dt, 6),
            "point_mode": mode,
            "heatmap_final": bool(heatmap_final and _uses_qpoint(mode)),
        }
        _debug(f"burst stable total: {total_dt:.2f}s")
        return stable


class AIDetector:
    def __init__(
        self,
        display_scale=1.5,
        burst_size=2,
        min_stable_views=2,
        yolo_path=None,
        qpoint_path=None,
        conf=AI_CONFIDENCE,
        iom_thresh=AI_IOM_THRESHOLD,
        target_class=AI_TARGET_CLASS,
    ):
        self.yolo_path, self.yolo_backend = _select_yolo_model_path(yolo_path)

        # Resolve qpoint path: explicit arg > DEFAULT_QPOINT_MODEL config > None (disabled)
        if qpoint_path is not None:
            self.qpoint_path = _resolve_weight_path(qpoint_path)
        elif DEFAULT_QPOINT_MODEL is not None:
            self.qpoint_path = _resolve_weight_path(DEFAULT_QPOINT_MODEL)
        else:
            self.qpoint_path = None

        self.display_scale = display_scale
        self.burst_size = burst_size
        self.min_stable_views = min_stable_views
        self.target_class = target_class
        self.window_name = "AI Detector - Stereo Pair"

        core_kwargs = dict(
            conf=conf,
            iom_thresh=iom_thresh,
            target_class=target_class,
            yolo_backend=self.yolo_backend,
            yolo_device=YOLO_DEVICE,
            yolo_half=YOLO_HALF,
        )
        self.cv_left  = _WeedCVCore(self.yolo_path, self.qpoint_path, verbose=True,  **core_kwargs)
        self.cv_right = _WeedCVCore(self.yolo_path, self.qpoint_path, verbose=False, **core_kwargs)
        # AI_CLASS_CONFIDENCE is loaded inside _WeedCVCore.__init__ automatically.

    def warmup(self, imgsz=YOLO_WARMUP_IMGSZ, iters=YOLO_WARMUP_ITERS, enabled=YOLO_WARMUP):
        if not enabled:
            print("[YOLO] Warmup disabled.")
            return {"enabled": False, "warmup_time_s": 0.0}

        print(f"[YOLO] Warmup start: imgsz={imgsz} iters={iters}")
        t0 = time.perf_counter()
        left_dt = self.cv_left.warmup(imgsz=imgsz, iters=iters)
        right_dt = self.cv_right.warmup(imgsz=imgsz, iters=iters)
        total_dt = time.perf_counter() - t0
        print(f"[YOLO] Warmup done in {total_dt:.3f}s (left={left_dt:.3f}s right={right_dt:.3f}s)")
        return {
            "enabled": True,
            "warmup_time_s": round(total_dt, 6),
            "warmup_left_time_s": round(left_dt, 6),
            "warmup_right_time_s": round(right_dt, 6),
            "warmup_imgsz": imgsz,
            "warmup_iters": iters,
        }

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

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="stable-live") as pool:
            left_future = pool.submit(
                self.cv_left.return_burst_stable,
                left_frames,
                min_stable_views=self.min_stable_views,
                classes_override=classes_override,
                point_mode="qpoint",
            )
            right_future = pool.submit(
                self.cv_right.return_burst_stable,
                right_frames,
                min_stable_views=self.min_stable_views,
                classes_override=classes_override,
                point_mode="qpoint",
            )
            stable_left = left_future.result()
            stable_right = right_future.result()
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
