from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import re
import time
import warnings
import cv2
import numpy as np
import torch
from vision.visualization import (
    BOX_COLOR_RAW,
    QPOINT_COLOR, QPOINT_OUTLINE_COLOR, QPOINT_RADIUS, QPOINT_OUTLINE_RADIUS,
    LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS,
    draw_stable_detections as _vis_draw_stable,
)
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
    TARGET_CLASSES,
    AVOID_CLASSES,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PT,
    DEFAULT_MODEL_ENGINE,
    DEFAULT_QPOINT_MODEL,
    MODEL_MAP,
    CV_WEIGHTS_DIR,
    QPOINT_DEBUG,
    YOLO_BACKEND,
    USE_TENSORRT_ENGINE,
    YOLO_DEVICE,
    YOLO_ENGINE_BATCH_SIZE,
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


class SpatialSoftArgmax2d(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        softmax = torch.softmax(x / self.temperature, dim=-1)
        softmax = softmax.view(b, c, h, w)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=x.device),
            torch.linspace(0.0, 1.0, w, device=x.device),
            indexing='ij'
        )

        expected_x = torch.sum(softmax * grid_x, dim=[2, 3])
        expected_y = torch.sum(softmax * grid_y, dim=[2, 3])
        return torch.cat([expected_x, expected_y], dim=-1)


class SniperSoftArgmaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None).features
        self.final_conv = nn.Conv2d(576, 1, kernel_size=1)
        self.soft_argmax = SpatialSoftArgmax2d(temperature=1.0)

    def forward(self, x):
        features = self.backbone(x)
        heatmap_logits = self.final_conv(features)
        return self.soft_argmax(heatmap_logits)


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
    if mode not in ("box_center", "qpoint", "softargmax"):
        raise ValueError(f"Unknown point mode: {point_mode}")
    return mode


def _uses_qpoint(point_mode):
    return _normalise_point_mode(point_mode) in ("qpoint", "softargmax")


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


def _normalise_imgsz(imgsz):
    if isinstance(imgsz, (tuple, list)):
        if len(imgsz) == 1:
            return int(imgsz[0])
        return (int(imgsz[0]), int(imgsz[1]))
    return int(imgsz)


def _infer_engine_batch_size(path):
    if YOLO_ENGINE_BATCH_SIZE is not None:
        return max(1, int(YOLO_ENGINE_BATCH_SIZE))
    match = re.search(r"(?:^|[_-])batch[_-]?(\d+)(?:[_\-.]|$)", Path(path).name, re.IGNORECASE)
    if match:
        return max(1, int(match.group(1)))
    return 1


def _select_yolo_model_path(explicit_path=None):
    if explicit_path is not None:
        path = _resolve_weight_path(explicit_path)
        backend = "engine" if path and path.suffix.lower() == ".engine" else "pt"
        return path, backend

    backend_cfg = str(YOLO_BACKEND or "auto").lower()
    if backend_cfg not in ("pt", "engine", "auto"):
        raise ValueError(f"Unknown YOLO_BACKEND: {YOLO_BACKEND}")

    pt_path = _resolve_weight_path(DEFAULT_MODEL_PT or DEFAULT_MODEL)
    engine_path = _resolve_weight_path(DEFAULT_MODEL_ENGINE)
    want_engine = backend_cfg == "engine" or (backend_cfg == "auto" and USE_TENSORRT_ENGINE)

    if want_engine and engine_path is not None and engine_path.exists():
        return engine_path, "engine"

    if want_engine:
        print(f"[YOLO] TensorRT engine requested but not found: {engine_path}. Falling back to .pt.")

    return pt_path, "pt"


class _WeedCVCore:
    def __init__(self, yolo_path, qpoint_path=None, conf=AI_CONFIDENCE,
                 iom_thresh=AI_IOM_THRESHOLD,
                 target_classes=TARGET_CLASSES, avoid_classes=AVOID_CLASSES,
                 class_conf=None, verbose=True, yolo_backend=None,
                 yolo_device=YOLO_DEVICE, yolo_half=YOLO_HALF,
                 iom_enabled=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_device = _resolve_yolo_device(yolo_device)
        self.yolo_half = bool(yolo_half and _is_cuda_device(self.yolo_device))
        self.yolo_path = Path(yolo_path)
        self.yolo_backend = yolo_backend or ("engine" if self.yolo_path.suffix.lower() == ".engine" else "pt")
        self.yolo_engine_batch_size = _infer_engine_batch_size(self.yolo_path) if self.yolo_backend == "engine" else None
        self.qpoint_path = Path(qpoint_path) if qpoint_path is not None else None
        self.verbose = verbose

        self.target_classes = list(target_classes) if target_classes is not None else None
        self.avoid_classes  = list(avoid_classes)  if avoid_classes  else []
        self.avoid_confidence_override = None
        # Per-class confidence thresholds: {class_id: conf}. Falls back to self.conf.
        self.class_conf = dict(class_conf or AI_CLASS_CONFIDENCE or {})

        # Use same loading as eval.py: infer task from weights
        self.yolo = YOLO(str(self.yolo_path))

        if verbose:
            print(
                f"[YOLO] backend={self.yolo_backend} model={self.yolo_path} "
                f"device={self.yolo_device} half={self.yolo_half}"
                f"{f' engine_batch={self.yolo_engine_batch_size}' if self.yolo_engine_batch_size else ''}"
            )
            if self.yolo.names:
                print(f"[INFO] Model classes: {self.yolo.names}")
            if self.target_classes is not None:
                names = [self.yolo.names.get(c, "???") for c in self.target_classes]
                print(f"[CV] TARGET_CLASSES: {self.target_classes} = {names}")
            else:
                print("[CV] TARGET_CLASSES: all (no filter)")
            if self.avoid_classes:
                names = [self.yolo.names.get(c, "???") for c in self.avoid_classes]
                print(f"[CV] AVOID_CLASSES:  {self.avoid_classes} = {names}  (suppress overlapping targets)")
            else:
                print("[CV] AVOID_CLASSES:  none")
            if self.class_conf:
                named = {self.yolo.names.get(k, k): v for k, v in self.class_conf.items()}
                print(f"[CV] Per-class conf overrides: {named}")

        self.qpoint_model = None
        self.filtered_boxes = []

        self.conf = conf
        self.iom_thresh = iom_thresh
        self.iom_enabled = iom_enabled
        self.TRAIN_SIZE = 224
        self.SOFT_IMG_SIZE = 320

        self._load_qpoint_model()

    def _load_qpoint_model(self):
        if self.qpoint_path is None or not self.qpoint_path.exists():
            if self.verbose:
                print("[WARN] No qpoint model found. Falling back to box centers.")
            return

        try:
            # Determine which architecture to use based on filename.
            # Filenames with 'softargmax' use the new SniperSoftArgmaxModel.
            if "softargmax" in self.qpoint_path.name.lower():
                self.qpoint_model = SniperSoftArgmaxModel().to(self.device)
                self.qpoint_is_softargmax = True
            else:
                self.qpoint_model = MeristemPredictor().to(self.device)
                self.qpoint_is_softargmax = False

            self.qpoint_model.load_state_dict(
                torch.load(str(self.qpoint_path), map_location=self.device, weights_only=True)
            )
            self.qpoint_model.half().eval()
            if self.verbose:
                print(f"[INFO] Using qpoint model ({'SoftArgmax' if self.qpoint_is_softargmax else 'Heatmap'}): {self.qpoint_path.name}")
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

    def _all_detect_classes(self, target_override=None):
        """Union of effective target classes + avoid classes for YOLO. None = all."""
        targets = target_override if target_override is not None else self.target_classes
        if targets is None and not self.avoid_classes:
            return None
        all_cls = set(self.avoid_classes)
        if targets is not None:
            all_cls.update(targets)
        return sorted(all_cls) if all_cls else None

    def _is_target(self, cls_id, target_override=None):
        """True if cls_id should be targeted (not in avoid, in effective target set)."""
        if target_override == "all":
            return True
        if cls_id in self.avoid_classes:
            return False
        targets = target_override if target_override is not None else self.target_classes
        return targets is None or cls_id in targets

    def _filter_to_targets(self, boxes, masks, target_override=None):
        """Keep only target-class boxes/masks (drops avoid-class detections)."""
        kept_b, kept_m = [], []
        for b, m in zip(boxes, masks):
            if self._is_target(int(b.cls[0].cpu().item()), target_override):
                kept_b.append(b)
                kept_m.append(m)
        return kept_b, kept_m

    def _suppress_avoid_overlaps(self, stable, target_override=None):
        """Remove targets overlapped by avoid-class detections.

        With an explicit avoid confidence, any avoid detection at/above that
        threshold can veto an overlapping target. Otherwise preserve the older
        behavior: the avoid detection must be higher-confidence than the target.
        """
        if target_override == "all":
            return stable
        if not self.avoid_classes or not stable:
            return stable
        avoid = [s for s in stable if s["cls"] in self.avoid_classes]
        if not avoid:
            return stable
        kept = []
        for s in stable:
            if s["cls"] in self.avoid_classes:
                kept.append(s)
                continue
            if self.avoid_confidence_override is not None:
                dominated = any(
                    self._iom(s["box"], a["box"]) >= self.iom_thresh
                    and a["conf"] >= float(self.avoid_confidence_override)
                    for a in avoid
                )
            else:
                dominated = any(
                    self._iom(s["box"], a["box"]) >= self.iom_thresh and a["conf"] > s["conf"]
                    for a in avoid
                )
            if not dominated:
                kept.append(s)
        return kept

    def _avoid_threshold_for_class(self, cls_id):
        cls_id = int(cls_id)
        if self.avoid_confidence_override is not None:
            return float(self.avoid_confidence_override)
        if cls_id in self.class_conf:
            return float(self.class_conf[cls_id])
        return float(self.conf)

    def _choose_group_class(self, class_mean_conf, target_override=None):
        """Pick a representative class for one grouped burst detection.

        If any avoid class in the group meets its configured avoid threshold and
        the group also contains a target class, force the group to that avoid
        class so target output is vetoed consistently.
        """
        if not class_mean_conf:
            return None
        
        if target_override == "all":
            return max(class_mean_conf.items(), key=lambda item: float(item[1]))[0]

        target_present = any(
            self._is_target(int(cls_id), target_override)
            for cls_id in class_mean_conf.keys()
        )
        avoid_candidates = [
            (int(cls_id), float(conf))
            for cls_id, conf in class_mean_conf.items()
            if int(cls_id) in self.avoid_classes
            and float(conf) >= self._avoid_threshold_for_class(cls_id)
        ]

        if target_present and avoid_candidates:
            return max(avoid_candidates, key=lambda item: item[1])[0]

        return max(class_mean_conf.items(), key=lambda item: float(item[1]))[0]

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

    def _run_qpoints_batch(self, frame, boxes, masks, mode="qpoint"):
        """Batch qpoint inference. Returns [(gx, gy, box_idx, peak_conf), ...]."""
        mode = _normalise_point_mode(mode)
        is_soft = (mode == "softargmax")
        
        # Ensure we have the right model loaded for the requested mode
        if is_soft and not getattr(self, "qpoint_is_softargmax", False):
            if self.verbose:
                print(f"[WARN] SoftArgmax requested but Heatmap model loaded. SoftArgmax may fail or be inaccurate.")
        elif mode == "qpoint" and getattr(self, "qpoint_is_softargmax", False):
            if self.verbose:
                print(f"[WARN] Heatmap requested but SoftArgmax model loaded. Forcing SoftArgmax logic.")
            is_soft = True

        img_size = self.SOFT_IMG_SIZE if is_soft else self.TRAIN_SIZE

        # Standard ImageNet normalization for legacy model
        norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1).half()
        norm_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1).half()

        # Convert frame to tensor [3, H, W] in 0.0-1.0 range
        frame_t = torch.from_numpy(frame).to(self.device)
        frame_t = frame_t[:, :, [2, 1, 0]].permute(2, 0, 1).half() / 255.0

        all_tensors, all_masks_t, all_metadata = [], [], []

        for i in range(len(boxes)):
            if masks[i] is None:
                continue
            b = boxes[i].xyxy[0].int()
            
            # Add padding like sniper_factory.py
            padding = 5
            x1 = max(0, b[0].item() - padding)
            y1 = max(0, b[1].item() - padding)
            x2 = min(frame.shape[1], b[2].item() + padding)
            y2 = min(frame.shape[0], b[3].item() + padding)
            
            if x2 <= x1 or y2 <= y1:
                continue

            # 1. Extract RGB crop and corresponding mask slice
            crop_img  = frame_t[:, y1:y2, x1:x2]
            crop_mask = masks[i][y1:y2, x1:x2].unsqueeze(0) # [1, h, w]
            
            # 2. Apply mask to image: black out everything outside the YOLO segmentation.
            # We threshold the mask at 0.5 to ensure a clean cutout.
            binary_mask = (crop_mask > 0.5).to(crop_img.dtype)
            masked_crop = (crop_img * binary_mask).unsqueeze(0) # [1, 3, h, w]
            
            ch, cw = y2 - y1, x2 - x1
            
            if is_soft:
                # Squashing resize for softargmax (matches visualizer script)
                final_img = F.interpolate(masked_crop, size=(img_size, img_size), mode='bilinear', align_corners=False)
                # No padding or ImageNet normalization for softargmax
                dx = dy = 0
                scale_x = img_size / cw
                scale_y = img_size / ch
            else:
                # Preserve aspect ratio with padding for heatmap model
                scale = img_size / max(ch, cw)
                nw, nh = int(cw * scale), int(ch * scale)
                dx, dy = (img_size - nw) // 2, (img_size - nh) // 2
                
                crop_res = F.interpolate(masked_crop, size=(nh, nw), mode='bilinear', align_corners=False)
                
                pad_l, pad_r = dx, img_size - nw - dx
                pad_t, pad_b = dy, img_size - nh - dy
                
                final_img = F.pad(crop_res, (pad_l, pad_r, pad_t, pad_b), value=0)
                
                # Apply ImageNet normalization only for legacy model
                final_img = (final_img - norm_mean) / norm_std
                
                # Prepare mask for peak extraction logic
                crop_mask_batch = crop_mask.unsqueeze(0)
                crop_mask_res = F.interpolate(crop_mask_batch, size=(nh, nw), mode='nearest')
                final_mask = F.pad(crop_mask_res, (pad_l, pad_r, pad_t, pad_b), value=0)
                all_masks_t.append(final_mask.squeeze(0).squeeze(0))
                
                scale_x = scale_y = scale

            all_tensors.append(final_img.squeeze(0))
            all_metadata.append({
                'box_idx': i, 
                'x1': x1, 'y1': y1, 
                'scale_x': scale_x, 'scale_y': scale_y, 
                'dx': dx, 'dy': dy, 
                'cw': cw, 'ch': ch
            })

        if not all_tensors:
            return []

        batch_t = torch.stack(all_tensors)
        
        with torch.no_grad():
            output = self.qpoint_model(batch_t)

        results = []
        if is_soft:
            # SoftArgmax returns [B, 2] normalized coordinates [X, Y]
            coords_cpu = output.detach().cpu().float().numpy()
            for i, meta in enumerate(all_metadata):
                pred_x_pct, pred_y_pct = coords_cpu[i, 0], coords_cpu[i, 1]
                # Scale back to original box coordinates
                lx = pred_x_pct * meta['cw']
                ly = pred_y_pct * meta['ch']
                gx = int(round(lx)) + meta['x1']
                gy = int(round(ly)) + meta['y1']
                gx = max(0, min(frame.shape[1] - 1, gx))
                gy = max(0, min(frame.shape[0] - 1, gy))
                results.append((gx, gy, meta['box_idx'], 1.0))
        else:
            # Heatmap returns [B, 1, H, W]
            heatmaps = output.squeeze(1)
            heatmaps_cpu    = heatmaps.detach().cpu().float().numpy()
            batch_masks_cpu = torch.stack(all_masks_t).detach().cpu().float().numpy()
            for i, meta in enumerate(all_metadata):
                pt_224, peak_conf = self._extract_qpoint(heatmaps_cpu[i], batch_masks_cpu[i])
                if pt_224 is None:
                    continue
                lx, ly = pt_224
                gx = int(round((lx - meta['dx']) / meta['scale_x'])) + meta['x1']
                gy = int(round((ly - meta['dy']) / meta['scale_y'])) + meta['y1']
                gx = max(0, min(frame.shape[1] - 1, gx))
                gy = max(0, min(frame.shape[0] - 1, gy))
                results.append((gx, gy, meta['box_idx'], peak_conf))

        return results

    def _effective_yolo_conf(self, conf_override=None):
        if conf_override is not None:
            return conf_override
        if self.class_conf:
            return min(self.conf, min(self.class_conf.values()))
        return self.conf

    def _resolve_imgsz(self, frame_or_frames, imgsz):
        if self.yolo_backend == "engine":
            return _normalise_imgsz(YOLO_WARMUP_IMGSZ or 1280)
        if imgsz is not None:
            return imgsz
        # Default to 640 to match standard YOLO/eval.py behavior.
        # Running at full-frame resolution (e.g. 1440) on a model trained at 640
        # can significantly degrade detection performance.
        return 640

    def _yolo_predict_kwargs(self):
        kwargs = {}
        if self.yolo_device is not None:
            kwargs["device"] = self.yolo_device
        if self.yolo_half:
            kwargs["half"] = True
        return kwargs

    def _pad_engine_batch(self, frames):
        batch_size = int(self.yolo_engine_batch_size or 1)
        real_count = len(frames)
        if real_count >= batch_size:
            return list(frames), real_count

        ref = frames[0]
        dummy = np.zeros_like(ref)
        padded = list(frames) + [dummy.copy() for _ in range(batch_size - real_count)]
        return padded, real_count

    def _run_yolo_batch(self, frames, classes_arg, conf, imgsz, retina_masks=True):
        run_imgsz = self._resolve_imgsz(frames, imgsz)
        results = self.yolo(
            list(frames),
            imgsz=run_imgsz,
            verbose=False,
            conf=conf,
            retina_masks=retina_masks,
            classes=classes_arg,
            **self._yolo_predict_kwargs(),
        )
        if results is None:
            return []
        if not isinstance(results, list):
            return [results]
        return results

    def _run_yolo_engine_frames(self, frames, classes_arg, conf, imgsz, retina_masks=True):
        batch_size = int(self.yolo_engine_batch_size or 1)
        all_results = []
        for start in range(0, len(frames), batch_size):
            chunk = list(frames[start:start + batch_size])
            padded_chunk, real_count = self._pad_engine_batch(chunk)
            results = self._run_yolo_batch(
                padded_chunk,
                classes_arg=classes_arg,
                conf=conf,
                imgsz=imgsz,
                retina_masks=retina_masks,
            )
            all_results.extend(results[:real_count])
        return all_results

    def _filter_yolo_result(self, result, conf_override=None):
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if result is None or boxes is None or len(boxes) == 0:
            return [], []

        raw_boxes = boxes
        raw_masks = masks.data.half() if masks is not None else [None] * len(boxes)

        # IoM suppression (keep highest-confidence non-overlapping boxes)
        if self.iom_enabled and self.iom_thresh < 1.0:
            order = sorted(
                range(len(raw_boxes)),
                key=lambda i: float(raw_boxes[i].conf[0]),
                reverse=True
            )

            keep = []
            for i in order:
                bi = raw_boxes[i].xyxy[0].cpu().numpy()
                cls_i = int(raw_boxes[i].cls[0].cpu().item())
                keep_i = True
                for j in keep:
                    bj = raw_boxes[j].xyxy[0].cpu().numpy()
                    if self._iom(bi, bj) <= self.iom_thresh:
                        continue
                    cls_j = int(raw_boxes[j].cls[0].cpu().item())
                    # Preserve target+avoid overlaps so avoid-veto logic can apply
                    # the avoid-class confidence threshold after burst grouping.
                    if (cls_i in self.avoid_classes) != (cls_j in self.avoid_classes):
                        continue
                    keep_i = False
                    break
                if keep_i:
                    keep.append(i)
        else:
            keep = list(range(len(raw_boxes)))

        # Per-class confidence filter (applied after IoM so we deduplicate first)
        if self.class_conf and conf_override is None:
            keep = [
                i for i in keep
                if float(raw_boxes[i].conf[0].cpu().item())
                   >= self.class_conf.get(int(raw_boxes[i].cls[0].cpu().item()), self.conf)
            ]

        filtered_boxes = [raw_boxes[i] for i in keep]
        filtered_masks = [raw_masks[i] for i in keep]
        return filtered_boxes, filtered_masks

    def _get_filtered_results(self, frame, classes_override=None, conf_override=None, imgsz=None):
        # Run YOLO on all relevant classes (target + avoid) so cross-class IoM
        # suppression can happen in _filter_yolo_result (highest conf wins).
        if classes_override == "all":
            target_override = "all"
            classes_arg = None
        else:
            target_override = _resolve_classes(None, classes_override) if classes_override is not None else None
            classes_arg = self._all_detect_classes(target_override)
            
        run_imgsz = self._resolve_imgsz(frame, imgsz)

        # Run at lowest effective threshold so per-class filtering happens
        # after IoM suppression rather than before it.
        if self.yolo_backend == "engine":
            results = self._run_yolo_engine_frames(
                [frame],
                classes_arg=classes_arg,
                conf=self._effective_yolo_conf(conf_override),
                imgsz=run_imgsz,
                retina_masks=True,
            )
        else:
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
        # filtered_boxes holds all-class results (for debug visualization).
        self.filtered_boxes, filtered_masks = self._filter_yolo_result(
            result,
            conf_override=conf_override,
        )
        return self.filtered_boxes, filtered_masks

    def _get_filtered_results_batch(self, frames, classes_override=None, conf_override=None, imgsz=None):
        if not frames:
            self.filtered_boxes = []
            return []

        if classes_override == "all":
            target_override = "all"
            classes_arg = None
        else:
            target_override = _resolve_classes(None, classes_override) if classes_override is not None else None
            classes_arg = self._all_detect_classes(target_override)

        run_imgsz = self._resolve_imgsz(frames, imgsz)
        if self.yolo_backend == "engine":
            results = self._run_yolo_engine_frames(
                list(frames),
                classes_arg=classes_arg,
                conf=self._effective_yolo_conf(conf_override),
                imgsz=run_imgsz,
                retina_masks=True,
            )
        else:
            results = self._run_yolo_batch(
                frames,
                classes_arg=classes_arg,
                conf=self._effective_yolo_conf(conf_override),
                imgsz=run_imgsz,
                retina_masks=True,
            )

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
        Used for sensitivity analysis after a survey burst. Counts target-class only."""
        target_override = _resolve_classes(None, classes_override) if classes_override is not None else None
        classes_arg = self._all_detect_classes(target_override)
        if self.yolo_backend == "engine":
            results = self._run_yolo_engine_frames(
                [frame],
                classes_arg=classes_arg,
                conf=conf_override,
                imgsz=1280,
                retina_masks=False,
            )
        else:
            results = self.yolo(
                frame, imgsz=1280, verbose=False,
                conf=conf_override, retina_masks=False, classes=classes_arg,
                **self._yolo_predict_kwargs(),
            )
        if not results or len(results[0].boxes) == 0:
            return 0
        return sum(
            1 for b in results[0].boxes
            if self._is_target(int(b.cls[0].cpu().item()), target_override)
        )

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
            if self.yolo_backend == "engine":
                batch_size = int(self.yolo_engine_batch_size or 1)
                frames = [dummy.copy() for _ in range(batch_size)]
                self._run_yolo_engine_frames(
                    frames,
                    classes_arg=self._all_detect_classes(),
                    conf=self._effective_yolo_conf(None),
                    imgsz=run_imgsz,
                    retina_masks=True,
                )
            else:
                self.yolo(
                    dummy,
                    imgsz=run_imgsz,
                    verbose=False,
                    conf=self._effective_yolo_conf(None),
                    retina_masks=True,
                    classes=self._all_detect_classes(),
                    **self._yolo_predict_kwargs(),
                )
        return time.perf_counter() - t0

    def detect_points(self, frame, classes_override=None, point_mode=None):
        all_boxes, all_masks = self._get_filtered_results(frame, classes_override=classes_override)
        target_override = _resolve_classes(None, classes_override) if classes_override is not None else None
        # Drop avoid-class boxes (cross-class IoM already suppressed most overlaps;
        # this removes any remaining avoid detections from the output).
        boxes, masks = self._filter_to_targets(all_boxes, all_masks, target_override)
        if not boxes:
            return []

        mode = _normalise_point_mode(point_mode, default="qpoint")
        use_qpoint = self.qpoint_model is not None and mode != "box_center" and (mode == "softargmax" or any(m is not None for m in masks))

        if not use_qpoint:
            return [
                (
                    int((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                    int((b.xyxy[0][1] + b.xyxy[0][3]) / 2)
                )
                for b in boxes
            ]

        coords = []
        for i, (gx, gy, box_idx, peak_conf) in enumerate(self._run_qpoints_batch(frame, boxes, masks, mode=mode)):
            if QPOINT_DEBUG:
                box = boxes[box_idx]
                yolo_conf = float(box.conf[0].cpu().item())
                cls_id    = int(box.cls[0].cpu().item())
                cls_name  = self.yolo.names.get(cls_id, str(cls_id))
                print(
                    f"  [{mode}] det {i}: {cls_name} "
                    f"yolo={yolo_conf:.2f} | "
                    f"{'peak=' if mode=='qpoint' else 'soft='}{peak_conf:.4f} | "
                    f"px=({gx},{gy})"
                )
            coords.append((gx, gy))
        return coords

    def detect_rich_points(self, frame, classes_override=None, point_mode=None):
        """Returns [{"point": (gx,gy), "cls": cls_id, "conf": yolo_conf}, ...]."""
        all_boxes, all_masks = self._get_filtered_results(frame, classes_override=classes_override)
        target_override = _resolve_classes(None, classes_override) if classes_override is not None else None
        boxes, masks = self._filter_to_targets(all_boxes, all_masks, target_override)
        if not boxes:
            return []

        mode = _normalise_point_mode(point_mode, default="qpoint")
        use_qpoint = self.qpoint_model is not None and mode != "box_center" and (mode == "softargmax" or any(m is not None for m in masks))

        if not use_qpoint:
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
            for gx, gy, box_idx, _ in self._run_qpoints_batch(frame, boxes, masks, mode=mode)
        ]

    def draw_detections(self, frame, boxes=None, points=None):
        """Draw YOLO boxes + qpoint markers on frame.

        Uses internal YOLO box objects (not the public dict format).
        Called by render_trial_video.py and internally during debug.
        Colours come from vision.visualization — edit there to restyle.
        """
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
            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR_RAW, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
            
            # Black background, white text for consistent readability
            ty = max(th + 4, y1 - 6)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
            cv2.putText(out, label, (x1 + 2, ty - 2), LABEL_FONT, LABEL_FONT_SCALE, (255, 255, 255), LABEL_THICKNESS)
            
            if i < len(points):
                px, py = int(points[i][0]), int(points[i][1])
                cv2.circle(out, (px, py), QPOINT_RADIUS, QPOINT_COLOR, -1)
                cv2.circle(out, (px, py), QPOINT_OUTLINE_RADIUS, QPOINT_OUTLINE_COLOR, 1)
        return out

    def draw_stable_detections(self, frame, stable_points):
        """Draw burst-stable cluster results.

        Delegates to vision.visualization.draw_stable_detections so that the
        dashboard and runtime always render detections identically.
        Edit vision/visualization.py to restyle colours, labels, or dot sizes.
        """
        return _vis_draw_stable(frame, stable_points, cls_names=self.yolo.names)

    def refine_one_heatmap(self, crop_pt, mode="qpoint"):
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

        qpoints = self._run_qpoints_batch(merged, [last_boxes[best_i]], [last_masks[best_i]], mode=mode)
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

        is_soft = getattr(self, "qpoint_is_softargmax", False)
        img_size = self.SOFT_IMG_SIZE if is_soft else self.TRAIN_SIZE

        norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1).half()
        norm_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1).half()

        img_t = torch.from_numpy(crop_img).to(self.device)
        img_t = img_t[:, :, [2, 1, 0]].permute(2, 0, 1).half() / 255.0

        if is_soft:
            # Squashing resize for softargmax
            final_img = F.interpolate(img_t.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False)
            # No normalization for softargmax
            dx = dy = 0
            scale_x = img_size / w
            scale_y = img_size / h
        else:
            # Aspect-ratio-preserving resize with padding for legacy model
            scale = img_size / max(h, w)
            nw, nh = int(w * scale), int(h * scale)
            dx = (img_size - nw) // 2
            dy = (img_size - nh) // 2
            pad_l = dx
            pad_r = img_size - nw - dx
            pad_t = dy
            pad_b = img_size - nh - dy

            resized = F.interpolate(img_t.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False)
            final_img = F.pad(resized, (pad_l, pad_r, pad_t, pad_b), value=0)
            
            # Apply normalization for legacy model
            final_img = (final_img - norm_mean) / norm_std
            scale_x = scale_y = scale

        with torch.no_grad():
            output = self.qpoint_model(final_img).squeeze(0)

        if is_soft:
            coords = output.detach().cpu().float().numpy()
            pred_x_pct, pred_y_pct = coords[0], coords[1]
            # Map percentages back to crop space
            gx = int(round(pred_x_pct * w))
            gy = int(round(pred_y_pct * h))
        else:
            heatmap = output.squeeze(0)
            heatmap_np = heatmap.detach().cpu().float().numpy()
            full_mask = np.ones((img_size, img_size), dtype=np.float32)
            pt_224, _ = self._extract_qpoint(heatmap_np, full_mask)
            if pt_224 is None:
                return None
            gx = int(round((pt_224[0] - dx) / scale_x))
            gy = int(round((pt_224[1] - dy) / scale_y))
            
        return (max(0, min(w - 1, gx)), max(0, min(h - 1, gy)))

    def return_burst_stable(
        self,
        frames,
        min_stable_views=2,
        group_iou_thresh=0.25,
        group_radius_px=None,
        classes_override=None,
        debug_label=None,
        imgsz=None,
        heatmap_final=True,
        point_mode=None,
    ):
        t_total = time.perf_counter()
        mode = _normalise_point_mode(point_mode, default="qpoint")
        self.last_burst_timing = {}
        
        if classes_override == "all":
            target_override = "all"
        else:
            target_override = _resolve_classes(None, classes_override) if classes_override is not None else None

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

        t_boxpoint = time.perf_counter()
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

            # Calculate average confidence for each class across the whole burst
            n_frames = len(frames)
            cls_totals = Counter()
            for c, f in zip(g["cls_votes"], g["confs"]):
                cls_totals[c] += f
            class_mean_conf = {int(c): float(total / n_frames) for c, total in cls_totals.items()}
            modal_cls = self._choose_group_class(class_mean_conf, target_override=target_override)
            mean_conf = class_mean_conf.get(int(modal_cls), float(np.mean(g["confs"])))

            stable.append({
                "point": (int(round(mean_pt[0])), int(round(mean_pt[1]))),
                "box":   mean_box,
                "views": g["views"],
                "cls":   modal_cls,
                "conf":  mean_conf,
                "all_confs": class_mean_conf,
                "point_source": "box_center",
            })

        # Suppress target detections dominated by a higher-confidence avoid-class
        # detection, then drop all non-target classes from the output.
        n_raw = len(stable)
        stable = self._suppress_avoid_overlaps(stable, target_override=target_override)
        stable = [s for s in stable if self._is_target(s["cls"], target_override)]
        stable.sort(key=lambda d: d["box"][0])

        group_dt = time.perf_counter() - t_group
        boxpoint_dt = time.perf_counter() - t_boxpoint
        suppressed = n_raw - len(stable)
        _debug(
            f"grouped {sum(len(d) for d in all_detections)} detection(s) "
            f"into {len(stable)} stable target(s)"
            + (f" ({suppressed} suppressed by avoid/class filter)" if suppressed else "")
            + f" in {group_dt:.2f}s"
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
        is_soft = (mode == "softargmax")
        if (
            heatmap_final
            and _uses_qpoint(mode)
            and self.qpoint_model is not None
            and stable
            and last_boxes
            and (last_masks or is_soft)
        ):
            t_qpoint = time.perf_counter()
            qpoints = self._run_qpoints_batch(merged, last_boxes, last_masks, mode=mode)
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
            "boxpoint_time_s": round(boxpoint_dt, 6),
            "grouping_time_s": round(group_dt, 6),
            "merge_time_s": round(merge_dt, 6),
            "qpoint_time_s": round(qpoint_dt, 6),
            "total_time_s": round(total_dt, 6),
            "point_mode": mode,
            "heatmap_final": bool(heatmap_final and _uses_qpoint(mode)),
            "qpoint_batched": bool(self.qpoint_model is not None),
            "qpoint_candidate_count": int(len(last_boxes) if last_boxes is not None else 0),
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
        target_classes=TARGET_CLASSES,
        avoid_classes=AVOID_CLASSES,
        iom_enabled=True,
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
        self.target_classes = target_classes
        self.avoid_classes = avoid_classes
        self.window_name = "AI Detector - Stereo Pair"

        core_kwargs = dict(
            conf=conf,
            iom_thresh=iom_thresh,
            target_classes=target_classes,
            avoid_classes=avoid_classes,
            yolo_backend=self.yolo_backend,
            yolo_device=YOLO_DEVICE,
            yolo_half=YOLO_HALF,
            iom_enabled=iom_enabled,
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
