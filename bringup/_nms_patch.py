"""
bringup/_nms_patch.py
----------------------
Monkey-patch torchvision.ops.nms with a pure-PyTorch implementation.
Import this BEFORE any ultralytics/YOLO code to avoid the "Couldn't load
custom C++ ops" crash on this Jetson (torchvision C++ extensions are broken).

Usage:
    import bringup._nms_patch  # or: from pathlib import Path; exec(...)
    # or simply: import _nms_patch  (when CWD is bringup/)
"""

import torch
import torchvision.ops as _tvops


def _nms_pytorch(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        order = order[1:]
        xx1 = torch.clamp(x1[order], min=float(x1[i]))
        yy1 = torch.clamp(y1[order], min=float(y1[i]))
        xx2 = torch.clamp(x2[order], max=float(x2[i]))
        yy2 = torch.clamp(y2[order], max=float(y2[i]))
        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        iou = inter / (areas[i] + areas[order] - inter + 1e-6)
        order = order[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


# Apply patch
_tvops.nms = _nms_pytorch
print("[NMS PATCH] torchvision.ops.nms patched with pure-PyTorch fallback.")
