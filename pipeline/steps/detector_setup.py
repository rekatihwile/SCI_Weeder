"""Detector setup step helpers."""

import importlib.util
import time
from pathlib import Path


def _apply_nms_patch():
    repo_root = Path(__file__).resolve().parents[2]
    patch_path = repo_root / "bringup" / "_nms_patch.py"
    if not patch_path.exists():
        return
    spec = importlib.util.spec_from_file_location("_nms_patch", patch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def build_and_warm_detector():
    _apply_nms_patch()

    from config import AI_CONFIDENCE, AI_DISPLAY_SCALE
    from vision.detectors.ai_detector import AIDetector

    t_model = time.perf_counter()
    detector = AIDetector(
        display_scale=AI_DISPLAY_SCALE,
        conf=AI_CONFIDENCE,
    )
    model_load_time_s = round(time.perf_counter() - t_model, 3)

    warmup_info = {}
    if hasattr(detector, "warmup"):
        warmup_info = detector.warmup()

    return detector, warmup_info, model_load_time_s
