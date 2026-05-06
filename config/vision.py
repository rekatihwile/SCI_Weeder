"""
config/vision.py — AI detector model selection and inference parameters.

KNOBS most often changed:
  AI_CONFIDENCE     — detection threshold (lower = catch more plants, higher = fewer false positives)
  DEFAULT_MODEL     — which YOLO weight file to use (see MODEL_MAP)
  YOLO_DEVICE       — "cuda:0" for Jetson GPU, "cpu" for offline testing
  TARGET_CLASSES    — class IDs the laser should target (list[int] or None for all)
  AVOID_CLASSES     — class IDs to detect but never target (crops/non-weeds)
  QPOINT_DEBUG      — True to log per-detection qpoint output
"""

from .paths import CV_WEIGHTS_DIR

# =============================================================================
# KNOBS — model and confidence
# =============================================================================

MODEL_MAP = {
    "plastic_nano": "26_plastic_nano.pt",
    "targeting_tall_plastic": "new_best_targeting_tall_plastic.pth",
}
# Used by vision/detectors/ai_detector.py and hardware/cameras.py debug mode.
# You can also set DEFAULT_MODEL/DEFAULT_QPOINT_MODEL directly to a filename.

DEFAULT_MODEL        = "plastic_nano"
DEFAULT_MODEL_PT     = DEFAULT_MODEL
DEFAULT_MODEL_ENGINE = None
DEFAULT_QPOINT_MODEL = "targeting_tall_plastic"

YOLO_BACKEND      = "auto"
# Used by AIDetector. Options: "pt", "engine", or "auto".

USE_TENSORRT_ENGINE = False
# Used by AIDetector. True prefers DEFAULT_MODEL_ENGINE when that file exists.

YOLO_DEVICE = "cuda:0"
# Used by AIDetector YOLO inference. Use 0, "cuda:0", "cpu", or "auto".

YOLO_HALF = True
# Used by AIDetector YOLO inference on CUDA.

YOLO_WARMUP       = True
YOLO_WARMUP_IMGSZ = 640
YOLO_WARMUP_ITERS = 3
# Used by main.py/AIDetector before live survey timing starts.

AI_CONFIDENCE = 0.40
# Used by AIDetector. Turn UP for fewer false positives; DOWN if plants are missed.

AI_CLASS_CONFIDENCE = None
# Used by AIDetector. Per-class overrides beat AI_CONFIDENCE.
# Turn a class UP to be stricter for that class only.

AI_IOM_THRESHOLD = 0.80
# Used by AIDetector. Turn UP to merge only very-overlapping masks; DOWN to merge more.

AI_TARGET_CLASS = 1
# Legacy alias kept for backward compat (ros2_ws/cv_node.py, etc.).
# Prefer TARGET_CLASSES / AVOID_CLASSES below for new code.

TARGET_CLASSES = [1]
# Class IDs the laser should target. YOLO is run on TARGET_CLASSES + AVOID_CLASSES
# together so avoid detections can suppress overlapping targets.
# None = all classes (minus AVOID_CLASSES), list[int] = explicit set.

AVOID_CLASSES = []
# Class IDs to detect but NEVER target (e.g., crop rows, ornamentals).
# When an avoid-class detection overlaps a target detection with higher
# confidence (IoM >= AI_IOM_THRESHOLD), the target detection is suppressed.

AI_DISPLAY_SCALE     = 1.0
MANUAL_DISPLAY_SCALE = 0.75
# Used by main.py detector construction. Turn UP for larger debug windows.

QPOINT_DEBUG = True
# Used by AIDetector. False quiets per-detection qpoint debug logs.
