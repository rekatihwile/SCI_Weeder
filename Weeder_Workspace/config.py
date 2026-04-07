"""
Central configuration file for the Weeder system.
Only loads constants and JSON configs.
No hardware logic should live here.
"""

import json
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

HARDWARE_CONFIG_PATH = BASE_DIR / "params/hardware_config.json"
CAMERA_CONFIG_PATH = BASE_DIR / "params/camera_config.json"

# Model lookup for weights in params/
MODEL_MAP = {
    "yolo_w_kale": "yolo_w_kale.pt",
    "yolo_weed": "yolo_weed.pt",
    "sniper": "sniper.pt",
    "best_targeting_v3": "new_best_targeting_v3.pth",
    "best_pigweed_145": "best_pigweed_145.pt",
    # New models
    "plastic_nano": "26_plastic_nano.pt",
    "targeting_tall_plastic": "new_best_targeting_tall_plastic.pth",
}

DEFAULT_MODEL = "plastic_nano"
DEFAULT_QPOINT_MODEL = "targeting_tall_plastic"
CV_PIPELINE_MODE = "two_stage"

# ---------------------------------------------------
# TRAINING PHOTO COLLECTION
# ---------------------------------------------------

TRAINING_PHOTOS_DIR = BASE_DIR / "training_photos"

WORKSPACE_X_MIN = 0.0
WORKSPACE_X_MAX = 450.0
WORKSPACE_Y_MIN = 0.0
WORKSPACE_Y_MAX = 440.0

X_SUBSECTIONS = 4
Y_SUBSECTIONS = 3

PHOTO_SETTLE_SEC = 1.5

# ---------------------------------------------------
# LOAD HARDWARE CONFIG
# ---------------------------------------------------

if HARDWARE_CONFIG_PATH.exists():
    with open(HARDWARE_CONFIG_PATH, "r") as f:
        _hardware_cfg = json.load(f)
else:
    raise FileNotFoundError("hardware_config.json not found")

GRBL_PORT = _hardware_cfg["serial"]["grbl_port"]

LEFT_CAMERA_INDEX = _hardware_cfg["cameras"]["left"]["index"]
RIGHT_CAMERA_INDEX = _hardware_cfg["cameras"]["right"]["index"]

_runtime_cfg = _hardware_cfg.get("runtime", {})
_default_has_display = bool(
    os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
    or sys.platform.startswith("win")
)
HAS_DISPLAY = bool(_runtime_cfg.get("has_display", _default_has_display))
HEADLESS = bool(_runtime_cfg.get("headless", not HAS_DISPLAY))
UI_MODE = "headless" if HEADLESS else "window"
DISPLAY_BACKEND = _runtime_cfg.get(
    "display_backend",
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or ""
)
AUTO_MODE = False

# ---------------------------------------------------
# CAMERA SETTINGS
# ---------------------------------------------------

if CAMERA_CONFIG_PATH.exists():
    with open(CAMERA_CONFIG_PATH, "r") as f:
        CAMERA_SETTINGS = json.load(f)
else:
    CAMERA_SETTINGS = {}

# ---------------------------------------------------
# CAMERA PARAMETERS
# ---------------------------------------------------

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

TARGET_Y_L = FRAME_HEIGHT // 2
TARGET_Y_R = FRAME_HEIGHT // 2

# ---------------------------------------------------
# CALIBRATION & TRIANGULATION
# ---------------------------------------------------

CALIB_NPZ_PATH = BASE_DIR / "Triangulation/stereo_checkerboard_fisheye_calib.npz"
RECT_NPZ_PATH = BASE_DIR / "Triangulation/stereo_checkerboard_fisheye_rectify_maps.npz"

CALIBRATION_EXPECTS_UNFLIPPED = False

TRI_SIGN_X = 1.0
TRI_SIGN_Y = -1.0

LASER_OFFSET_X_MM = 33.0
LASER_OFFSET_Y_MM = 0.0

TRI_X_GAIN = 1.0
TRI_Y_GAIN = 1.0

# Use pixel error correction in triangulation to compensate for lens distortion and other non-linearities

USE_PIXEL_ERROR_CORRECTION = True
PIXEL_ERROR_MODEL_PATH = BASE_DIR / "params/stereo_pixel_error_model.json"
# ---------------------------------------------------
# MACHINE POSITIONS
# ---------------------------------------------------

SURVEY_POS_X = 200.0
SURVEY_POS_Y = 200.0

# ---------------------------------------------------
# CONTROL MODES
# ---------------------------------------------------

DETECTOR_MODE = "ai"  # "manual", "ai", or "off"
COARSE_MOVE_MODE = "triangulation"
FINE_ALIGN_MODE = "pixel_pd"

TRIANGULATION_ONLY_MODE = False
SHOW_TRIANGULATION_PLOT = True
SHOW_MATCH_DEBUG_WINDOW = HAS_DISPLAY
SAVE_MATCH_DEBUG_IMAGE = True

# ---------------------------------------------------
# DETECTOR / AI SETTINGS
# ---------------------------------------------------

MANUAL_DISPLAY_SCALE = .75

AI_DISPLAY_SCALE = 2.0
AI_BURST_SIZE = 5
AI_MIN_STABLE_VIEWS = 3
AI_CONFIDENCE = .50
AI_IOM_THRESHOLD = 0.80

# ---------------------------------------------------
# CLASS SELECTION
# ---------------------------------------------------
# AI_TARGET_CLASS  — used for fine-align detection (and survey if SURVEY_TARGET_CLASSES is None)
#   int        →  single class,  e.g. 0
#   list[int]  →  multiple,      e.g. [0, 2]
#   None       →  all classes
#
# SURVEY_TARGET_CLASSES  — override for the survey burst only
#   None  →  use AI_TARGET_CLASS (default)
#   list  →  e.g. [0, 1, 2] to survey for all plant types even if fine-align targets one
#
# Quick class lookup:
#   python -c "from ultralytics import YOLO; m=YOLO('params/26_plastic_nano.pt'); print(m.names)"
AI_TARGET_CLASS       = 1     # fine-align class filter
SURVEY_TARGET_CLASSES = None  # None = use AI_TARGET_CLASS; set e.g. [0,1,2] to survey all

# ---------------------------------------------------
# GLOBAL SURVEY SETTINGS
# ---------------------------------------------------

# IoU threshold for grouping detections across burst frames during survey.
# 0.15 is permissive enough for boxes that shift slightly between frames.
SURVEY_BOX_IOU_THRESH = 0.15

SURVEY_BURST_COUNT = 10
SURVEY_MIN_HITS = 8
SURVEY_CLUSTER_RADIUS_PX = 30.0

# ---------------------------------------------------
# FINE ALIGN SETTINGS
# ---------------------------------------------------

FINE_ALIGN_CROP_SCALE = 0.5

FINE_ALIGN_LK_WIN_SIZE = 31
FINE_ALIGN_LK_MAX_LEVEL = 3

FINE_ALIGN_KP_X = 12.5
FINE_ALIGN_KD_X = 2.5
FINE_ALIGN_KP_Y = 10.0
FINE_ALIGN_KD_Y = 2.5

FINE_ALIGN_STEP_MM = 0.001
FINE_ALIGN_DEADZONE_PX = 3.0
FINE_ALIGN_MAX_JOG_MM = 10.0
FINE_ALIGN_FEED = 5000

FINE_ALIGN_BURST_COUNT = 4
FINE_ALIGN_MIN_HITS = 3
FINE_ALIGN_CLUSTER_RADIUS_PX = 12.0

FINE_ALIGN_MAX_TIME_SEC = 15.0
FINE_ALIGN_SETTLE_FRAMES = 10

# ---------------------------------------------------
# STRIKE SETTINGS
# ---------------------------------------------------

STRIKE_PATTERN = "pulse"
LASER_FIRE_POWER = 1000
LASER_FIRE_DURATION_SEC = 2
LASER_ARM_DELAY_SEC = 0.100
LASER_TRIGGER_FEED = 100

# ---------------------------------------------------
# PLATFORM SETTINGS
# ---------------------------------------------------

IS_WINDOWS = sys.platform.startswith("win")