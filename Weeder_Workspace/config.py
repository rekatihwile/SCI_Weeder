"""
config.py

Central configuration file for the Weeder system.
Only loads constants and JSON configs.
No hardware logic should live here.
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

HARDWARE_CONFIG_PATH = BASE_DIR / "params/hardware_config.json"
CAMERA_CONFIG_PATH   = BASE_DIR / "params/camera_config.json"


# ---------------------------------------------------
# LOAD HARDWARE CONFIG
# ---------------------------------------------------

if HARDWARE_CONFIG_PATH.exists():
    with open(HARDWARE_CONFIG_PATH, "r") as f:
        _hardware_cfg = json.load(f)
else:
    raise FileNotFoundError("hardware_config.json not found")


GRBL_PORT = _hardware_cfg["serial"]["grbl_port"]

LEFT_CAMERA_INDEX  = _hardware_cfg["cameras"]["left"]["index"]
RIGHT_CAMERA_INDEX = _hardware_cfg["cameras"]["right"]["index"]


# ---------------------------------------------------
# CAMERA SETTINGS
# ---------------------------------------------------

if CAMERA_CONFIG_PATH.exists():
    with open(CAMERA_CONFIG_PATH, "r") as f:
        CAMERA_SETTINGS = json.load(f)
else:
    CAMERA_SETTINGS = {}


# ---------------------------------------------------
# CAMERA PARAMETERS OV5640
# ---------------------------------------------------

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

TARGET_Y_L = FRAME_HEIGHT // 2
TARGET_Y_R = FRAME_HEIGHT // 2

# ---------------------------------------------------
# CALIBRATION & TRIANGULATION
# ---------------------------------------------------

CALIB_NPZ_PATH = BASE_DIR / "params/stereo_charuco_fisheye_calib.npz"
RECT_NPZ_PATH  = BASE_DIR / "params/stereo_fisheye_rectify_maps.npz"

# Runtime frames are rotated 180 deg in cameras.py.
# Set this True if your calibration files were made from the raw, unrotated camera images.
CALIBRATION_EXPECTS_UNFLIPPED = False

# Coarse triangulation tuning
TRI_SIGN_X = -1.0
TRI_SIGN_Y = -1.0

LASER_OFFSET_X_MM = 33.0
LASER_OFFSET_Y_MM = 0.0

TRI_X_GAIN = 1.0
TRI_Y_GAIN = 1.0

USE_AFFINE_CORRECTION = True
AFFINE_X_COEFFS = [-1.192043821870919, 1.001531761830695, 0.012082374519007066]
AFFINE_Y_COEFFS = [10.508343050916622, -0.09675010667374007, 0.9966484990708326]

# ---------------------------------------------------
# MACHINE POSITIONS
# ---------------------------------------------------

# Starting location for weed survey
SURVEY_POS_X = 200.0
SURVEY_POS_Y = 200.0


# ---------------------------------------------------
# CONTROL MODES
# ---------------------------------------------------

# Which detector to use
DETECTOR_MODE = "ai"
# options: "manual", "ai"

# Coarse move strategy
COARSE_MOVE_MODE = "triangulation"
# options: "triangulation", "pixel_pd"

# Fine alignment strategy
FINE_ALIGN_MODE = "pixel_pd"

# Laser strike pattern
STRIKE_PATTERN = "spiral"


# ---------------------------------------------------
# PLATFORM SETTINGS
# ---------------------------------------------------

IS_WINDOWS = sys.platform.startswith("win")