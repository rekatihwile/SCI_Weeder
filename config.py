"""
Runtime configuration for LaserWeeder.

Keep this file boring on purpose: constants, paths, and JSON loading only.
Hardware camera indices and serial ports live in params/hardware/*.json.
"""

import json
import os
import sys
from pathlib import Path


# =============================================================================
# Paths Loaded By Many Modules
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

_HARDWARE_CONFIG_PATH = BASE_DIR / "params/hardware/hardware_config.json"
_CAMERA_CONFIG_PATH = BASE_DIR / "params/hardware/camera_config.json"

CV_WEIGHTS_DIR = BASE_DIR / "params/cv_weights"
TRAINING_PHOTOS_DIR = BASE_DIR / "training_photos"
TRIAL_RECORDINGS_DIR = BASE_DIR / "trial_recordings"


# =============================================================================
# Hardware JSON
# =============================================================================

if not _HARDWARE_CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing hardware config: {_HARDWARE_CONFIG_PATH}")

with open(_HARDWARE_CONFIG_PATH, "r") as f:
    _hardware_cfg = json.load(f)

GRBL_PORT = _hardware_cfg["serial"]["grbl_port"]  # Used by main.py and tools to open GRBL.
LEFT_CAMERA_INDEX = _hardware_cfg["cameras"]["left"]["index"]  # Used by hardware/cameras.py.
RIGHT_CAMERA_INDEX = _hardware_cfg["cameras"]["right"]["index"]  # Used by hardware/cameras.py.

if _CAMERA_CONFIG_PATH.exists():
    with open(_CAMERA_CONFIG_PATH, "r") as f:
        CAMERA_SETTINGS = json.load(f)  # Used by hardware/cameras.py for exposure/gain/WB.
else:
    CAMERA_SETTINGS = {}


# =============================================================================
# Operator Toggles
# =============================================================================

HOMING = False
# Used by main.py. True = home the gantry at startup.

FIRE = False
# Used by control/strike.py. Keep False for dry runs; True actually pulses laser.

RECORD_TRIAL = True
# Used by main.py/hardware/cameras.py. False disables trial recording.
# When True, the default path records lightweight raw stereo frames plus a manifest.

TRIANGULATION_ONLY_MODE = False
# Used by main.py. True skips fine-align/strike and logs coarse triangulation results only.

FULL_AUTO = True
# Used by main.py. True skips all operator input prompts so the machine runs unattended.

AUTO_MODE = False
# Used by hardware/cameras.py. True lets cameras auto exposure/WB after startup settings.

DETECTOR_MODE = "ai"
# Used by main.py/fine_align.py. Options: "ai" or "manual".


# =============================================================================
# Display And Debug Views
# =============================================================================

_runtime_cfg = _hardware_cfg.get("runtime", {})


def _probe_display():
    """Return True only when a graphical display is set AND actually reachable.

    Checking the env var alone isn't enough: SSH sessions on the Jetson inherit
    DISPLAY=:1 from .bashrc even when no X11 forwarding is active.  We probe the
    X socket so that cv2/Qt GUI code is only enabled when the server truly answers.
    """
    import re as _re
    import socket as _socket

    if sys.platform.startswith("win"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True

    display = os.environ.get("DISPLAY", "")
    if not display:
        return False

    # Local Unix-socket display  e.g. ":0"  ":1"  ":0.0"
    m = _re.match(r"^:(\d+)", display)
    if m:
        sock_path = f"/tmp/.X11-unix/X{m.group(1)}"
        try:
            s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(sock_path)
            s.close()
            return True
        except OSError:
            return False

    # TCP display — covers X11 forwarding ("localhost:10.0") and remote X
    m = _re.match(r"^([^:]+):(\d+)", display)
    if m:
        host, num = m.group(1), int(m.group(2))
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((host, 6000 + num))
            s.close()
            return True
        except OSError:
            return False

    return False


_env_has_display = _probe_display()

# Respect the JSON config, but only if the display probe succeeded.
# This prevents Qt from crashing when has_display=true is set for local use but
# the process is launched over SSH without X11 forwarding.
HAS_DISPLAY = bool(_runtime_cfg.get("has_display", _env_has_display)) and _env_has_display
# Used by UI/OpenCV windows. False for headless Jetson/SSH sessions.

# When headless, suppress any stray cv2 GUI call before it reaches Qt.
# Note: this cv2 build only ships libqxcb.so (no offscreen plugin), so the env
# var alone cannot save us — the real guard is HAS_DISPLAY=False throughout the
# codebase.  Setting it anyway silences secondary Qt warnings on other systems.
if not HAS_DISPLAY and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

SHOW_TRIANGULATION_PLOT = False
# Used by main.py. False skips the workspace overview plot after survey.

SHOW_MATCH_DEBUG_WINDOW = False
# Used by main.py. False keeps the survey match debug image off-screen.

SAVE_MATCH_DEBUG_IMAGE = False
# Used by main.py. False stops writing triangulation_debug_view.png.


# =============================================================================
# Camera Frames And Workspace
# =============================================================================

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# Used throughout calibration, detection, matching, and recording.

WORKSPACE_X_MIN = 0.0
WORKSPACE_X_MAX = 450.0
WORKSPACE_Y_MIN = 0.0
WORKSPACE_Y_MAX = 440.0
# Used by coarse_move.py, fine_align.py, workspace_plot.py, and grid_capture.py.
# Shrink these if the gantry gets too close to walls.

IS_WINDOWS = sys.platform.startswith("win")
# Used by hardware/cameras.py to choose the OpenCV backend.


# =============================================================================
# Calibration And Triangulation
# =============================================================================

CALIB_NPZ_PATH = BASE_DIR / "params/calibration/stereo_checkerboard_fisheye_calib.npz"
RECT_NPZ_PATH = BASE_DIR / "params/calibration/stereo_checkerboard_fisheye_rectify_maps.npz"
# Used by control/coarse_move.py for stereo triangulation.

CALIBRATION_EXPECTS_UNFLIPPED = False
# Used by coarse_move.py. Change only if the calibration images were not camera-flipped.

TRI_SIGN_X = 1.0
TRI_SIGN_Y = -1.0
# Used by coarse_move.py. Flip sign if triangulated motion goes the wrong direction.

TRI_X_GAIN = 1.0
TRI_Y_GAIN = 1.0
# Used by coarse_move.py. Turn UP if triangulated moves undershoot; DOWN if they overshoot.

LASER_OFFSET_X_MM = 33.0
LASER_OFFSET_Y_MM = 0.0
# Used by coarse_move.py. Physical laser offset from stereo midpoint, in mm.

USE_PIXEL_ERROR_CORRECTION = True
PIXEL_ERROR_MODEL_PATH = BASE_DIR / "params/calibration/stereo_pixel_error_model.json"
# Used by coarse_move.py. False disables the learned stereo pixel correction.


# =============================================================================
# AI Detector
# =============================================================================

MODEL_MAP = {
    "plastic_nano": "26_plastic_nano.pt",
    "targeting_tall_plastic": "new_best_targeting_tall_plastic.pth",
}
# Used by vision/detectors/ai_detector.py and hardware/cameras.py debug mode.
# You can also set DEFAULT_MODEL/DEFAULT_QPOINT_MODEL directly to a filename.

DEFAULT_MODEL = "plastic_nano"
DEFAULT_MODEL_PT = DEFAULT_MODEL
DEFAULT_MODEL_ENGINE = None
DEFAULT_QPOINT_MODEL = "targeting_tall_plastic"

YOLO_BACKEND = "auto"
# Used by AIDetector. Options: "pt", "engine", or "auto".

USE_TENSORRT_ENGINE = False
# Used by AIDetector. True prefers DEFAULT_MODEL_ENGINE when that file exists.

YOLO_DEVICE = "cuda:0"
# Used by AIDetector YOLO inference. Use 0, "cuda:0", "cpu", or "auto".

YOLO_HALF = True
# Used by AIDetector YOLO inference on CUDA.

YOLO_WARMUP = True
YOLO_WARMUP_IMGSZ = 640
YOLO_WARMUP_ITERS = 3
# Used by main.py/AIDetector before live survey timing starts.

AI_CONFIDENCE = 0.60
# Used by AIDetector. Turn UP for fewer false positives; DOWN if plants are missed.

AI_CLASS_CONFIDENCE = {0: 0.10, 1: 0.10, 2: 0.10}
# Used by AIDetector. Per-class overrides beat AI_CONFIDENCE.
# Turn a class UP to be stricter for that class only.

AI_IOM_THRESHOLD = 0.80
# Used by AIDetector. Turn UP to merge only very-overlapping masks; DOWN to merge more.

AI_TARGET_CLASS = 2
# Used by AIDetector for default class filtering.
# None = all classes, int = one class, list[int] = several classes.

AI_DISPLAY_SCALE = 1.0
MANUAL_DISPLAY_SCALE = 0.75
# Used by main.py detector construction. Turn UP for larger debug windows.

QPOINT_DEBUG = True
# Used by AIDetector. False quiets per-detection qpoint debug logs.


# =============================================================================
# State-Specific CV Selection
# =============================================================================

OVERRIDE_BURST_NUMBER = True
OVERRIDE_BURST_COUNT = 1
# Used by main.py/fine_align.py. True forces every state to use OVERRIDE_BURST_COUNT.

OVERRIDE_POINT_MODE = False
OVERRIDE_POINT_MODE_VALUE = "box_center"
# Options: "box_center", "qpoint", "heatmap". True forces every CV state to this mode.

SURVEY_POINT_MODE = "box_center"
FINE_ALIGN_REID_POINT_MODE = "box_center"
FINAL_SNAP_POINT_MODE = "qpoint"
# Survey/Re-ID defaults stay fast. Final snap may use qpoint only when snap is enabled.


# =============================================================================
# Global Survey Burst
# =============================================================================

SURVEY_POS_X = 200.0
SURVEY_POS_Y = 200.0
# Used by main.py. Gantry pose where the global survey burst is captured.

SURVEY_FRAME_WIDTH = None
SURVEY_FRAME_HEIGHT = None
# Used by coarse_move.py. Set both to higher resolution for HD survey.
# Higher values can improve detection but cost camera switch/settling time.

SURVEY_BURST_COUNT = 5
# Used by main.py/coarse_move.py. Turn UP for more stable survey detections; slower.

SURVEY_MIN_HITS = 1
# Used by main.py/coarse_move.py. Turn UP to require repeat detections; DOWN to catch weak plants.

SURVEY_CLUSTER_RADIUS_PX = 10.0
# Used by coarse_move.py. Turn UP if burst detections jitter; DOWN to split nearby plants.

SURVEY_YOLO_IMGSZ = None
# Used by coarse_move.py survey burst. None = use actual frame size, no requested upscaling.
# Set an int/tuple only if you intentionally want YOLO resizing.

SURVEY_CROP_HALF_PX = 352
# Used by coarse_move.py. If set (int), survey frames are cropped to a centred square of
# 2*SURVEY_CROP_HALF_PX before YOLO — same approach as re-ID, much faster than full-frame.
# E.g. 350 → 700×700 crop. None = full frame (default, safe, but slow).

SURVEY_TARGET_CLASSES = None
# Used by main.py/coarse_move.py. None uses AI_TARGET_CLASS; list[int] overrides survey only.

SURVEY_CONF_SENSITIVITY_DEBUG = False
# Used by coarse_move.py. True runs extra YOLO passes after survey; very slow/noisy.


# =============================================================================
# Stereo Matching
# =============================================================================

STEREO_MATCH_MIN_DISPARITY_PX = 5.0
STEREO_MATCH_MAX_DISPARITY_PX = 250.0
# Used by vision/matching.py. Widen only if true stereo pairs are being rejected.

STEREO_MATCH_MAX_Y_DIFF_PX = 25.0
# Used by vision/matching.py. Turn UP only if valid left/right pairs are rejected.
# Keep LOW: large y differences mean bad epipolar geometry or a wrong plant match.

STEREO_MATCH_RADIUS_PX = 60.0
# Used by point-only constellation matching. Turn UP if no boxes are available and true pairs are missed.

STEREO_MATCH_MIN_SCORE = 0.1
# Used by vision/matching.py. Turn UP to reject weak constellation matches.

STEREO_MATCH_MIN_BOX_IOU = 0.15
STEREO_MATCH_IOU_WEIGHT = 0.50
# Used by box-IoU constellation matching. Turn UP IoU weight to trust box geometry more.


# =============================================================================
# Fine Align And Re-ID
# =============================================================================

FINE_ALIGN_CROP_SCALE = 1.0
# Used by fine_align.py. Center crop used for Re-ID acceptance and LK tracking.
# Turn UP to allow targets farther from center; DOWN to keep fine-align tighter.

FINE_ALIGN_BURST_COUNT = 1
# Used by fine_align.py Re-ID. Turn UP for more stable Re-ID; slower.

FINE_ALIGN_REID_BURST_COUNT = FINE_ALIGN_BURST_COUNT
# Used by fine_align.py Re-ID. Preferred state-specific name; old name remains as an alias.

FINAL_SNAP_BURST_COUNT = 1
# Used by fine_align.py final snap. Values >1 average snap crops before qpoint.

FINE_ALIGN_MIN_HITS = 1
# Used by fine_align.py Re-ID. Turn UP to require repeat detections across burst frames.

FINE_ALIGN_CLUSTER_RADIUS_PX = 35.0
# Used by fine_align.py Re-ID. Turn UP if Re-ID detections jitter between burst frames.

FINE_ALIGN_REID_CROP_HALF_PX = 128
# Used by fine_align.py Re-ID burst crop. Turn UP to search a wider center square.

FINE_ALIGN_REID_STEREO_DISP_PX = 0
# Historical alias for the old asymmetric Re-ID crop. Re-ID now uses the same
# centered crop in both cameras, so this remains 0 for compatibility.

FINE_ALIGN_REID_YOLO_IMGSZ = None
# Used by fine_align.py Re-ID YOLO. None = use actual Re-ID crop size, no requested upscaling.
# Set an int/tuple only if you intentionally want YOLO resizing.

FINE_ALIGN_REID_MAX_Y_DIFF_PX = STEREO_MATCH_MAX_Y_DIFF_PX
# Used by fine_align.py after matching.py. Turn DOWN to reject more slanted stereo pairs.

FINE_ALIGN_REID_MIN_DISPARITY_PX = 10.0
FINE_ALIGN_REID_MAX_DISPARITY_PX = 500.0
# Used by fine_align.py after matching.py. Widen only if valid stereo pairs are rejected.

FINE_ALIGN_REID_MAX_PD_ERROR_PX = 150.0
# Used by fine_align.py candidate ranking. This is for a LEFT+RIGHT stereo pair average,
# not individual points. Turn DOWN to require less fine-align travel.

FINE_ALIGN_REID_EPIPOLAR_TOL_MULT = 3.0
# Used by fine_align.py. Multiplier on the survey-fitted epipolar slope std dev.
# Survey std is tight (measured across many pairs); re-ID individual pairs deviate more due
# to depth variation and gantry-position effects.  3× gives ±3σ from survey mean.
# A minimum floor of 0.15 is applied in code regardless of this multiplier.

FINE_ALIGN_REID_MAX_TRI_DIST_MM = 20.0
# Used by fine_align.py. Hard cutoff on re-ID candidate tri_dist from planned coarse position.
# Good re-ID picks are <10mm. This blocks bad fallbacks (plants 40+mm away) when the
# correct candidate was filtered out, preventing cascade duplicate-rejection failures.

FINE_ALIGN_LK_WIN_SIZE = 31
FINE_ALIGN_LK_MAX_LEVEL = 3
# Used by fine_align.py optical flow. Larger tracks bigger motion but can drift more.

FINE_ALIGN_KP_X = 15.0
FINE_ALIGN_KD_X = 2.5
FINE_ALIGN_KP_Y = 20.0
FINE_ALIGN_KD_Y = 2.5
# Used by fine_align.py PD control. Turn UP Kp for stronger correction; KD damps changes.

FINE_ALIGN_STEP_MM = 0.001
# Used by fine_align.py. Pixel-error to mm jog scale. Turn UP for larger jogs per pixel.

FINE_ALIGN_DEADZONE_PX = 2.5
# Used by fine_align.py. Turn UP to accept looser locks; DOWN for tighter locks.

FINE_ALIGN_MAX_JOG_MM = 10.0
# Used by fine_align.py. Safety cap on one fine-align jog.

FINE_ALIGN_FEED = 5000
# Used by fine_align.py. Gantry feed rate for fine-align jogs.

FINE_ALIGN_MAX_TIME_SEC = 15.0
# Used by fine_align.py. Turn UP to give difficult targets more time before timeout.

FINE_ALIGN_SETTLE_FRAMES = 10
# Used by main.py/fine_align.py. Turn UP to require a longer stable lock before firing.

FINE_ALIGN_SNAP_SETTLE_FRAMES = 4
# Used by fine_align.py. Settle frames for the final snap re-lock pass (after qpoint snap).
# Shorter than FINE_ALIGN_SETTLE_FRAMES since the snap corrects drift and LK just confirms.

FINE_ALIGN_ENABLE_SNAP = False
# Used by fine_align.py. False skips the final qpoint heatmap snap and fires from LK/PD lock.

FINE_ALIGN_SNAP_MODE = "qpoint"
# Used by fine_align.py. Options: "qpoint" or "none".

FINE_ALIGN_SNAP_ON_DEADZONE = True
# Used by fine_align.py. True runs the snap only after the deadzone settle count is reached.

FINE_ALIGN_SNAP_CROP_HALF_PX = None
# Used by fine_align.py. None keeps the existing bbox-scaled snap crop; int forces a half-size.


# =============================================================================
# Strike
# =============================================================================

STRIKE_PATTERN = "pulse"
# Used by control/strike.py. Only "pulse" is currently supported.

LASER_FIRE_POWER = 25
LASER_FIRE_DURATION_SEC = 2.0
LASER_ARM_DELAY_SEC = 0.01
# Used by control/strike.py. Effective only when FIRE = True.


# =============================================================================
# Trial Recording
# =============================================================================

RECORD_RAW_FRAMES_ONLY = True
# Used by hardware/cameras.py. Default live path saves raw left/right images plus JSONL manifest.

RECORD_FRAME_FORMAT = "jpg"
# Used by hardware/cameras.py. Options are "jpg", "jpeg", or "png".

RECORD_JPEG_QUALITY = 90
# Used by hardware/cameras.py when RECORD_FRAME_FORMAT is jpg/jpeg.

RECORD_EVERY_N_FRAMES = 1
# Used by hardware/cameras.py. 1 records every frame pair sent to the recorder.

RECORD_MAX_FPS = None
RECORD_MIN_INTERVAL_SEC = 0.0
# Used by hardware/cameras.py. RECORD_MAX_FPS, when set, takes precedence over min interval.

RECORD_LIVE_VIDEO = False
# Used by hardware/cameras.py. True enables the old live stitched-video recorder path.

RECORD_LIVE_OVERLAYS = False
# Used by hardware/cameras.py. True lets old live video consume detector/tracker overlays.

RECORD_VIDEO_FPS = 15.0
# Used by hardware/cameras.py. Raw-recorder sampling and optional legacy video FPS.

RECORD_VIDEO_SCALE = 0.5
# Used by hardware/cameras.py. 0.5 records half-size; 1.0 records full-size.

RECORD_VIDEO_TIMESTAMP = True
# Used by hardware/cameras.py. Burns elapsed time and wall clock into videos.

RECORD_VIDEO_OVERLAY = RECORD_LIVE_OVERLAYS
# Backward-compatible alias for the optional legacy live video path.

RECORD_VIDEO_DEBUG = False
# Used by hardware/cameras.py. True prints recorder heartbeat/stats.


# =============================================================================
# Training Photo Grid Capture
# =============================================================================

X_SUBSECTIONS = 4
Y_SUBSECTIONS = 3
# Used by data_collection/grid_capture.py. Turn UP for denser training-photo grid.

PHOTO_SETTLE_SEC = 1.5
# Used by data_collection/grid_capture.py. Turn UP if gantry/cameras need more settle time.


# =============================================================================
# Experiment Logging
# =============================================================================

ENABLE_EXPERIMENT_LOGGING = True
# Used by main.py. False disables all metrics logging.

EXPERIMENT_TRIAL_ID = ""
# Human-readable label for this trial, e.g. "N8_grid_01".

EXPERIMENT_TRIAL_TYPE = "timing"
# Category of experiment, e.g. "timing", "accuracy", "weed_plus_kale".

EXPERIMENT_LAYOUT_TYPE = "grid"
# Spatial arrangement of plants, e.g. "grid", "random_uniform", "clustered", "manual".

EXPECTED_WEED_COUNT = 0
# Number of weed targets physically placed in the workspace.

EXPECTED_KALE_COUNT = 0
# Number of kale/non-target plants physically placed in the workspace.

EXPERIMENT_NOTES = ""
# Free-text notes appended to every run row in the CSV.
