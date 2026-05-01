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
# Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

_HARDWARE_CONFIG_PATH = BASE_DIR / "params/hardware/hardware_config.json"
_CAMERA_CONFIG_PATH = BASE_DIR / "params/hardware/camera_config.json"

CV_WEIGHTS_DIR = BASE_DIR / "params/cv_weights"
TRAINING_PHOTOS_DIR = BASE_DIR / "training_photos"
TRIAL_RECORDINGS_DIR = BASE_DIR / "trial_recordings"


# =============================================================================
# Hardware Config (loaded from JSON)
# =============================================================================

if not _HARDWARE_CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing hardware config: {_HARDWARE_CONFIG_PATH}")

with open(_HARDWARE_CONFIG_PATH, "r") as f:
    _hardware_cfg = json.load(f)

if not isinstance(_hardware_cfg, dict):
    _hardware_cfg = {}

_serial_cfg = _hardware_cfg.get("serial") or {}
_cameras_cfg = _hardware_cfg.get("cameras") or {}
_left_camera_cfg = _cameras_cfg.get("left") or {}
_right_camera_cfg = _cameras_cfg.get("right") or {}

GRBL_PORT = _serial_cfg.get("grbl_port", "/dev/ttyUSB0")  # Serial port for the GRBL motion controller.
LEFT_CAMERA_INDEX = _left_camera_cfg.get("index", 0)       # V4L2/MSMF device index for the left camera.
RIGHT_CAMERA_INDEX = _right_camera_cfg.get("index", 1)     # V4L2/MSMF device index for the right camera.

if _CAMERA_CONFIG_PATH.exists():
    with open(_CAMERA_CONFIG_PATH, "r") as f:
        CAMERA_SETTINGS = json.load(f)  # Per-camera exposure, gain, and white-balance startup values.
else:
    CAMERA_SETTINGS = {}


# =============================================================================
# Operator Toggles
# =============================================================================

HOMING = False
# Home the gantry at startup.

FIRE = False
# Enable laser pulses. False = dry run (all motion, no laser); True = live fire.

RECORD_TRIAL = True
# Record raw stereo frame pairs and a JSONL manifest each run.

AUTO_RENDER_TRIAL = True
# Post-render an annotated video after the run completes. Requires RECORD_TRIAL=True.

AUTO_RENDER_DELETE_RAW = False
# Delete raw left/right frame folders after render. Requires AUTO_RENDER_TRIAL=True.

TRIANGULATION_ONLY_MODE = False
# Skip fine-align and strike; log coarse triangulation positions only.

FULL_AUTO = True
# Skip all operator prompts and run unattended.

RUN_PIPELINE_CYCLES = 2
# Number of full HOME->SURVEY->PLAN->EXECUTE cycles per main.py run.
# 2 means the full targeting pipeline is repeated once after completion.

AUTO_MODE = False
# Let cameras self-adjust exposure and white-balance after startup settings are applied.

DETECTOR_MODE = "ai"
# Detection backend. Options: "ai" (YOLO) or "manual" (click-to-select).


# =============================================================================
# Display And Debug
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

    # TCP display -- covers X11 forwarding ("localhost:10.0") and remote X
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

# JSON config can force True, but the display probe is the final authority.
# Prevents Qt from crashing when has_display=true is set for local use but
# the process launches over SSH without X11 forwarding.
HAS_DISPLAY = bool(_runtime_cfg.get("has_display", _env_has_display)) and _env_has_display

# Silence Qt warnings on headless runs. The real guard is HAS_DISPLAY=False in
# calling code -- this cv2 build has no offscreen plugin, so the env var alone won't save us.
if not HAS_DISPLAY and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

SHOW_TRIANGULATION_PLOT = False
# Show the workspace triangulation map after survey.

SHOW_MATCH_DEBUG_WINDOW = False
# Show the stereo-match debug image on screen after survey.

SAVE_MATCH_DEBUG_IMAGE = False
# Write triangulation_debug_view.png to the planning directory after survey.


# =============================================================================
# Camera And Workspace
# =============================================================================

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# Capture resolution. Must match hardware camera settings.

WORKSPACE_X_MIN = 0.0
WORKSPACE_X_MAX = 450.0
WORKSPACE_Y_MIN = 0.0
WORKSPACE_Y_MAX = 440.0
# Gantry workspace bounds in mm. Shrink if the gantry is getting too close to the frame walls.

IS_WINDOWS = sys.platform.startswith("win")
# Selects the OpenCV capture backend: MSMF on Windows, V4L2 on Linux/Jetson.


# =============================================================================
# Calibration And Triangulation
# =============================================================================

CALIB_NPZ_PATH = BASE_DIR / "params/calibration/stereo_checkerboard_fisheye_calib.npz"
RECT_NPZ_PATH = BASE_DIR / "params/calibration/stereo_checkerboard_fisheye_rectify_maps.npz"
# Stereo calibration (.npz) and pre-computed rectification maps (.npz).

CALIBRATION_EXPECTS_UNFLIPPED = False
# Set True if calibration images were captured without the 180 degree camera flip.

TRI_X_SIGN = 1.0
TRI_Y_SIGN = -1.0
# Sign correction for triangulated coordinates. Flip if gantry moves in the wrong direction.

TRI_X_GAIN = 1.0
TRI_Y_GAIN = 1.0
# Scale factor for triangulated distance. Turn UP if moves undershoot; DOWN if they overshoot.

LASER_OFFSET_X_MM = 33.0
LASER_OFFSET_Y_MM = 0.0
# Physical offset from the stereo camera midpoint to the laser beam, in mm.

USE_PIXEL_ERROR_CORRECTION = True
PIXEL_ERROR_MODEL_PATH = BASE_DIR / "params/calibration/stereo_pixel_error_model.json"
# Apply the learned stereo pixel error correction model before triangulation.


# =============================================================================
# AI Detector
# =============================================================================

MODEL_MAP = {
    "plastic_nano": "26_plastic_nano.pt",
    "targeting_tall_plastic": "new_best_targeting_tall_plastic.pth",
}
# Logical name -> filename lookup for weight files in CV_WEIGHTS_DIR.
# DEFAULT_MODEL and DEFAULT_QPOINT_MODEL can also be set directly to a filename.

DEFAULT_MODEL = "plastic_nano"
# YOLO segmentation model used for all detection phases.

DEFAULT_MODEL_ENGINE = None
# TensorRT .engine path/name. None = use .pt only.

DEFAULT_QPOINT_MODEL = "targeting_tall_plastic"
# Heatmap model that refines YOLO box centers to meristem keypoints.

YOLO_BACKEND = "auto"
# YOLO runtime. "pt" = always PyTorch; "engine" = always TensorRT; "auto" = prefer engine if available.

USE_TENSORRT_ENGINE = False
# Prefer TensorRT engine over .pt when YOLO_BACKEND="auto".

YOLO_DEVICE = "cuda:0"
# CUDA device for YOLO inference. Options: "cuda:0", "cpu", or 0.

YOLO_HALF = True
# FP16 inference on CUDA. Faster; disable if NaN results appear.

YOLO_WARMUP = True
YOLO_WARMUP_IMGSZ = 640
YOLO_WARMUP_ITERS = 3
# Run dummy inference passes at startup before trial timing begins.

AI_CONFIDENCE = 0.001
# Global YOLO confidence floor. Per-class overrides in AI_CLASS_CONFIDENCE take priority.

AI_CLASS_CONFIDENCE = {0: 0.10, 1: 0.001, 2: 0.02}
# Per-class confidence thresholds. Turn a class UP to be stricter for that class only.

AI_IOM_THRESHOLD = 0.80
# Intersection-over-minimum to merge overlapping masks. Turn DOWN to merge more aggressively.

AI_TARGET_CLASS = None
# Default class filter: None = all classes, int = single class, list[int] = multiple classes.

AI_DISPLAY_SCALE = 1.0
MANUAL_DISPLAY_SCALE = 0.75
# Debug window scale for AI and manual detector modes respectively.

QPOINT_DEBUG = True
# Print per-detection qpoint pixel coordinates and heatmap peak confidence.


# =============================================================================
# CV Overrides And Per-State Point Modes
# =============================================================================

OVERRIDE_BURST_ENABLED = False
OVERRIDE_BURST_COUNT = 25
# Force all CV states to use OVERRIDE_BURST_COUNT instead of their per-state burst counts.

OVERRIDE_POINT_MODE = False
OVERRIDE_POINT_MODE_VALUE = "box_center"
# Force all CV states to this point mode. Options: "box_center", "qpoint". Overrides per-state values.

SURVEY_POINT_MODE = "box_center"
# Point mode for survey detection. "box_center" is fast; "qpoint" refines to meristem.

FINE_ALIGN_REID_POINT_MODE = "box_center"
# Point mode for Re-ID candidate detection.

FINAL_SNAP_POINT_MODE = "qpoint"
# Point mode for the final snap pass. Only active when FINE_ALIGN_ENABLE_SNAP=True.


# =============================================================================
# Survey
# =============================================================================

SURVEY_POS_X = 200.0
SURVEY_POS_Y = 200.0
# Gantry position (mm) for the global detection burst.

SURVEY_FRAME_WIDTH = None
SURVEY_FRAME_HEIGHT = None
# Camera resolution during survey. None = use FRAME_WIDTH/HEIGHT.
# Set higher (e.g. 1920x1080) to improve detection range at the cost of settle time.

SURVEY_BURST_COUNT = 30
# Frames captured per survey. Turn UP for more stable detections; slower.
# 30: tuned for good lighting to reduce cycle time while keeping stable detections.
# Repair mechanism can synthesize ghost targets from marginal edge detections — avoid relying on it.

SURVEY_MIN_HITS = 1
# Minimum burst-frame hits for a detection to count as stable. Turn UP to filter weak detections.

SURVEY_CLUSTER_RADIUS_PX = 10.0
# Pixel radius to cluster repeated detections across burst frames. Turn UP if detections jitter.

SURVEY_YOLO_IMGSZ = None
# YOLO imgsz override for survey. None = native crop size; set int only to force resizing.

SURVEY_CROP_MODE = "centered"
# Crop applied before survey YOLO. "centered" = fixed pixel rect; "projected" = workspace-projected rect.

SURVEY_CROP_HALF_X_PX = 250
SURVEY_CROP_HALF_Y_PX = 250
# Half-width and half-height of the centered survey crop in pixels.
# Crop spans [cx +/- HALF_X, cy +/- HALF_Y] around the frame center.

SURVEY_PROJECT_CROP_MARGIN_PX = 40
# Pixel padding around the projected workspace boundary when SURVEY_CROP_MODE="projected".

SURVEY_PROJECT_Z_MIN_MM = 250.0
SURVEY_PROJECT_Z_MAX_MM = 650.0
SURVEY_PROJECT_Z_SAMPLES = 7
# Plant height sweep range (mm) and sample count for workspace-to-image projection.
# Covers the depth range where plants could appear in the survey frame.

SURVEY_TARGET_CLASSES = None
# Legacy single-pass class filter passed directly to YOLO. None defers to AI_TARGET_CLASS.
# Prefer SURVEY_CAN_TARGET_CLASSES / SURVEY_CANT_TARGET_CLASSES for finer control.

SURVEY_DETECT_ALL_CLASSES = True
SURVEY_DETECT_CLASS_IDS = [0, 1, 2]
# When True, YOLO runs on all classes in SURVEY_DETECT_CLASS_IDS, then the allow/deny
# filter below is applied before stereo matching.

SURVEY_CAN_TARGET_CLASSES = [1, 2]
SURVEY_CANT_TARGET_CLASSES = None
# Allow/deny class filter applied after detection. CANT always wins over CAN.
# None/int/list[int]. Any class appearing in both lists is blocked.

SURVEY_CONF_SENSITIVITY_DEBUG = False
# Run extra YOLO passes at multiple confidence levels after survey. Very slow; diagnostic only.


# =============================================================================
# Stereo Matching
# =============================================================================

STEREO_MATCH_MIN_DISPARITY_PX = 5.0
STEREO_MATCH_MAX_DISPARITY_PX = 250.0
# Horizontal pixel disparity range for valid stereo pairs. Widen only if true pairs are being rejected.

STEREO_MATCH_MAX_Y_DIFF_PX = 30.0
# Max vertical pixel difference between a matched left/right pair.
# Keep LOW -- large Y differences indicate epipolar error or a wrong plant match.

STEREO_MATCH_RADIUS_PX = 60.0
# Point proximity threshold for point-only (no-box) constellation matching.

STEREO_MATCH_MIN_SCORE = 0.1
# Minimum constellation match score to accept a stereo pair. Turn UP to reject weak matches.

STEREO_MATCH_MIN_BOX_IOU = 0.15
STEREO_MATCH_IOU_WEIGHT = 0.50
# Box IoU floor and score weight for box-IoU constellation matching.
# Turn IOU_WEIGHT UP to trust box geometry more heavily over point proximity.


# =============================================================================
# Fine Align -- Re-ID
# =============================================================================

FINE_ALIGN_CROP_SCALE = 1
# LK/PD tracking window scale relative to full frame. 1.0 = full frame; lower = centered sub-region.
# Re-ID candidates are rejected if their pixel position falls outside this window.

FINE_ALIGN_REID_BURST_COUNT = 2
# Frames captured per Re-ID burst. Turn UP for more stable re-detection; slower.
# 20: min_hits=1 + epipolar/tri_dist/geo_score ranking guards against mismatches.

FINE_ALIGN_MIN_HITS = 1
# Minimum burst frames a Re-ID detection must appear in to be accepted.

FINE_ALIGN_CLUSTER_RADIUS_PX = 35.0
# Cluster radius for Re-ID burst detections. Turn UP if detections jitter between frames.

FINE_ALIGN_REID_CROP_HALF_PX = 75
# Pixel radius of the YOLO detection window during Re-ID burst.
# YOLO scans a [cx +/- half, cy +/- half] square. Independent of FINE_ALIGN_CROP_SCALE.

FINE_ALIGN_REID_YOLO_IMGSZ = None
# YOLO imgsz for Re-ID burst. None = native crop size; set int only to force resizing.

FINE_ALIGN_REID_MAX_Y_DIFF_PX = 80.0
# Max vertical pixel error for Re-ID stereo pairs. Re-ID runs close to the crop,
# where valid local mates can have much larger y separation than survey pairs.

FINE_ALIGN_REID_MIN_DISPARITY_PX = 10.0
FINE_ALIGN_REID_MAX_DISPARITY_PX = 500.0
# Disparity range for Re-ID stereo pairs. Wider than survey to handle close-range targets.

FINE_ALIGN_REID_MAX_PD_ERROR_PX = 150.0
# Max pixel distance (left+right average) from Re-ID triangulated position to frame center.
# Turn DOWN to require less initial travel before fine-align starts.

FINE_ALIGN_REID_EPIPOLAR_TOL_MULT = 3.0
# Multiplier on the survey-fitted epipolar std dev for Re-ID pair acceptance (+/- N x sigma).
# A minimum floor of 0.15 radians is enforced in code.

FINE_ALIGN_REID_MAX_TRI_DIST_MM = 20.0
# Hard cutoff on how far a Re-ID triangulated position can be from the coarse planned position.
# Rejects plants >20mm away. Good re-IDs are typically <10mm.

FINE_ALIGN_REID_SYNTHETIC_MAX_TRI_DIST_MM = 35.0
# Wider Re-ID cutoff for survey targets whose right-side stereo mate was synthesized
# from the median survey offset. Their coarse XY is intentionally less certain.


# =============================================================================
# Fine Align -- LK / PD Tracking
# =============================================================================

FINE_ALIGN_LK_WIN_SIZE = 31
FINE_ALIGN_LK_MAX_LEVEL = 3
# Lucas-Kanade optical flow window size and pyramid levels. Larger window tracks bigger motion.

FINE_ALIGN_KP_X = 15.0
FINE_ALIGN_KD_X = 2.5
FINE_ALIGN_KP_Y = 20.0
FINE_ALIGN_KD_Y = 2.5
# PD controller gains for X and Y axes. Turn UP Kp for stronger correction; KD damps oscillation.

FINE_ALIGN_MM_PER_PX = 0.001
# Converts PD output (pixels) to gantry jog distance (mm). Turn UP for larger jogs per pixel.

FINE_ALIGN_DEADZONE_PX = 3.0
# Pixel error below which the gantry holds position (lock accepted). Turn UP for a looser lock.
# Increased from 2.5: still sub-mm accuracy; duplicate guard unchanged at 8mm tol.

FINE_ALIGN_MAX_JOG_MM = 10.0
# Hard cap on a single fine-align jog step in mm.

FINE_ALIGN_FEED = 5000
# Gantry feed rate (mm/min) for fine-align jogs.

FINE_ALIGN_MAX_TIME_SEC = 15.0
# Timeout before abandoning fine-align on a difficult target.

FINE_ALIGN_SETTLE_FRAMES = 7
# Consecutive frames within the deadzone required before firing.
# Reduced from 10: 7 consecutive in-deadzone frames is still a robust lock signal.


# =============================================================================
# Fine Align -- Final Snap
# =============================================================================

FINE_ALIGN_ENABLE_SNAP = False
# Run a final qpoint heatmap snap after PD lock before firing.
# Improves meristem precision at the cost of extra settling time.

FINE_ALIGN_SNAP_MODE = "qpoint"
# Snap refinement method when FINE_ALIGN_ENABLE_SNAP=True. Options: "qpoint" or "none".

FINE_ALIGN_SNAP_ON_DEADZONE = True
# Only trigger snap after the main deadzone settle count is reached.

FINE_ALIGN_SNAP_SETTLE_FRAMES = 4
# Settle frames for the re-lock pass after qpoint snap. Shorter than FINE_ALIGN_SETTLE_FRAMES
# since snap corrects coarse drift and LK just needs to confirm the new position.

FINAL_SNAP_BURST_COUNT = 1
# Frames averaged for the snap crop. 1 = single frame; higher values reduce noise.

FINE_ALIGN_SNAP_CROP_HALF_PX = None
# Snap crop half-size in pixels. None = auto-scale from the detection bounding box.


# =============================================================================
# Strike
# =============================================================================

STRIKE_PATTERN = "pulse"
# Laser firing pattern. Only "pulse" is currently implemented.

LASER_FIRE_POWER = 25
# Laser power level (0-100). Effective only when FIRE=True.

LASER_FIRE_DURATION_SEC = 2.0
# Pulse duration in seconds.

LASER_ARM_DELAY_SEC = 0.01
# Delay between the arm command and the fire trigger.


# =============================================================================
# Trial Recording
# =============================================================================

RECORD_RAW_FRAMES_ONLY = True
# Save raw left/right image pairs and a JSONL manifest (default).
# Set False to use the legacy live stitched-video path instead.

RECORD_FRAME_FORMAT = "jpg"
# Image format for raw frame saves. Options: "jpg" or "png".

RECORD_JPEG_QUALITY = 90
# JPEG quality (0-100). Only applies when RECORD_FRAME_FORMAT="jpg".

RECORD_EVERY_N_FRAMES = 1
# Save every Nth frame pair. 1 = save all frames.

RECORD_MAX_FPS = None
RECORD_MIN_INTERVAL_SEC = 0.0
# Frame-rate cap for the recorder. RECORD_MAX_FPS takes precedence when set.

RECORD_LIVE_VIDEO = False
# Enable the legacy live stitched-video recorder. Requires RECORD_RAW_FRAMES_ONLY=False.

RECORD_LIVE_OVERLAYS = False
# Burn CV detection overlays into the legacy live video.

RECORD_VIDEO_FPS = 15.0
# Background grab rate and FPS for the legacy live video.

RECORD_VIDEO_SCALE = 0.5
# Downsample scale for the legacy live video. 0.5 = half-res; 1.0 = full.

RECORD_VIDEO_TIMESTAMP = True
# Burn elapsed time and wall clock into legacy live video frames.

RECORD_VIDEO_DEBUG = False
# Print recorder heartbeat and stats to terminal.


# =============================================================================
# Training Photo Grid Capture
# =============================================================================

GRID_X_SUBSECTIONS = 4
GRID_Y_SUBSECTIONS = 3
# Grid density for training photo capture. Turn UP for denser spatial coverage.

GRID_PHOTO_SETTLE_SEC = 1.5
# Wait after each gantry move before capturing. Turn UP if motion blur appears in photos.


# =============================================================================
# Experiment Logging
# =============================================================================

ENABLE_EXPERIMENT_LOGGING = True
# Write per-target and per-run metrics to CSV and JSON.

EXPERIMENT_TRIAL_ID = ""
# Human-readable label for this run, e.g. "N8_grid_01".

EXPERIMENT_TRIAL_TYPE = "timing"
# Category tag, e.g. "timing", "accuracy", "weed_plus_kale".

EXPERIMENT_LAYOUT_TYPE = "grid"
# Plant arrangement tag, e.g. "grid", "random_uniform", "clustered", "manual".

EXPECTED_WEED_COUNT = 16
# Ground-truth weed count physically placed in the workspace.

EXPECTED_KALE_COUNT = 0
# Ground-truth kale/non-target count physically placed in the workspace.

EXPERIMENT_NOTES = "Ground truth target plants: 10 Green Spiked (class 1), 6 Red Plant (class 2)."
# Free-text note appended to every row in the output CSV.
