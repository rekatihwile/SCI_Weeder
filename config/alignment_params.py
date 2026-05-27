"""
config/alignment_params.py — fine-align PD control, Re-ID, LK optical flow,
                              final snap, and laser strike parameters.

KNOBS most often changed:
  FINE_ALIGN_KP_X/Y       — proportional gain for gantry PD controller (higher = stronger correction)
  FINE_ALIGN_KD_X/Y       — derivative gain (higher = more damping)
  FINE_ALIGN_DEADZONE_PX  — pixel radius to consider "locked" before firing (lower = tighter)
  FINE_ALIGN_MAX_TIME_SEC — how long the PD loop runs before giving up on a target
  FINE_ALIGN_SETTLE_FRAMES — how many consecutive in-deadzone frames required to declare lock
  LASER_FIRE_POWER        — laser PWM/spindle power S-value (0-1000); only matters when FIRE=True
  LASER_FIRE_DURATION_SEC — pulse duration in seconds; only matters when FIRE=True
"""

from .survey_params import STEREO_MATCH_MAX_Y_DIFF_PX  # shared epipolar tolerance baseline

# =============================================================================
# Re-ID burst settings
# =============================================================================

FINE_ALIGN_CROP_SCALE = 1.0
# Used by fine_align.py. Center crop used for Re-ID acceptance and LK tracking.

FINE_ALIGN_BURST_COUNT      = 8
FINE_ALIGN_REID_BURST_COUNT = FINE_ALIGN_BURST_COUNT   # preferred alias
FINAL_SNAP_BURST_COUNT      = 1
# Turn UP for more stable Re-ID; slower.

FINE_ALIGN_MIN_HITS           = 1
FINE_ALIGN_CLUSTER_RADIUS_PX  = 35.0
FINE_ALIGN_REID_CROP_HALF_PX  = 186
FINE_ALIGN_REID_STEREO_DISP_PX = 0          # historical alias; keep 0
FINE_ALIGN_REID_YOLO_IMGSZ    = None

FINE_ALIGN_REID_MAX_Y_DIFF_PX     = 10
FINE_ALIGN_REID_MIN_DISPARITY_PX  = 10
FINE_ALIGN_REID_MAX_DISPARITY_PX  = 500
FINE_ALIGN_REID_MAX_PD_ERROR_PX   = 220
FINE_ALIGN_REID_EPIPOLAR_TOL_MULT = 6.0
FINE_ALIGN_REID_MAX_TRI_DIST_MM   = 35
FINE_ALIGN_REID_SETTLE_SEC        = 0.5
# Pause after the coarse gantry move and immediately before Re-ID camera reads.


# =============================================================================
# LK optical flow
# =============================================================================

FINE_ALIGN_LK_WIN_SIZE  = 31
FINE_ALIGN_LK_MAX_LEVEL = 3
# Larger window tracks bigger motion but can drift more.


# =============================================================================
# KNOBS — PD controller
# =============================================================================

FINE_ALIGN_KP_X = 15.0
FINE_ALIGN_KD_X = 2.5
FINE_ALIGN_KP_Y = 20.0
FINE_ALIGN_KD_Y = 2.5
# Turn UP Kp for stronger correction; KD damps oscillation.

FINE_ALIGN_STEP_MM    = 0.001
# Pixel-error to mm jog scale. Turn UP for larger jogs per pixel.

FINE_ALIGN_DEADZONE_PX = 2.5
# Turn UP to accept looser locks; DOWN for tighter locks (longer settle time).

FINE_ALIGN_MAX_JOG_MM = 10.0
# Safety cap on one fine-align jog.

FINE_ALIGN_FEED = 5000
# Gantry feed rate for fine-align jogs.

FINE_ALIGN_MAX_TIME_SEC = 15.0
# Turn UP to give difficult targets more time before timeout.

FINE_ALIGN_SETTLE_FRAMES      = 10
# Turn UP to require a longer stable lock before firing.

FINE_ALIGN_SNAP_SETTLE_FRAMES = 4
# Settle frames for the final snap re-lock pass.

FINE_ALIGN_ENABLE_SNAP    = False
# False skips the final qpoint heatmap snap and fires from LK/PD lock.

FINE_ALIGN_SNAP_MODE      = "qpoint"
FINE_ALIGN_SNAP_ON_DEADZONE = True
FINE_ALIGN_SNAP_CROP_HALF_PX = None


# =============================================================================
# Laser strike
# =============================================================================

STRIKE_PATTERN        = "pulse"
# Used by control/strike.py. Only "pulse" is currently supported.

LASER_FIRE_POWER        = 1000
LASER_FIRE_DURATION_SEC = 2.0
LASER_ARM_DELAY_SEC     = 0.01
# Used by control/strike.py. Effective only when FIRE = True (in runtime_flags.py).
