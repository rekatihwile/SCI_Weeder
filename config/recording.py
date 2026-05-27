"""
config/recording.py — trial recording parameters.

KNOBS most often changed:
  RECORD_TRIAL         — master switch (in runtime_flags.py — see there)
  RECORD_JPEG_QUALITY  — JPEG quality for saved frames (90 = good balance)
  RECORD_EVERY_N_FRAMES — 1 = record every frame, 2 = every other, etc.
  RECORD_MAX_FPS       — cap recording rate (None = uncapped)
"""

# =============================================================================
# Raw frame recorder (default path)
# =============================================================================

RECORD_RAW_FRAMES_ONLY = False
# Used by hardware/cameras.py. Default live path saves raw left/right images
# plus a JSONL manifest.

RECORD_FRAME_FORMAT  = "jpg"
# Used by hardware/cameras.py. Options: "jpg", "jpeg", or "png".

RECORD_JPEG_QUALITY  = 75
# Used by hardware/cameras.py when RECORD_FRAME_FORMAT is jpg/jpeg.

RECORD_EVERY_N_FRAMES   = 5
RECORD_MAX_FPS          = None
RECORD_MIN_INTERVAL_SEC = 0.0
# RECORD_MAX_FPS, when set, takes precedence over RECORD_MIN_INTERVAL_SEC.


# =============================================================================
# Legacy live-video recorder (usually disabled)
# =============================================================================

RECORD_LIVE_VIDEO    = False
RECORD_LIVE_OVERLAYS = False
RECORD_VIDEO_FPS     = 15.0
RECORD_VIDEO_SCALE   = 0.5
RECORD_VIDEO_TIMESTAMP = True
RECORD_VIDEO_OVERLAY = RECORD_LIVE_OVERLAYS   # backward-compat alias
RECORD_VIDEO_DEBUG   = False


# =============================================================================
# Training photo grid capture
# =============================================================================

X_SUBSECTIONS  = 4
Y_SUBSECTIONS  = 3
# Used by data_collection/grid_capture.py. Turn UP for denser training-photo grid.

PHOTO_SETTLE_SEC = 1.5
# Used by data_collection/grid_capture.py. Turn UP if gantry/cameras need more settle time.
