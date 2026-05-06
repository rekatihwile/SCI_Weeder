"""
config/hardware.py — hardware JSON config, camera/gantry settings, workspace
                     dimensions, calibration paths, and display detection.

KNOBS most often changed:
  FRAME_WIDTH / FRAME_HEIGHT    — camera resolution throughout the pipeline
  WORKSPACE_X_MAX / Y_MAX       — gantry travel limits in mm
  HAS_DISPLAY                   — override True only when X11 is available
  LASER_OFFSET_X_MM             — physical laser-to-camera centre offset (mm)
"""

import json
import os
import sys

from .paths import BASE_DIR, CV_WEIGHTS_DIR

# =============================================================================
# Hardware JSON loading
# =============================================================================

_HARDWARE_CONFIG_PATH = BASE_DIR / "params/hardware/hardware_config.json"
_CAMERA_CONFIG_PATH   = BASE_DIR / "params/hardware/camera_config.json"

if not _HARDWARE_CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing hardware config: {_HARDWARE_CONFIG_PATH}")

with open(_HARDWARE_CONFIG_PATH, "r") as _f:
    _hardware_cfg = json.load(_f)

GRBL_PORT          = _hardware_cfg["serial"]["grbl_port"]
LEFT_CAMERA_INDEX  = _hardware_cfg["cameras"]["left"]["index"]
RIGHT_CAMERA_INDEX = _hardware_cfg["cameras"]["right"]["index"]

if _CAMERA_CONFIG_PATH.exists():
    with open(_CAMERA_CONFIG_PATH, "r") as _f:
        CAMERA_SETTINGS = json.load(_f)
else:
    CAMERA_SETTINGS = {}

_runtime_cfg = _hardware_cfg.get("runtime", {})


# =============================================================================
# Camera frame size and workspace
# =============================================================================

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
# Used throughout calibration, detection, matching, and recording.

WORKSPACE_X_MIN = 0.0
WORKSPACE_X_MAX = 420.0
WORKSPACE_Y_MIN = 0.0
WORKSPACE_Y_MAX = 420.0
# Used by coarse_move.py, fine_align.py, workspace_plot.py, and grid_capture.py.
# Shrink these if the gantry gets too close to walls.

IS_WINDOWS = sys.platform.startswith("win")
# Used by hardware/cameras.py to choose the OpenCV backend.


# =============================================================================
# Calibration and triangulation
# =============================================================================

CALIB_NPZ_PATH = BASE_DIR / "params/calibration/stereo_checkerboard_fisheye_calib.npz"
RECT_NPZ_PATH  = BASE_DIR / "params/calibration/stereo_checkerboard_fisheye_rectify_maps.npz"
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
# Display detection
# =============================================================================

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
HAS_DISPLAY = bool(_runtime_cfg.get("has_display", _env_has_display)) and _env_has_display
# Used by UI/OpenCV windows. False for headless Jetson/SSH sessions.

if not HAS_DISPLAY and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

SHOW_TRIANGULATION_PLOT  = False
SHOW_MATCH_DEBUG_WINDOW  = False
SAVE_MATCH_DEBUG_IMAGE   = False
# Used by main.py. Set True only when HAS_DISPLAY is also True.
