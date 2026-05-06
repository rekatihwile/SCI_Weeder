"""
config/__init__.py
------------------
LaserWeeder runtime configuration package.

This file re-exports every constant from the sub-modules below, so all
existing code that does:

    from config import AI_CONFIDENCE
    import config; config.FIRE

continues to work unchanged.

To find and change a specific knob, look in the sub-module listed below.
Each sub-module has a KNOBS section at the top for the most-changed params.

  config/runtime_flags.py   — HOMING, FIRE, MOCK_GANTRY, FULL_AUTO, DETECTOR_MODE
  config/paths.py            — BASE_DIR, CV_WEIGHTS_DIR, TRIAL_RECORDINGS_DIR
  config/hardware.py         — GRBL_PORT, cameras, workspace dims, calibration, HAS_DISPLAY
  config/vision.py           — AI_CONFIDENCE, YOLO_*, DEFAULT_MODEL, AI_TARGET_CLASS
  config/survey_params.py    — SURVEY_*, STEREO_MATCH_* (written by dashboard "Save Config Settings")
  config/alignment_params.py — FINE_ALIGN_*, LASER_FIRE_POWER, LASER_FIRE_DURATION_SEC
  config/recording.py        — RECORD_*, PHOTO_SETTLE_SEC
  config/experiment.py       — ENABLE_EXPERIMENT_LOGGING, EXPERIMENT_*
"""

# Import order matters: hardware depends on paths; alignment depends on survey.
from .paths            import *   # noqa: F401, F403
from .hardware         import *   # noqa: F401, F403
from .runtime_flags    import *   # noqa: F401, F403
from .vision           import *   # noqa: F401, F403
from .survey_params    import *   # noqa: F401, F403
from .alignment_params import *   # noqa: F401, F403
from .recording        import *   # noqa: F401, F403
from .experiment       import *   # noqa: F401, F403
