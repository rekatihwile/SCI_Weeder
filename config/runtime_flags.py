"""
config/runtime_flags.py — top-level operator toggles for the laser weeder.

KNOBS — the switches you flip most often:
  HOMING       — True to home the gantry at startup
  FIRE         — True to actually pulse the laser (keep False for dry runs!)
  MOCK_GANTRY  — True to run the pipeline without real hardware (offline testing)
  FULL_AUTO    — True to skip all operator input prompts (unattended machine run)
  DETECTOR_MODE — "ai" (YOLO) or "manual" (click-to-detect)
"""

# =============================================================================
# KNOBS — flip these first when changing machine behaviour
# =============================================================================

HOMING = False
# Used by main.py. True = home the gantry at startup.

FIRE = False
# Used by control/strike.py. Keep False for dry runs; True actually pulses laser.

MOCK_GANTRY = False
# Used by main.py. True replaces Gantry with MockGantry so the full pipeline runs
# without opening serial, homing, or physically moving hardware.
# Set True for offline testing; always False in production.

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
