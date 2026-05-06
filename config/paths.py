"""
config/paths.py — workspace path constants.

These are imported by virtually every sub-module in the config package.
No dependencies on any other config submodule.
"""

from pathlib import Path

# =============================================================================
# Root path (workspace directory, one level above config/)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# Derived paths used throughout the codebase.
CV_WEIGHTS_DIR      = BASE_DIR / "params/cv_weights"
TRAINING_PHOTOS_DIR = BASE_DIR / "training_photos"
TRIAL_RECORDINGS_DIR = BASE_DIR / "trial_recordings"
