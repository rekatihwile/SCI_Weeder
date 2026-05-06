"""
config/experiment.py — experiment logging and trial metadata.

KNOBS:
  ENABLE_EXPERIMENT_LOGGING  — False to skip all metrics CSV logging
  EXPERIMENT_TRIAL_ID        — human-readable label for this trial
  EXPECTED_WEED_COUNT        — number of weed targets physically placed in the workspace
"""

# =============================================================================
# Experiment logging
# =============================================================================

ENABLE_EXPERIMENT_LOGGING = True
# Used by main.py. False disables all metrics logging.

EXPERIMENT_TRIAL_ID   = ""
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
