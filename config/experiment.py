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


# =============================================================================
# Controlled grid-cell trials
# =============================================================================

EXPERIMENT_GRID_ENABLED = True
# True only annotates/logs per-target grid metadata unless TRIAL_FILTER_ENABLED is
# also True. Bounds below must match the robot workspace coordinate frame.

GRID_ROWS = 5
GRID_COLS = 7
GRID_X_MIN_MM = 0.0
GRID_X_MAX_MM = 420.0
GRID_Y_MIN_MM = 0.0
GRID_Y_MAX_MM = 420.0
# Default bounds mirror config/hardware.py workspace limits. If the usable
# experiment workspace differs from gantry travel, update these bounds.

SURVEY_ORIGIN_X_MM = 200.0
SURVEY_ORIGIN_Y_MM = 150.0
# Origin for radius/angle/ring labels. Defaults mirror SURVEY_POS_X/Y.

TRIAL_FILTER_ENABLED = True
TRIAL_FILTER_MODE = "random_cells"
# Modes: "none", "random_cells", "custom_cells", "radius_cells", "ring".
# Filtering happens after survey/stereo/triangulation and before path planning.

REQUESTED_ACTIVE_CELL_COUNT = 5
# For random_cells this is also treated as the minimum selected target goal:
# the picker prioritizes planning this many eligible matched weeds when possible.
ACTIVE_CELL_IDS = []
ACTIVE_RADIUS_CELLS = 0
ACTIVE_RING_INDEX = 0
RANDOM_SEED = None
FILTER_ONLY_WEED_CLASSES = True

DRY_RUN_GRID_FILTER = False
# When True, stop after survey/stereo/triangulation/grid filtering and metrics
# manifest write; no path execution, movement to targets, fine-align, or firing.
