"""
config/survey_params.py — global survey burst, stereo matching, and crop parameters.

This file is the target of "Save Config Settings" in the dashboard — changes made
via the dashboard UI are written here.  You can also edit values manually.

KNOBS most often changed:
  SURVEY_BURST_COUNT       — frames per burst (more = more stable, slower)
  SURVEY_MIN_HITS          — min frames a plant must appear in to be counted
  SURVEY_CONFIDENCE_OVERRIDE — confidence threshold for survey only (None → AI_CONFIDENCE)
  SURVEY_CROP_MODE         — which part of the frame to search ("center_facing" is usually best)
  SURVEY_POS_X/Y           — where the gantry parks for the global survey
"""

# =============================================================================
# CV mode overrides (apply to all states when enabled)
# =============================================================================

OVERRIDE_BURST_NUMBER     = False
OVERRIDE_BURST_COUNT      = 1
# Used by main.py/fine_align.py. True forces every state to use OVERRIDE_BURST_COUNT.

OVERRIDE_POINT_MODE       = False
OVERRIDE_POINT_MODE_VALUE = "box_center"
# Options: "box_center", "qpoint", "heatmap". True forces every CV state to this mode.

SURVEY_POINT_MODE          = 'box_center'
FINE_ALIGN_REID_POINT_MODE = "box_center"
FINAL_SNAP_POINT_MODE      = "qpoint"
# Survey/Re-ID defaults stay fast. Final snap may use qpoint only when snap is enabled.


# =============================================================================
# KNOBS — survey position and burst quality
# =============================================================================

SURVEY_POS_X = 200.0
SURVEY_POS_Y = 150.0
# Used by main.py. Gantry pose where the global survey burst is captured.

SURVEY_FRAME_WIDTH  = None
SURVEY_FRAME_HEIGHT = None
# Used by coarse_move.py. Set both to higher resolution for HD survey.

SURVEY_BURST_COUNT = 20
# Used by main.py/coarse_move.py. Turn UP for more stable survey detections; slower.

SURVEY_MIN_HITS = 1
# Used by main.py/coarse_move.py. Turn UP to require repeat detections; DOWN to catch weak plants.

SURVEY_CLUSTER_RADIUS_PX = 10.0
# Used by coarse_move.py. Turn UP if burst detections jitter; DOWN to split nearby plants.

SURVEY_YOLO_IMGSZ = [608, 704]
# Used by coarse_move.py survey burst. None = use actual frame size.

SURVEY_CROP_HALF_PX = None
# Legacy symmetric square crop. Overridden by SURVEY_CROP_MODE when that is set.

SURVEY_CROP_MODE = 'center_facing'
# Crop mode for the survey burst. Options:
# "center_facing": left cam crops right-of-center, right cam crops left-of-center.
# "center", "full", "left", "right", "top", "bottom"
# None disables mode-based cropping and falls back to SURVEY_CROP_HALF_PX.
# Written by dashboard "Save Config Settings"; can also be set manually.

SURVEY_CROP_W = 704
SURVEY_CROP_H = 608
# Crop dimensions in pixels used when SURVEY_CROP_MODE is set.

SURVEY_LEFT_OFFSET_X  = -328
SURVEY_LEFT_OFFSET_Y  = 44
# Full-frame pixel offset applied to the left-camera crop center.

SURVEY_RIGHT_OFFSET_X = 228
SURVEY_RIGHT_OFFSET_Y = 48
# Same as above for the right camera.

SURVEY_CONFIDENCE_OVERRIDE = 0.7
# When set (float), overrides AI_CONFIDENCE for the survey burst only.
# None = use AI_CONFIDENCE. Written by dashboard "Save Config Settings".

SURVEY_TARGET_CLASSES = [1]
# Used by main.py/coarse_move.py. None uses AI_TARGET_CLASS; list[int] overrides survey only.

SURVEY_CONF_SENSITIVITY_DEBUG = False
# Used by coarse_move.py. True runs extra YOLO passes after survey; very slow/noisy.


# =============================================================================
# Stereo matching
# =============================================================================

STEREO_MATCH_MIN_DISPARITY_PX = 5.0
STEREO_MATCH_MAX_DISPARITY_PX = 250.0
# Used by vision/matching.py. Widen only if true stereo pairs are being rejected.

STEREO_MATCH_MAX_Y_DIFF_PX = 10.0
# Used by vision/matching.py. Turn UP only if valid left/right pairs are rejected.

STEREO_MATCH_RADIUS_PX  = 60.0
STEREO_MATCH_MIN_SCORE  = 0.1
STEREO_MATCH_MIN_BOX_IOU = 0.15
STEREO_MATCH_IOU_WEIGHT  = 0.50
# Used by vision/matching.py. Adjust IoU weight to trust box geometry more/less.
