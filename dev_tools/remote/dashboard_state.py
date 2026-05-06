"""Shared state for the remote dashboard.

This file owns:
- repo path setup
- shared dashboard state object
- camera/gantry locks
- commonly used config constants

Other dashboard modules import shared state from here.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

# Repo paths
REMOTE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REMOTE_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    RECT_NPZ_PATH,
    SURVEY_BURST_COUNT,
    SURVEY_MIN_HITS,
    SURVEY_YOLO_IMGSZ,
    SURVEY_TARGET_CLASSES,
    SURVEY_AVOID_CLASSES,
    SURVEY_POINT_MODE,
    TARGET_CLASSES,
    AVOID_CLASSES,
    GRBL_PORT,
)


@dataclass
class DashboardState:
    cameras: object = None
    detector: object = None
    gantry: object = None
    last_scan: dict = None
    last_triangulation: dict = None
    rectify_cache: dict = None
    workspace_projector: object = None

    camera_lock: Lock = field(default_factory=Lock)
    gantry_lock: Lock = field(default_factory=Lock)


state = DashboardState()

# Backward-compatible aliases for modules created during the split.
lock = state.camera_lock