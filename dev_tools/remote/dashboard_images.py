"""Image drawing and encoding helpers for the dashboard.

All drawing logic and colour constants live in vision/visualization.py so that
the dashboard and the live runtime share a single source of truth.  This module
re-exports those symbols for backward compatibility with existing dashboard
imports, and adds a numpy import that dashboard callers depend on.
"""

import numpy as np  # noqa: F401  (imported so callers can rely on this module)
import sys
from pathlib import Path

# Ensure workspace root is in path so the runtime vision package is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision.visualization import (  # noqa: F401  (re-export for dashboard callers)
    jpg_bytes,
    b64_img,
    draw_crop,
    draw_detections,
    draw_stable_detections,
    draw_matches,
)


def draw_horizontal_lines(frame, spacing=40):
    out = frame.copy()
    h, w = out.shape[:2]
    for y in range(0, h, spacing):
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 255), 1)
        cv2.putText(
            out,
            str(y),
            (8, max(16, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
    return out
