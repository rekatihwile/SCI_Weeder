"""
bringup/06_survey_detect_only.py
---------------------------------
At current gantry position, run survey burst detection only.
No matching. No planning. No movement.

Run with:
    ./run_with_eli_venv.sh bringup/06_survey_detect_only.py | tee bringup/logs/06_survey_detect_only.log
"""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Patch broken torchvision C++ ops BEFORE importing ultralytics
import importlib.util as _ilu
_patch_path = Path(__file__).resolve().parent / "_nms_patch.py"
_spec = _ilu.spec_from_file_location("_nms_patch", _patch_path)
_mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FLUSH_FRAMES = 8


def main():
    from config import (
        SURVEY_BURST_COUNT,
        SURVEY_MIN_HITS,
        SURVEY_CLUSTER_RADIUS_PX,
        SURVEY_TARGET_CLASSES,
        SURVEY_POINT_MODE,
        DETECTOR_MODE,
    )
    from vision.detectors.ai_detector import AIDetector
    from hardware.cameras import StereoCameras
    from control.coarse_move import TriangulationCoarseMover
    from pipeline.steps.survey import run_survey_detection

    print("=" * 60)
    print("BRINGUP 06 — Survey Detection Only")
    print("=" * 60)

    print(f"\n  SURVEY_BURST_COUNT     : {SURVEY_BURST_COUNT}")
    print(f"  SURVEY_MIN_HITS        : {SURVEY_MIN_HITS}")
    print(f"  SURVEY_CLUSTER_RADIUS  : {SURVEY_CLUSTER_RADIUS_PX}")
    print(f"  SURVEY_TARGET_CLASSES  : {SURVEY_TARGET_CLASSES}")
    print(f"  SURVEY_POINT_MODE      : {SURVEY_POINT_MODE}")
    print(f"  DETECTOR_MODE          : {DETECTOR_MODE}")

    print("\n--- Building detector ---")
    detector = AIDetector()

    print("\n--- Detector warmup ---")
    detector.warmup()

    print("\n--- Opening cameras ---")
    cameras = StereoCameras()
    cameras.open(start_recorder=False)

    print("\n--- Building coarse_mover ---")
    coarse_mover = TriangulationCoarseMover()

    print(f"\n--- Running survey detection (flush={FLUSH_FRAMES}) ---")
    left_dets, right_dets = run_survey_detection(cameras, detector, coarse_mover)

    print(f"\n  Left  detections: {len(left_dets)}")
    for i, d in enumerate(left_dets):
        pt   = d.get("point") if isinstance(d, dict) else d
        cls  = d.get("cls",  "?") if isinstance(d, dict) else "?"
        conf = d.get("conf", "?") if isinstance(d, dict) else "?"
        print(f"    [{i}] class={cls}  conf={conf}  point={pt}")

    print(f"\n  Right detections: {len(right_dets)}")
    for i, d in enumerate(right_dets):
        pt   = d.get("point") if isinstance(d, dict) else d
        cls  = d.get("cls",  "?") if isinstance(d, dict) else "?"
        conf = d.get("conf", "?") if isinstance(d, dict) else "?"
        print(f"    [{i}] class={cls}  conf={conf}  point={pt}")

    # Save representative frames if available
    fL = getattr(coarse_mover, "last_survey_frameL", None)
    fR = getattr(coarse_mover, "last_survey_frameR", None)
    if fL is not None and fR is not None:
        left_out  = fL.copy()
        right_out = fR.copy()
        for d in left_dets:
            pt = d.get("point") if isinstance(d, dict) else d
            if pt:
                cv2.circle(left_out, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)
        for d in right_dets:
            pt = d.get("point") if isinstance(d, dict) else d
            if pt:
                cv2.circle(right_out, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)
        cv2.imwrite(str(LOGS_DIR / "06_left.jpg"),  left_out)
        cv2.imwrite(str(LOGS_DIR / "06_right.jpg"), right_out)
        print(f"\n  Saved: {LOGS_DIR}/06_left.jpg")
        print(f"  Saved: {LOGS_DIR}/06_right.jpg")
    else:
        print("\n  (No representative frames available to save)")

    cameras.close()
    print("  Cameras closed.")

    print("\n" + "=" * 60)
    print(f"RESULT: PASS  (left={len(left_dets)} right={len(right_dets)} detections; 0 is okay if no plants visible)")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
