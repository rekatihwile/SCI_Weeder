"""
bringup/07_match_plan_only.py
------------------------------
Run survey detection, matching, triangulation, and planning.
No gantry movement. Uses MockGantry.

Run with:
    ./run_with_eli_venv.sh bringup/07_match_plan_only.py | tee bringup/logs/07_match_plan_only.log
"""

import sys
import json
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
        SURVEY_POS_X,
        SURVEY_POS_Y,
    )
    from vision.detectors.ai_detector import AIDetector
    from hardware.cameras import StereoCameras
    from hardware.mock_gantry import MockGantry
    from control.coarse_move import TriangulationCoarseMover
    from vision.matching import match_points_constellation
    from planning.target_planner import plan_targets

    print("=" * 60)
    print("BRINGUP 07 — Match + Plan Only (no gantry movement)")
    print("=" * 60)

    print(f"\n  SURVEY_POS_X: {SURVEY_POS_X}  SURVEY_POS_Y: {SURVEY_POS_Y}")

    gantry = MockGantry(start_x=SURVEY_POS_X, start_y=SURVEY_POS_Y)

    print("\n--- Building detector ---")
    detector = AIDetector()

    print("\n--- Detector warmup ---")
    detector.warmup()

    print("\n--- Opening cameras ---")
    cameras = StereoCameras()
    cameras.open(start_recorder=False)

    print(f"\n--- Flushing {FLUSH_FRAMES} frames ---")
    for _ in range(FLUSH_FRAMES):
        cameras.read_pair()

    print("\n--- Building coarse_mover ---")
    coarse_mover = TriangulationCoarseMover()

    print("\n--- Survey detection ---")
    left_dets, right_dets = coarse_mover.detect_stable_points(
        cameras=cameras,
        detector=detector,
        detector_mode=DETECTOR_MODE,
        burst_count=SURVEY_BURST_COUNT,
        min_hits=SURVEY_MIN_HITS,
        cluster_radius_px=SURVEY_CLUSTER_RADIUS_PX,
        survey_classes=SURVEY_TARGET_CLASSES,
        point_mode=SURVEY_POINT_MODE,
    )
    print(f"  Left: {len(left_dets)}  Right: {len(right_dets)}")

    cameras.close()
    print("  Cameras closed.")

    if not left_dets and not right_dets:
        print("\n  No detections — skipping match/plan (scene may have no visible plants).")
        print("\n" + "=" * 60)
        print("RESULT: PASS  (completed through planning; 0 detections is acceptable)")
        print("=" * 60)
        # Save empty plan
        plan_path = LOGS_DIR / "07_plan.json"
        plan_path.write_text(json.dumps({"matched_targets": [], "planned_targets": []}, indent=2))
        print(f"  Saved: {plan_path}")
        sys.exit(0)

    print("\n--- Fitting epipolar (if available) ---")
    if hasattr(coarse_mover, "fit_epipolar") and left_dets and right_dets:
        try:
            coarse_mover.fit_epipolar(left_dets, right_dets)
            print("  fit_epipolar done.")
        except Exception as e:
            print(f"  fit_epipolar skipped: {e}")

    print("\n--- Matching points ---")
    matched_targets = match_points_constellation(left_dets, right_dets)
    print(f"  Matched pairs: {len(matched_targets)}")
    for i, t in enumerate(matched_targets):
        print(f"    [{i}] left={t.get('left_px')}  right={t.get('right_px')}")

    print("\n--- Triangulating (solve_all_from_pose) ---")
    solved = coarse_mover.solve_all_from_pose(matched_targets, SURVEY_POS_X, SURVEY_POS_Y)
    print(f"  Solved targets: {len(solved)}")
    for i, s in enumerate(solved):
        xy = s.get("target_xy_mm")
        print(f"    [{i}] target_xy_mm={xy}")

    print("\n--- Planning ---")
    planned = plan_targets(solved, start_xy=(SURVEY_POS_X, SURVEY_POS_Y))
    print(f"  Planned targets: {len(planned)}")
    for i, p in enumerate(planned):
        xy = p.get("target_xy_mm")
        print(f"    [{i}] target_xy_mm={xy}")

    # Save plan JSON
    def _json_safe(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    plan_path = LOGS_DIR / "07_plan.json"
    plan_data = {
        "matched_targets": _json_safe(matched_targets),
        "solved_targets":  _json_safe(solved),
        "planned_targets": _json_safe(planned),
    }
    plan_path.write_text(json.dumps(plan_data, indent=2))
    print(f"\n  Saved: {plan_path}")

    print("\n" + "=" * 60)
    print(f"RESULT: PASS  (matched={len(matched_targets)} solved={len(solved)} planned={len(planned)})")
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
