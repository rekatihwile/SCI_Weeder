"""
bringup/08_runtime_step_machine.py
------------------------------------
New minimal pipeline script.

Runs: ENV -> CAMERAS -> DETECTOR -> optional GANTRY_HOME -> SURVEY_MOVE -> SURVEY_DETECT -> MATCH -> PLAN

USE_REAL_GANTRY = False  =>  uses MockGantry (no serial, no motion)

Does NOT:
  - fire laser
  - fine-align
  - execute target moves
  - record trial video

Run with:
    ./run_with_eli_venv.sh bringup/08_runtime_step_machine.py | tee bringup/logs/08_runtime_step_machine.log
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Patch broken torchvision C++ ops BEFORE importing ultralytics
import importlib.util as _ilu
_patch_path = Path(__file__).resolve().parent / "_nms_patch.py"
_spec = _ilu.spec_from_file_location("_nms_patch", _patch_path)
_mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Top-level toggle ──────────────────────────────────────────────────────────
USE_REAL_GANTRY = False   # Set True to use real hardware gantry
FLUSH_FRAMES    = 8


def step(label):
    print(f"\n{'=' * 60}")
    print(f"STEP: {label}")
    print(f"{'=' * 60}")


def main():
    from config import (
        GRBL_PORT,
        SURVEY_BURST_COUNT,
        SURVEY_MIN_HITS,
        SURVEY_CLUSTER_RADIUS_PX,
        SURVEY_TARGET_CLASSES,
        SURVEY_POINT_MODE,
        DETECTOR_MODE,
        SURVEY_POS_X,
        SURVEY_POS_Y,
        HOMING,
    )
    from vision.detectors.ai_detector import AIDetector
    from hardware.cameras import StereoCameras
    from control.coarse_move import TriangulationCoarseMover
    from vision.matching import match_points_constellation
    from planning.target_planner import plan_targets

    # ── STEP 1: ENV ────────────────────────────────────────────────────────────
    step("ENV")
    import torch
    import cv2
    import numpy as np
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  cv2.__version__          : {cv2.__version__}")
    print(f"  SURVEY_POS: ({SURVEY_POS_X}, {SURVEY_POS_Y})")
    print(f"  USE_REAL_GANTRY          : {USE_REAL_GANTRY}")

    # ── STEP 2: DETECTOR (warmup before cameras) ───────────────────────────────
    step("DETECTOR")
    detector = AIDetector()
    detector.warmup()
    print("  Detector ready.")

    # ── STEP 3: CAMERAS ────────────────────────────────────────────────────────
    step("CAMERAS")
    cameras = StereoCameras()
    cameras.open(start_recorder=False)
    print("  Cameras ready.")

    # ── STEP 4: GANTRY ────────────────────────────────────────────────────────
    step("GANTRY")
    if USE_REAL_GANTRY:
        from hardware.gantry import Gantry
        gantry = Gantry(GRBL_PORT)
        print("  Real gantry opened.")
        if HOMING:
            print("  Homing...")
            gantry.home()
            print(f"  Homed. Position: {gantry.get_position()}")
    else:
        from hardware.mock_gantry import MockGantry
        gantry = MockGantry(start_x=SURVEY_POS_X, start_y=SURVEY_POS_Y)
        print("  MockGantry in use — no serial, no motion.")

    # ── STEP 5: SURVEY MOVE ───────────────────────────────────────────────────
    step("SURVEY_MOVE")
    gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
    pos = gantry.get_position()
    print(f"  At survey position: {pos}")

    # ── STEP 6: SURVEY_DETECT ─────────────────────────────────────────────────
    step("SURVEY_DETECT")
    coarse_mover = TriangulationCoarseMover()

    print(f"  Flushing {FLUSH_FRAMES} frames...")
    for _ in range(FLUSH_FRAMES):
        cameras.read_pair()

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
    print(f"  Survey detections: left={len(left_dets)}  right={len(right_dets)}")

    cameras.close()
    print("  Cameras closed.")

    # ── STEP 7: MATCH ─────────────────────────────────────────────────────────
    step("MATCH")
    if left_dets and right_dets:
        matched_targets, unmatched_left, unmatched_right = match_points_constellation(left_dets, right_dets)
        print(f"  Matched pairs: {len(matched_targets)}")
        print(f"  Unmatched left: {len(unmatched_left)}  Unmatched right: {len(unmatched_right)}")
    else:
        matched_targets = []
        print("  No detections to match.")

    # ── STEP 8: PLAN ──────────────────────────────────────────────────────────
    step("PLAN")
    if matched_targets:
        solved  = coarse_mover.solve_all_from_pose(matched_targets, SURVEY_POS_X, SURVEY_POS_Y)
        planned = plan_targets(solved, start_xy=(SURVEY_POS_X, SURVEY_POS_Y))
    else:
        solved  = []
        planned = []
    print(f"  Solved: {len(solved)}  Planned: {len(planned)}")
    for i, p in enumerate(planned):
        print(f"    [{i}] target_xy_mm={p.get('target_xy_mm')}")

    # Save plan
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        try:
            import numpy as np
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
        except ImportError:
            pass
        return obj

    plan_path = LOGS_DIR / "08_plan.json"
    plan_path.write_text(json.dumps(_json_safe({
        "survey_pos": {"x": SURVEY_POS_X, "y": SURVEY_POS_Y},
        "detections": {"left": len(left_dets), "right": len(right_dets)},
        "matched_targets": matched_targets,
        "solved_targets":  solved,
        "planned_targets": planned,
    }), indent=2))
    print(f"\n  Plan saved: {plan_path}")

    # ── FINAL RESULT ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULT: PASS  (pipeline reached PLAN step; planned={len(planned)})")
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
