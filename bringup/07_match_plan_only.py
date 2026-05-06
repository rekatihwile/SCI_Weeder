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
    from pipeline.steps.match_plan import run_match_and_plan

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
    try:
        cameras.open(start_recorder=False)
    except RuntimeError:
        cameras.recover()

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

    plan_path = LOGS_DIR / "07_plan.json"
    print("\n--- Matching + triangulating + planning ---")
    matched_targets, solved, planned = run_match_and_plan(
        left_dets,
        right_dets,
        coarse_mover,
        start_xy=(SURVEY_POS_X, SURVEY_POS_Y),
        output_path=plan_path,
    )

    # Save normalised plan for fine align debug pipeline
    from pipeline.steps.fine_align_debug import (
        normalize_planned_targets_to_plan,
        save_latest_plan,
    )
    fine_align_plan = normalize_planned_targets_to_plan(
        planned,
        survey_ref_xy=(SURVEY_POS_X, SURVEY_POS_Y),
        frame_mode="raw",
    )
    save_latest_plan(fine_align_plan)

    print(f"  Matched pairs: {len(matched_targets)}")
    print(f"  Solved targets: {len(solved)}")
    for i, s in enumerate(solved):
        xy = s.get("target_xy_mm")
        print(f"    [{i}] target_xy_mm={xy}")

    print(f"  Planned targets: {len(planned)}")
    for i, p in enumerate(planned):
        xy = p.get("target_xy_mm")
        print(f"    [{i}] target_xy_mm={xy}")
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
