"""
bringup/08_fine_align_debug.py
------------------------------
Interactive Fine Align debug script.

Load a cached survey/match/plan result, select one planned target,
optionally coarse-move the gantry, and/or run one Re-ID attempt.

No PD loop.  No laser.  Does not run main.py.

Run with:
    ./run_with_eli_venv.sh bringup/08_fine_align_debug.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Patch broken torchvision C++ NMS BEFORE importing ultralytics/YOLO
import importlib.util as _ilu
_patch_path = Path(__file__).resolve().parent / "_nms_patch.py"
_spec = _ilu.spec_from_file_location("_nms_patch", _patch_path)
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def main():
    from pipeline.steps.fine_align_debug import (
        load_latest_plan,
        list_cached_targets,
        get_cached_target,
        move_to_cached_target,
        run_reid_once,
    )
    from config import GRBL_PORT

    print("=" * 60)
    print("BRINGUP 08 — Fine Align Debug")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Load cached plan                                                     #
    # ------------------------------------------------------------------ #
    print("\n--- Loading cached plan ---")
    try:
        plan = load_latest_plan()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    targets = list_cached_targets(plan)
    if not targets:
        print("No targets in plan.  Run bringup/07_match_plan_only.py first.")
        sys.exit(1)

    print(
        f"\nPlan:  {len(targets)} target(s)  "
        f"frame_mode={plan.get('frame_mode', '?')}  "
        f"created={plan.get('created_at', '?')}"
    )
    print(
        f"\n{'ID':>3}  {'coarse_x_mm':>12}  {'coarse_y_mm':>12}  "
        f"{'score':>7}  {'cls':>4}  left_px_survey"
    )
    print("-" * 72)
    for t in targets:
        lp = t.get("left_px_survey", [None, None])
        print(
            f"{t['target_id']:>3}  "
            f"{t['coarse_x_mm']:>12.2f}  "
            f"{t['coarse_y_mm']:>12.2f}  "
            f"{t.get('match_score', 0.0):>7.3f}  "
            f"{str(t.get('class_id', '-')):>4}  "
            f"{lp}"
        )

    # ------------------------------------------------------------------ #
    # Select target                                                        #
    # ------------------------------------------------------------------ #
    print()
    try:
        raw = input("Enter target_id: ").strip()
        tid = int(raw)
    except (ValueError, EOFError):
        print("Aborted.")
        sys.exit(0)

    try:
        target = get_cached_target(plan, tid)
    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nSelected: {target}")

    # ------------------------------------------------------------------ #
    # Select action                                                        #
    # ------------------------------------------------------------------ #
    print("\nAction:")
    print("  1 = Move coarse only")
    print("  2 = Run Re-ID once only")
    print("  3 = Move coarse then run Re-ID once")
    print("  q = Quit")
    try:
        action = input("Action [1/2/3/q]: ").strip().lower()
    except EOFError:
        print("Aborted.")
        sys.exit(0)

    if action == "q":
        print("Quit.")
        sys.exit(0)

    do_move = action in ("1", "3")
    do_reid = action in ("2", "3")

    if not do_move and not do_reid:
        print("Invalid action.")
        sys.exit(1)

    # Optional feed for coarse move
    feed = None
    if do_move:
        try:
            feed_str = input("Gantry feed mm/min [enter = default 12000]: ").strip()
            if feed_str:
                feed = float(feed_str)
        except (ValueError, EOFError):
            pass

    # ------------------------------------------------------------------ #
    # Open hardware that will be needed                                    #
    # ------------------------------------------------------------------ #
    gantry  = None
    cameras = None

    try:
        if do_move:
            from hardware.gantry import Gantry
            print("\n--- Opening gantry ---")
            gantry = Gantry(GRBL_PORT)
            print(f"  Gantry open on {GRBL_PORT}")

        if do_reid:
            from vision.detectors.ai_detector import AIDetector
            from hardware.cameras import StereoCameras

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

            print("  Flushing 5 frames ...")
            for _ in range(5):
                cameras.read_pair()

        # -------------------------------------------------------------- #
        # Coarse move                                                      #
        # -------------------------------------------------------------- #
        if do_move:
            print(f"\n--- Moving coarse to target {tid} ---")
            move_result = move_to_cached_target(gantry, target, feed=feed)
            print(f"  Result: {move_result}")

        # -------------------------------------------------------------- #
        # Re-ID once                                                       #
        # -------------------------------------------------------------- #
        if do_reid:
            print(f"\n--- Running Re-ID once for target {tid} ---")
            reid_result = run_reid_once(
                cameras=cameras,
                detector=detector,
                target=target,
                use_rectified=True,
            )

            print("\n--- Re-ID result ---")
            print(f"  ok:         {reid_result['ok']}")
            print(f"  frame_mode: {reid_result.get('frame_mode')}")
            print(f"  left dets:  {len(reid_result.get('left_detections', []))}")
            print(f"  right dets: {len(reid_result.get('right_detections', []))}")
            print(f"  matches:    {len(reid_result.get('matches', []))}")
            print(f"  chosen:     {reid_result.get('chosen')}")
            print(f"  timing:")
            for k, v in reid_result.get("timing", {}).items():
                print(f"    {k}: {v}")
            if reid_result.get("error"):
                print(f"  ERROR:\n{reid_result['error']}")
            print(f"\n  Debug files:")
            for k, v in reid_result.get("debug_files", {}).items():
                print(f"    {k}: {v}")

    finally:
        if cameras is not None:
            try:
                cameras.close()
                print("\n  Cameras closed.")
            except Exception:
                pass
        if gantry is not None:
            try:
                gantry.close()
                print("  Gantry closed.")
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
