"""
Reproduce the exact main.py camera open → warmup → flush → burst sequence,
first with a plain sleep (no YOLO), then with the real YOLO GPU warmup.

Goal:
  - If no-YOLO sequence passes but YOLO sequence fails → GPU/model warmup
    is interacting with camera capture (USB DMA or memory pressure on Jetson).
  - If both pass → bug is inside detect_stable_points() or its immediate callers.
  - If both fail → the background grab-loop + timing alone is enough to trigger it.

Run from repo root:
    python diagnostics/main_camera_sequence_probe.py
    python diagnostics/main_camera_sequence_probe.py 2>&1 | tee diagnostics/main_seq_probe_log.txt
"""

import sys
import time
import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hardware.cameras import StereoCameras


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_pair(fl, fr, label=""):
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}fl: type={type(fl).__name__} None={fl is None} "
          f"shape={getattr(fl, 'shape', 'N/A')}")
    print(f"{prefix}fr: type={type(fr).__name__} None={fr is None} "
          f"shape={getattr(fr, 'shape', 'N/A')}")


def _flush(cams, n=8):
    """Mirror main.py _flush_camera_buffer(cameras, n=8)."""
    ok = 0
    for i in range(n):
        try:
            fl, fr = cams.read_pair()
            if fl is not None and fr is not None:
                ok += 1
            else:
                print(f"  [flush {i+1}] fl_None={fl is None} fr_None={fr is None}")
        except Exception as e:
            print(f"  [flush {i+1}] exception: {e}")
    print(f"  flush result: {ok}/{n} OK")
    return ok


def _burst(cams, n=30, label="burst"):
    """Mirror detect_stable_points survey burst (no YOLO, just the read loop)."""
    ok = 0
    fail = 0
    first_fl = None
    first_fr = None

    for i in range(n):
        try:
            fl, fr = cams.read_pair()
        except Exception as e:
            fail += 1
            print(f"  [{label} {i+1}] exception: {e}")
            time.sleep(0.05)
            continue

        if fl is not None and fr is not None:
            ok += 1
            if first_fl is None:
                first_fl = fl.copy()
                first_fr = fr.copy()
        else:
            fail += 1
            if fail <= 3:
                _describe_pair(fl, fr, f"{label} FAIL #{fail}")
            time.sleep(0.05)

    print(f"  {label} result: {ok}/{n} OK, {fail}/{n} failed")
    return ok, first_fl, first_fr


def _save_frames(fl, fr, stem, diag_dir):
    if fl is not None and fr is not None:
        lp = diag_dir / f"{stem}_left.jpg"
        rp = diag_dir / f"{stem}_right.jpg"
        cv2.imwrite(str(lp), fl)
        cv2.imwrite(str(rp), fr)
        print(f"  saved {lp.name} / {rp.name}")
    else:
        print(f"  no valid frames for {stem}")


# ---------------------------------------------------------------------------
# Part 3 — Sleep-based warmup (no YOLO/GPU)
# ---------------------------------------------------------------------------

def part3_no_yolo(diag_dir):
    print("\n" + "=" * 60)
    print("Part 3 — No YOLO: open → 4.5s sleep → flush → burst")
    print("=" * 60)

    cams = StereoCameras()
    cams.open(start_recorder=True)

    print("  Sleeping 4.5s to mimic YOLO warmup duration (no GPU)...")
    time.sleep(4.5)
    print("  Done sleeping.")

    flush_ok = _flush(cams, n=8)
    burst_ok, fl, fr = _burst(cams, n=30, label="no-yolo burst")
    _save_frames(fl, fr, "seq_no_yolo", diag_dir)

    cams.close()
    return flush_ok, burst_ok


# ---------------------------------------------------------------------------
# Part 4 — Real YOLO warmup (same as main.py)
# ---------------------------------------------------------------------------

def part4_with_yolo(diag_dir):
    print("\n" + "=" * 60)
    print("Part 4 — With YOLO: open → detector.warmup() → flush → burst")
    print("=" * 60)

    # Import the same build_detector / AIDetector path main.py uses.
    try:
        from config import DETECTOR_MODE, AI_CONFIDENCE, AI_DISPLAY_SCALE
        from vision.detectors.ai_detector import AIDetector
    except ImportError as e:
        print(f"  Cannot import AIDetector: {e}")
        print("  Skipping Part 4.")
        return None, None

    print("  Building AIDetector (loads YOLO model to GPU)...")
    t0 = time.perf_counter()
    try:
        detector = AIDetector(
            display_scale=AI_DISPLAY_SCALE,
            conf=AI_CONFIDENCE,
        )
    except Exception as e:
        print(f"  AIDetector build failed: {e}")
        return None, None
    print(f"  build done in {time.perf_counter()-t0:.2f}s")

    cams = StereoCameras()
    cams.open(start_recorder=True)

    print("  Running detector.warmup() (GPU forward passes)...")
    t1 = time.perf_counter()
    try:
        warmup_info = detector.warmup()
        print(f"  warmup done in {time.perf_counter()-t1:.2f}s  info={warmup_info}")
    except Exception as e:
        print(f"  warmup exception: {e}")

    flush_ok = _flush(cams, n=8)
    burst_ok, fl, fr = _burst(cams, n=30, label="yolo burst")
    _save_frames(fl, fr, "seq_yolo", diag_dir)

    cams.close()
    return flush_ok, burst_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    diag_dir = REPO_ROOT / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    print("=== main_camera_sequence_probe ===")
    print(f"repo_root={REPO_ROOT}")

    p3_flush, p3_burst = part3_no_yolo(diag_dir)

    time.sleep(2.0)

    p4_flush, p4_burst = part4_with_yolo(diag_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Part 3  no-YOLO   flush={p3_flush}/8   burst={p3_burst}/30")
    if p4_flush is not None:
        print(f"Part 4  with-YOLO flush={p4_flush}/8   burst={p4_burst}/30")
    else:
        print("Part 4  with-YOLO SKIPPED (AIDetector import failed)")

    print()
    if p4_flush is not None:
        if p3_burst == 30 and p4_burst == 0:
            print("RESULT: GPU/YOLO warmup is what causes the failure.")
            print("        Fix candidate: open cameras AFTER warmup completes.")
        elif p3_burst == 0 and p4_burst == 0:
            print("RESULT: Failure occurs without YOLO too. "
                  "Timing/bg-loop/USB is the cause, not GPU-specific.")
        elif p3_burst > 0 and p4_burst > 0:
            print("RESULT: Both pass. Bug is inside detect_stable_points() "
                  "or its callers, not the camera read path itself.")
        else:
            print("RESULT: Mixed. Inspect per-part logs above.")
    else:
        if p3_burst == 30:
            print("RESULT: No-YOLO sequence passes. GPU load is likely involved.")
        else:
            print("RESULT: No-YOLO sequence also fails.")

    print("\n=== probe complete ===")


if __name__ == "__main__":
    main()
