"""
bringup/09_class_filter_diagnostic.py
---------------------------------------
Survey burst diagnostic: detects ALL classes, then shows per-class breakdown
and how TARGET_CLASSES + AVOID_CLASSES would filter the result.

Useful for verifying that:
  - Avoid-class detections correctly suppress overlapping target detections
  - TARGET_CLASSES keeps only the intended classes
  - No target plants are being incorrectly suppressed

Run with:
    ./run_with_eli_venv.sh bringup/09_class_filter_diagnostic.py | tee bringup/logs/09_class_filter_diagnostic.log
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util as _ilu
_patch_path = Path(__file__).resolve().parent / "_nms_patch.py"
_spec = _ilu.spec_from_file_location("_nms_patch", _patch_path)
_mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FLUSH_FRAMES = 8

# Box colors for visualization (BGR)
_COLOR_TARGET   = (0,   200,   0)    # green  — target, kept
_COLOR_AVOID    = (0,   0,   220)    # red    — avoid class
_COLOR_OTHER    = (0,   180, 220)    # yellow — non-target, non-avoid
_COLOR_SUPPRESS = (0,   100, 220)    # orange — target suppressed by avoid


def _fmt_box(box):
    x1, y1, x2, y2 = box
    w = int(x2 - x1)
    h = int(y2 - y1)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return f"center=({cx},{cy}) size={w}x{h}"


def _analyse_stable(stable_all, target_classes, avoid_classes, core, cls_names, label):
    """
    Print a full breakdown of stable detections (pre-filter) and show what survives
    the TARGET_CLASSES / AVOID_CLASSES filter.

    Returns (kept, suppressed_by_avoid, dropped_by_class) lists.
    """
    iom_thresh = core.iom_thresh

    avoid_entries  = [s for s in stable_all if s["cls"] in avoid_classes]
    target_entries = [s for s in stable_all if
                      target_classes is None or s["cls"] in target_classes]
    target_entries = [s for s in target_entries if s["cls"] not in avoid_classes]
    other_entries  = [s for s in stable_all if
                      s["cls"] not in (avoid_classes or []) and
                      (target_classes is not None and s["cls"] not in target_classes)]

    # Which targets are suppressed by an avoid-class detection?
    suppressed_by_avoid = []
    surviving_targets   = []
    for s in target_entries:
        dominated = any(
            core._iom(s["box"], a["box"]) >= iom_thresh and a["conf"] > s["conf"]
            for a in avoid_entries
        )
        if dominated:
            suppressed_by_avoid.append(s)
        else:
            surviving_targets.append(s)

    # ---- Print header ----
    sep = "─" * 72
    print(f"\n{'═' * 72}")
    print(f"  {label}  —  ALL-CLASS STABLE DETECTIONS  ({len(stable_all)} total)")
    print(f"{'═' * 72}")

    if not stable_all:
        print("  (no detections)")
        return [], [], []

    # Full detection table
    print(f"  {'#':>3}  {'cls':>4}  {'name':<18}  {'conf':>6}  {'views':>5}  {'box'}")
    print(f"  {sep}")
    for i, s in enumerate(sorted(stable_all, key=lambda x: x["box"][0])):
        cls_id   = s["cls"]
        cls_name = cls_names.get(cls_id, f"cls{cls_id}")
        conf     = s["conf"]
        views    = s["views"]
        box_str  = _fmt_box(s["box"])

        if cls_id in (avoid_classes or []):
            tag = " [AVOID]"
        elif s in suppressed_by_avoid:
            tag = " [SUPPRESSED by avoid]"
        elif target_classes is not None and cls_id not in target_classes:
            tag = " [non-target class]"
        else:
            tag = " [TARGET ✓]"

        print(f"  {i:>3}  {cls_id:>4}  {cls_name:<18}  {conf:>6.3f}  {views:>5}  {box_str}{tag}")

    # ---- Per-class summary ----
    print(f"\n  {sep}")
    print("  CLASS BREAKDOWN:")
    cls_counter = Counter(s["cls"] for s in stable_all)
    for cls_id, count in sorted(cls_counter.items()):
        name = cls_names.get(cls_id, f"cls{cls_id}")
        role = ""
        if cls_id in (avoid_classes or []):
            role = "AVOID"
        elif target_classes is None or cls_id in target_classes:
            role = "TARGET"
        else:
            role = "other"
        print(f"    class {cls_id:>3} ({name:<16})  {count:>3} detection(s)  [{role}]")

    # ---- Filter outcome ----
    print(f"\n  FILTER OUTCOME  (TARGET_CLASSES={target_classes}  AVOID_CLASSES={avoid_classes}):")
    print(f"    Kept (targets, not suppressed) : {len(surviving_targets)}")
    print(f"    Suppressed by avoid-class      : {len(suppressed_by_avoid)}")
    print(f"    Dropped (non-target class)     : {len(other_entries)}")
    print(f"    Avoid-class detections (unseen): {len(avoid_entries)}")

    if suppressed_by_avoid:
        print("\n  SUPPRESSION DETAILS:")
        for s in suppressed_by_avoid:
            cls_name = cls_names.get(s["cls"], f"cls{s['cls']}")
            # Find the dominating avoid detection
            dominators = [
                a for a in avoid_entries
                if core._iom(s["box"], a["box"]) >= iom_thresh and a["conf"] > s["conf"]
            ]
            for a in dominators:
                avoid_name = cls_names.get(a["cls"], f"cls{a['cls']}")
                iom_val = core._iom(s["box"], a["box"])
                print(
                    f"    {cls_name} conf={s['conf']:.3f}  ←suppressed by→  "
                    f"{avoid_name} conf={a['conf']:.3f}  IoM={iom_val:.2f}"
                )

    return surviving_targets, suppressed_by_avoid, other_entries


def _draw_diagnostic(frame, stable_all, surviving_targets, suppressed_by_avoid,
                     avoid_entries, cls_names, offset_xy=(0, 0)):
    """Draw all detections with color-coded boxes showing filter outcome."""
    out = frame.copy()
    ox, oy = offset_xy

    suppress_set = {id(s) for s in suppressed_by_avoid}
    avoid_set    = {id(s) for s in avoid_entries}
    kept_set     = {id(s) for s in surviving_targets}

    for s in stable_all:
        x1 = int(s["box"][0]) + ox
        y1 = int(s["box"][1]) + oy
        x2 = int(s["box"][2]) + ox
        y2 = int(s["box"][3]) + oy
        cls_name = cls_names.get(s["cls"], f"cls{s['cls']}")
        label    = f"{cls_name} {s['conf']:.2f}"

        if id(s) in avoid_set:
            color = _COLOR_AVOID
            label += " [avoid]"
        elif id(s) in suppress_set:
            color = _COLOR_SUPPRESS
            label += " [suppressed]"
        elif id(s) in kept_set:
            color = _COLOR_TARGET
        else:
            color = _COLOR_OTHER
            label += " [non-target]"

        thickness = 2 if id(s) in kept_set else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        # X over suppressed targets
        if id(s) in suppress_set:
            cv2.line(out, (x1, y1), (x2, y2), _COLOR_SUPPRESS, 2)
            cv2.line(out, (x2, y1), (x1, y2), _COLOR_SUPPRESS, 2)

    return out


def main():
    from config import (
        TARGET_CLASSES,
        AVOID_CLASSES,
        SURVEY_BURST_COUNT,
        SURVEY_MIN_HITS,
        SURVEY_TARGET_CLASSES,
        AI_CONFIDENCE,
        SURVEY_CONFIDENCE_OVERRIDE,
    )
    from config.survey_params import resolve_burst_count
    from vision.detectors.ai_detector import AIDetector
    from hardware.cameras import StereoCameras

    print("=" * 72)
    print("BRINGUP 09 — Class Filter Diagnostic")
    print("=" * 72)
    print()
    print(f"  TARGET_CLASSES         : {TARGET_CLASSES}")
    print(f"  AVOID_CLASSES          : {AVOID_CLASSES}")
    print(f"  SURVEY_TARGET_CLASSES  : {SURVEY_TARGET_CLASSES}  (survey-level override)")
    print(f"  SURVEY_BURST_COUNT     : {SURVEY_BURST_COUNT}")
    print(f"  SURVEY_MIN_HITS        : {SURVEY_MIN_HITS}")
    conf = SURVEY_CONFIDENCE_OVERRIDE if SURVEY_CONFIDENCE_OVERRIDE is not None else AI_CONFIDENCE
    print(f"  Effective confidence   : {conf}  (survey_override={SURVEY_CONFIDENCE_OVERRIDE})")

    print("\n--- Building detector ---")
    detector = AIDetector()
    cls_names = detector.cv_left.yolo.names  # {int: str}
    print(f"  Model classes: {cls_names}")

    print("\n--- Detector warmup ---")
    detector.warmup()

    print("\n--- Opening cameras ---")
    cameras = StereoCameras()
    try:
        cameras.open(start_recorder=False)
    except RuntimeError:
        cameras.recover()

    # Flush stale buffer frames
    print(f"\n--- Flushing {FLUSH_FRAMES} buffer frame(s) ---")
    for _ in range(FLUSH_FRAMES):
        cameras.read_pair()

    # Collect burst frames
    burst_count = resolve_burst_count(SURVEY_BURST_COUNT)
    print(f"\n--- Collecting burst ({burst_count} frame(s)) ---")
    left_frames, right_frames = [], []
    for _ in range(burst_count):
        lf, rf = cameras.read_pair()
        left_frames.append(lf)
        right_frames.append(rf)
    print(f"  Captured {len(left_frames)} left / {len(right_frames)} right frames")
    cameras.close()
    print("  Cameras closed.")

    # Override conf if survey override is set
    if SURVEY_CONFIDENCE_OVERRIDE is not None:
        detector.cv_left.conf  = SURVEY_CONFIDENCE_OVERRIDE
        detector.cv_right.conf = SURVEY_CONFIDENCE_OVERRIDE

    # Run burst with ALL classes (target_classes=None, avoid_classes=[])
    # so we can see the full picture before any filtering.
    print("\n--- Running burst with ALL classes (no filter) ---")
    _orig_tc_l, _orig_ac_l = detector.cv_left.target_classes,  detector.cv_left.avoid_classes
    _orig_tc_r, _orig_ac_r = detector.cv_right.target_classes, detector.cv_right.avoid_classes
    try:
        detector.cv_left.target_classes  = None
        detector.cv_left.avoid_classes   = []
        detector.cv_right.target_classes = None
        detector.cv_right.avoid_classes  = []

        t0 = time.perf_counter()
        stable_left_all = detector.cv_left.return_burst_stable(
            left_frames,
            min_stable_views=SURVEY_MIN_HITS,
            classes_override=None,
            debug_label="[DIAG] LEFT",
            point_mode="box_center",
        )
        stable_right_all = detector.cv_right.return_burst_stable(
            right_frames,
            min_stable_views=SURVEY_MIN_HITS,
            classes_override=None,
            debug_label="[DIAG] RIGHT",
            point_mode="box_center",
        )
        burst_dt = time.perf_counter() - t0
    finally:
        detector.cv_left.target_classes  = _orig_tc_l
        detector.cv_left.avoid_classes   = _orig_ac_l
        detector.cv_right.target_classes = _orig_tc_r
        detector.cv_right.avoid_classes  = _orig_ac_r

    print(f"  Burst done in {burst_dt:.2f}s")
    print(f"  Raw stable: LEFT={len(stable_left_all)}  RIGHT={len(stable_right_all)}")

    # Restore original avoid_classes for analysis (they were wiped during burst)
    # Use config values directly for the analysis.
    avoid_classes  = list(AVOID_CLASSES or [])
    target_classes = list(TARGET_CLASSES) if TARGET_CLASSES is not None else None

    # ---- Analyse LEFT ----
    kept_L, supp_L, other_L = _analyse_stable(
        stable_left_all, target_classes, avoid_classes,
        detector.cv_left, cls_names, "LEFT CAMERA"
    )
    avoid_L = [s for s in stable_left_all if s["cls"] in avoid_classes]

    # ---- Analyse RIGHT ----
    kept_R, supp_R, other_R = _analyse_stable(
        stable_right_all, target_classes, avoid_classes,
        detector.cv_right, cls_names, "RIGHT CAMERA"
    )
    avoid_R = [s for s in stable_right_all if s["cls"] in avoid_classes]

    # ---- Overall summary ----
    print(f"\n{'═' * 72}")
    print("  OVERALL SUMMARY")
    print(f"{'═' * 72}")
    print(f"  {'':30}  {'LEFT':>6}  {'RIGHT':>6}")
    print(f"  {'─'*44}")
    print(f"  {'Raw stable detections (all cls)':30}  {len(stable_left_all):>6}  {len(stable_right_all):>6}")
    print(f"  {'→ Kept (target, not suppressed)':30}  {len(kept_L):>6}  {len(kept_R):>6}")
    print(f"  {'→ Suppressed by avoid-class':30}  {len(supp_L):>6}  {len(supp_R):>6}")
    print(f"  {'→ Dropped (non-target class)':30}  {len(other_L):>6}  {len(other_R):>6}")

    # ---- Save visualization ----
    print(f"\n--- Saving diagnostic images ---")
    for side_label, frames, stable_all, kept, supp, avoid_ents, out_name in [
        ("LEFT",  left_frames,  stable_left_all,  kept_L, supp_L, avoid_L, "09_left_diag.jpg"),
        ("RIGHT", right_frames, stable_right_all, kept_R, supp_R, avoid_R, "09_right_diag.jpg"),
    ]:
        frame = frames[-1].copy()
        vis = _draw_diagnostic(frame, stable_all, kept, supp, avoid_ents, cls_names)

        # Legend
        legend_items = [
            (_COLOR_TARGET,   "TARGET (kept)"),
            (_COLOR_AVOID,    "AVOID class"),
            (_COLOR_SUPPRESS, "Suppressed by avoid"),
            (_COLOR_OTHER,    "Non-target class"),
        ]
        lx, ly = 10, 10
        for color, text in legend_items:
            cv2.rectangle(vis, (lx, ly), (lx + 14, ly + 14), color, -1)
            cv2.putText(vis, text, (lx + 18, ly + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            ly += 18

        out_path = LOGS_DIR / out_name
        cv2.imwrite(str(out_path), vis)
        print(f"  {side_label}: {out_path}")

    print()
    print(f"  Filter: TARGET_CLASSES={target_classes}  AVOID_CLASSES={avoid_classes}")
    print(f"  {'═' * 68}")
    total_kept = len(kept_L) + len(kept_R)
    total_supp = len(supp_L) + len(supp_R)
    if total_supp > 0:
        print(f"  *** {total_supp} target detection(s) SUPPRESSED by avoid-class overlap ***")
    if total_kept == 0:
        print("  RESULT: No target detections (0 is OK if no plants visible)")
    else:
        print(f"  RESULT: {total_kept} target detection(s) kept  "
              f"({len(kept_L)} left / {len(kept_R)} right)")
    print(f"  {'═' * 68}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
