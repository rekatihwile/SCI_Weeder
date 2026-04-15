from __future__ import annotations
"""
cv_figure_maker.py
==================
For a chosen Photo_Tests stereo pair, this script:

  1. Loads the left + right JPEG images and the capture pose from manifest.json
  2. Runs the AI detector (YOLO bboxes + keypoint heatmap) on both frames
  3. Saves Figure 1 — stitched stereo image with bbox + keypoint overlays
  4. Triangulates the stereo-matched detections
  5. Plans a TSP visit order
  6. Saves Figure 2 — 3-D scatter of triangulated targets with TSP path and
     sensor-head footprint rectangle, axis limits = full workspace bounds

Outputs land in:
    Photo_Tests/figure_outputs/<stem>/
        <stem>__stitched_cv.png
        <stem>__3d_path.png

Edit the USER SETTINGS block below to select a different photo or tweak
model / visual parameters.
"""

# ============================================================
# USER SETTINGS  ← edit these
# ============================================================

PHOTO_NUMBER = 6         # integer index (matches pt_XXXXXX in filename)

# Model weights (filenames inside SCI_Weeder/Weeder_Workspace/params/)
# Defaults pull the new plastic models set in config.py; override here if needed.
YOLO_MODEL_OVERRIDE    = None   # e.g. "26_plastic_nano.pt"  or None → use config default
QPOINT_MODEL_OVERRIDE  = None   # e.g. "new_best_targeting_tall_plastic.pth" or None

DETECTOR_CONF          = 0.25   # YOLO confidence threshold
FLIP_Z                 = True   # flip Z so depth points downward in the 3-D plot
CAPTURE_BOX_SIZE_MM    = 50.0   # sensor-head footprint half-width (mm) shown in 3-D plot

# ============================================================

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    MODEL_MAP,
    DEFAULT_PLASTIC_MODEL,
    DEFAULT_PLASTIC_QPOINT_MODEL,
    BASE_DIR,
)
from vision.detectors.ai_detector import AIDetector, _WeedCVCore
from vision.matching import match_points
from control.coarse_move import TriangulationCoarseMover
from planning.target_planner import plan_targets

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PHOTO_TESTS_DIR = Path(__file__).resolve().parent
LEFT_DIR        = PHOTO_TESTS_DIR / "left"
RIGHT_DIR       = PHOTO_TESTS_DIR / "right"
MANIFEST_PATH   = PHOTO_TESTS_DIR / "manifest.json"
OUTPUT_BASE_DIR = PHOTO_TESTS_DIR / "figure_outputs"


# ============================================================
# Helpers
# ============================================================

def _resolve_model(override: str | None, config_key: str) -> Path:
    """Return absolute path to a model weight file."""
    params_dir = BASE_DIR / "params"
    name = override if override else MODEL_MAP.get(config_key, config_key)
    p = Path(name)
    if not p.is_absolute():
        p = params_dir / p
    if not p.exists():
        raise FileNotFoundError(f"Model weight not found: {p}")
    return p


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    data = json.loads(MANIFEST_PATH.read_text())
    return data.get("captures", [])


def _find_entry(captures: list[dict], photo_number: int) -> dict:
    for c in captures:
        if c.get("idx") == photo_number:
            return c
    raise ValueError(
        f"Photo number {photo_number} not found in manifest "
        f"({MANIFEST_PATH}).  Available indices: "
        f"{sorted(c.get('idx') for c in captures)}"
    )


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


# ============================================================
# Annotation helpers (all drawing in BGR on the cv2 frame)
# ============================================================

# Colour palette (BGR)
_COL_BOX_MATCHED   = (0,   200, 255)   # amber
_COL_BOX_UNMATCHED = (180, 180, 180)   # grey
_COL_KP_MATCHED    = (0,   0,   220)   # red
_COL_KP_UNMATCHED  = (180, 90,  0  )   # dark-cyan
_COL_TEXT          = (255, 255, 255)   # white
_COL_TEXT_SHADOW   = (0,   0,   0  )   # black


def _draw_label(img, text, x, y, fg=_COL_TEXT, bg=_COL_TEXT_SHADOW,
                scale=0.55, thick=1):
    """Draw text with a dark shadow for readability on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x + 1, y + 1), font, scale, bg,    thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x,     y    ), font, scale, fg,    thick,     cv2.LINE_AA)


def annotate_frame(
    frame:          np.ndarray,
    detections:     list[dict],
    matched_kps:    list[tuple],
    unmatched_kps:  list[tuple],
    matched_order:  dict,          # keypoint → visit-order label (1-based)
) -> np.ndarray:
    """
    Draw YOLO bounding boxes and keypoints onto a copy of *frame*.

    Parameters
    ----------
    frame          BGR image
    detections     output of _WeedCVCore.detect_with_visuals()
    matched_kps    list of (x, y) keypoints that were stereo-matched
    unmatched_kps  list of (x, y) keypoints that had no stereo match
    matched_order  {(x,y): label_int} mapping so matched points get a number
    """
    out = frame.copy()

    # --- bounding boxes (all detections) ---
    matched_kp_set = set(matched_kps)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        kp = det["keypoint"]
        is_matched = kp in matched_kp_set

        col = _COL_BOX_MATCHED if is_matched else _COL_BOX_UNMATCHED
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        _draw_label(out, f"{det['conf']:.2f}", x1 + 3, y1 - 5, fg=col)

    # --- unmatched keypoints ---
    for kp in unmatched_kps:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(out, (x, y), 6, _COL_KP_UNMATCHED, -1)
        cv2.circle(out, (x, y), 6, _COL_TEXT, 1)

    # --- matched keypoints (numbered) ---
    for kp in matched_kps:
        x, y = int(kp[0]), int(kp[1])
        label = str(matched_order.get(kp, "?"))
        cv2.circle(out, (x, y), 8,  _COL_KP_MATCHED, -1)
        cv2.circle(out, (x, y), 8,  _COL_TEXT, 1)
        _draw_label(out, label, x + 10, y - 6, scale=0.65, thick=2)

    return out


# ============================================================
# Figure 1 – stitched stereo pair with CV overlays
# ============================================================

def make_stitched_figure(
    left_ann:   np.ndarray,
    right_ann:  np.ndarray,
    stem:       str,
    out_path:   Path,
    n_left:     int,
    n_right:    int,
    n_matched:  int,
):
    """Save side-by-side annotated left + right images as a single PNG."""
    # Ensure same height before hconcat
    hl, wl = left_ann.shape[:2]
    hr, wr = right_ann.shape[:2]
    if hl != hr:
        scale = hl / hr
        right_ann = cv2.resize(right_ann, (int(wr * scale), hl))

    combo_bgr = cv2.hconcat([left_ann, right_ann])
    combo_rgb = cv2.cvtColor(combo_bgr, cv2.COLOR_BGR2RGB)

    h, w = combo_rgb.shape[:2]
    fig_w = 16
    fig_h = fig_w * h / w

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(combo_rgb)
    ax.axis("off")

    # Divider line between left and right
    ax.axvline(x=wl, color="white", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.text(wl / 2, 20, "LEFT", color="white", fontsize=11,
            ha="center", va="top", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"))
    ax.text(wl + wr / 2, 20, "RIGHT", color="white", fontsize=11,
            ha="center", va="top", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"))

    # Legend patches
    legend_elements = [
        mpatches.Patch(facecolor=tuple(c / 255 for c in _COL_BOX_MATCHED[::-1]),
                       label=f"Matched detection ({n_matched})"),
        mpatches.Patch(facecolor=tuple(c / 255 for c in _COL_BOX_UNMATCHED[::-1]),
                       label=f"Unmatched detection"),
        mpatches.Patch(facecolor=tuple(c / 255 for c in _COL_KP_MATCHED[::-1]),
                       label="Matched stem keypoint"),
        mpatches.Patch(facecolor=tuple(c / 255 for c in _COL_KP_UNMATCHED[::-1]),
                       label="Unmatched keypoint"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=9, facecolor="black", labelcolor="white",
              framealpha=0.6, edgecolor="none")

    fig.suptitle(
        f"Stereo Pixel Input  |  {stem}  |  "
        f"L={n_left}  R={n_right}  matched={n_matched}",
        fontsize=12, color="white", y=0.995,
    )
    fig.patch.set_facecolor("black")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  [fig1] saved → {out_path.name}")


# ============================================================
# Figure 2 – 3-D triangulated targets + TSP path + sensor box
# ============================================================

def _sensor_head_box_coords(cx: float, cy: float, half: float) -> tuple:
    """Return xs, ys, zs for the four sides of the sensor-head rectangle at z=0."""
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    zs = [0.0] * 5
    return xs, ys, zs


def make_3d_figure(
    solved_targets: list[dict],
    planned_targets: list[dict],
    ref_xy:         tuple[float, float],
    stem:           str,
    out_path:       Path,
):
    """
    3-D scatter of triangulated plant targets with TSP path and sensor-head box.
    XY axes span the full workspace; Z auto-ranges to the data.
    """
    if not solved_targets:
        print("  [fig2] no solved targets → skipping 3-D plot")
        return

    # --- collect XYZ for scatter ---
    xs, ys, zs = [], [], []
    for t in solved_targets:
        txy = t.get("target_xy_mm")
        x_rect = t.get("X_rect_m")
        if txy is None or x_rect is None:
            continue
        z_mm = float(x_rect[2]) * 1000.0
        if FLIP_Z:
            z_mm = -z_mm
        xs.append(float(txy[0]))
        ys.append(float(txy[1]))
        zs.append(z_mm)

    if not xs:
        print("  [fig2] all targets missing geometry → skipping 3-D plot")
        return

    # --- build XYZ lookup keyed by target_xy_mm for TSP path drawing ---
    xy_to_z: dict[tuple, float] = {}
    for t, z in zip(solved_targets, zs):
        txy = t.get("target_xy_mm")
        if txy is not None:
            xy_to_z[(float(txy[0]), float(txy[1]))] = z

    # --- figure setup ---
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # --- scatter: plants ---
    ax.scatter(xs, ys, zs, s=60, color="#4DA9FF", zorder=5, label="plants")

    # --- number each scatter point ---
    for i, (x, y, z) in enumerate(zip(xs, ys, zs), start=1):
        ax.text(x, y, z, f" {i}", fontsize=8, color="#4DA9FF")

    # --- sensor-head footprint box ---
    if ref_xy is not None:
        sx, sy, sz = _sensor_head_box_coords(ref_xy[0], ref_xy[1],
                                              CAPTURE_BOX_SIZE_MM / 2.0)
        ax.plot(sx, sy, sz, color="#FFA500", linewidth=2, label="Sensor head (survey pose)")
        ax.text(ref_xy[0], ref_xy[1], 0.0, "capture",
                fontsize=9, color="#FFA500")

    # --- TSP path ---
    plan_xy = []
    for pt in planned_targets:
        txy = pt.get("target_xy_mm")
        if txy is not None:
            plan_xy.append(txy)

    if len(plan_xy) >= 2:
        path_xs = [p[0] for p in plan_xy]
        path_ys = [p[1] for p in plan_xy]
        path_zs = [xy_to_z.get((float(p[0]), float(p[1])), 0.0) for p in plan_xy]

        # Lines connecting each step
        ax.plot(path_xs, path_ys, path_zs,
                color="black", linewidth=1.2, zorder=4)

        # Arrows (quiver) for direction  — one per segment
        for i in range(len(plan_xy) - 1):
            x0_, y0_, z0_ = path_xs[i], path_ys[i], path_zs[i]
            dx_ = path_xs[i + 1] - x0_
            dy_ = path_ys[i + 1] - y0_
            dz_ = path_zs[i + 1] - z0_
            ax.quiver(x0_, y0_, z0_, dx_, dy_, dz_,
                      length=0.6, normalize=True,
                      color="black", arrow_length_ratio=0.25,
                      linewidth=0)

        # Number each planned stop
        for i, (px_, py_) in enumerate(plan_xy, start=1):
            pz_ = xy_to_z.get((float(px_), float(py_)), 0.0)
            ax.text(px_, py_, pz_ + 3,
                    str(i), fontsize=9, color="black", fontweight="bold",
                    ha="center", va="bottom")

    # --- axes & labels ---
    ax.set_xlim(WORKSPACE_X_MIN, WORKSPACE_X_MAX)
    ax.set_ylim(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
    ax.set_xlabel("X base (mm)")
    ax.set_ylabel("Y base (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Triangulated 3D points + TSP path  |  {stem}")
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig2] saved → {out_path.name}")


# ============================================================
# Main
# ============================================================

def main():
    # ------------------------------------------------------------------
    # 1. Resolve model paths
    # ------------------------------------------------------------------
    yolo_path   = _resolve_model(YOLO_MODEL_OVERRIDE,   DEFAULT_PLASTIC_MODEL)
    qpoint_path = _resolve_model(QPOINT_MODEL_OVERRIDE, DEFAULT_PLASTIC_QPOINT_MODEL)
    print(f"YOLO model   : {yolo_path.name}")
    print(f"Qpoint model : {qpoint_path.name}")

    # ------------------------------------------------------------------
    # 2. Find the requested photo in the manifest
    # ------------------------------------------------------------------
    captures = _load_manifest()
    entry    = _find_entry(captures, PHOTO_NUMBER)
    stem     = re.sub(r"_left\.jpg$", "",
                      Path(entry["left_file"]).name)
    ref_xy   = (float(entry["x_mm"]), float(entry["y_mm"]))

    left_path  = PHOTO_TESTS_DIR / entry["left_file"]
    right_path = PHOTO_TESTS_DIR / entry["right_file"]

    print(f"\nPhoto  : {stem}")
    print(f"Left   : {left_path}")
    print(f"Right  : {right_path}")
    print(f"Pose   : X={ref_xy[0]:.1f} mm  Y={ref_xy[1]:.1f} mm")

    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(
            "Image file(s) missing. "
            f"Left: {left_path.exists()}  Right: {right_path.exists()}"
        )

    # ------------------------------------------------------------------
    # 3. Load frames
    # ------------------------------------------------------------------
    left_bgr  = _load_bgr(left_path)
    right_bgr = _load_bgr(right_path)

    # ------------------------------------------------------------------
    # 4. Run the detector
    # ------------------------------------------------------------------
    print("\nLoading AI detector …")
    core_left  = _WeedCVCore(str(yolo_path), str(qpoint_path), conf=DETECTOR_CONF)
    core_right = _WeedCVCore(str(yolo_path), str(qpoint_path), conf=DETECTOR_CONF)

    print("Running detection on left frame …")
    left_dets  = core_left.detect_with_visuals(left_bgr)
    print("Running detection on right frame …")
    right_dets = core_right.detect_with_visuals(right_bgr)

    left_kps  = [d["keypoint"] for d in left_dets]
    right_kps = [d["keypoint"] for d in right_dets]

    print(f"  Left detections : {len(left_dets)}")
    print(f"  Right detections: {len(right_dets)}")

    # ------------------------------------------------------------------
    # 5. Stereo matching
    # ------------------------------------------------------------------
    print("\nMatching stereo points …")
    if left_kps and right_kps:
        matched_targets, unmatched_left_kps, unmatched_right_kps = \
            match_points(left_kps, right_kps, verbose=True)
    else:
        matched_targets        = []
        unmatched_left_kps     = list(left_kps)
        unmatched_right_kps    = list(right_kps)

    print(f"  Matched pairs   : {len(matched_targets)}")

    # ------------------------------------------------------------------
    # 6. Annotate frames and build stitched figure
    # ------------------------------------------------------------------
    # Build order mapping: left_px → visit number (filled after TSP, for now 1-N)
    matched_left_kps  = [m["left_px"]  for m in matched_targets]
    matched_right_kps = [m["right_px"] for m in matched_targets]

    # Provisional 1-based labels before TSP reorder
    left_order  = {kp: i for i, kp in enumerate(matched_left_kps,  start=1)}
    right_order = {kp: i for i, kp in enumerate(matched_right_kps, start=1)}

    left_ann  = annotate_frame(left_bgr,  left_dets,  matched_left_kps,
                               unmatched_left_kps,  left_order)
    right_ann = annotate_frame(right_bgr, right_dets, matched_right_kps,
                               unmatched_right_kps, right_order)

    out_dir = OUTPUT_BASE_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_path = out_dir / f"{stem}__stitched_cv.png"
    make_stitched_figure(
        left_ann, right_ann, stem, fig1_path,
        n_left=len(left_dets), n_right=len(right_dets),
        n_matched=len(matched_targets),
    )

    # ------------------------------------------------------------------
    # 7. Triangulation
    # ------------------------------------------------------------------
    if not matched_targets:
        print("\nNo matched stereo pairs → skipping triangulation and 3-D plot.")
        return

    print("\nTriangulating …")
    cm = TriangulationCoarseMover()
    solved_targets = cm.solve_all_from_pose(matched_targets,
                                            ref_x=ref_xy[0], ref_y=ref_xy[1])

    for i, t in enumerate(solved_targets, start=1):
        txy = t.get("target_xy_mm")
        x_rect = t.get("X_rect_m")
        z_mm = -float(x_rect[2]) * 1000.0 if FLIP_Z else float(x_rect[2]) * 1000.0
        print(f"  [{i}] X={txy[0]:.1f}  Y={txy[1]:.1f}  Z={z_mm:.1f} mm")

    # ------------------------------------------------------------------
    # 8. TSP planning
    # ------------------------------------------------------------------
    print("\nPlanning TSP path …")
    planned_targets = plan_targets(solved_targets, start_xy=ref_xy)

    # Remap labels to TSP order for the stitched figure (re-annotate & re-save)
    planned_left_kps = [pt["source_target"]["left_px"] for pt in planned_targets]
    tsp_left_order   = {kp: i for i, kp in enumerate(planned_left_kps, start=1)}
    planned_right_kps = [pt["source_target"]["right_px"] for pt in planned_targets]
    tsp_right_order   = {kp: i for i, kp in enumerate(planned_right_kps, start=1)}

    left_ann  = annotate_frame(left_bgr,  left_dets,  matched_left_kps,
                               unmatched_left_kps,  tsp_left_order)
    right_ann = annotate_frame(right_bgr, right_dets, matched_right_kps,
                               unmatched_right_kps, tsp_right_order)
    make_stitched_figure(
        left_ann, right_ann, stem, fig1_path,
        n_left=len(left_dets), n_right=len(right_dets),
        n_matched=len(matched_targets),
    )

    # ------------------------------------------------------------------
    # 9. 3-D figure
    # ------------------------------------------------------------------
    fig2_path = out_dir / f"{stem}__3d_path.png"
    make_3d_figure(solved_targets, planned_targets, ref_xy, stem, fig2_path)

    print(f"\nDone.  Outputs in:\n  {out_dir}")


if __name__ == "__main__":
    main()