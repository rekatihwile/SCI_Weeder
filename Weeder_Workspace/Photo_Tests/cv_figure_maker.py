from __future__ import annotations
"""
cv_figure_maker.py
==================
Loads a single stereo photo pair from Photo_Tests/left + right, runs the AI
detector (YOLO bboxes + keypoint heatmap), triangulates detections, plans the
path via TSP, and saves two fully-vector-annotated publication figures:

  <stem>__stitched_cv.pdf   – side-by-side stereo pair with FULLY VECTOR overlays:
                               per-class bounding boxes, confidence labels,
                               numbered matched stem keypoints, legend.
                               The JPEG photo pixels are the only rasterized
                               element; every annotation is an editable vector
                               object in Illustrator.

  <stem>__3d_path.pdf       – 3D scatter (Z-depth encoded), vertical drop-lines,
                               sensor-head footprint, TSP path with arrows.

Output directory:  Photo_Tests/figure_outputs/<stem>/
"""

# ============================================================
# USER SETTINGS  <- edit these
# ============================================================

PHOTO_NUMBER = 6           # integer index matching pt_XXXXXX in filenames

# Model weights inside Weeder_Workspace/params/ — None → use config.py defaults
YOLO_MODEL_OVERRIDE   = None   # e.g. "26_plastic_nano.pt"
QPOINT_MODEL_OVERRIDE = None   # e.g. "new_best_targeting_tall_plastic.pth"

DETECTOR_CONF       = 0.25   # YOLO confidence threshold
FLIP_Z              = True   # negate Z (sensor head sits at z = 0)
CAPTURE_BOX_SIZE_MM = 50.0   # sensor-head footprint half-width (mm)

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
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

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
from vision.detectors.ai_detector import _WeedCVCore
from vision.matching import match_points
from control.coarse_move import TriangulationCoarseMover
from planning.target_planner import plan_targets

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PHOTO_TESTS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH   = PHOTO_TESTS_DIR / "manifest.json"
OUTPUT_BASE_DIR = PHOTO_TESTS_DIR / "figure_outputs"


# ============================================================
# Class colour palette  (RGB 0-1, colour-blind-friendly-ish)
# ============================================================
_PALETTE_RGB = [
    (0.949, 0.561, 0.035),  # amber
    (0.200, 0.780, 0.349),  # green
    (0.239, 0.639, 0.898),  # sky-blue
    (0.847, 0.231, 0.231),  # red
    (0.643, 0.349, 0.847),  # purple
    (0.098, 0.718, 0.718),  # teal
    (0.902, 0.431, 0.098),  # orange
    (0.800, 0.800, 0.100),  # yellow
]

# Keypoint colours (RGB)
_KP_MATCHED_RGB   = (0.863, 0.078, 0.235)   # crimson
_KP_UNMATCHED_RGB = (0.550, 0.550, 0.550)   # mid-grey


def _build_class_color_map(dets_left: list[dict], dets_right: list[dict]) -> dict:
    """Return {cls_name: rgb_tuple} for every class seen across both frames."""
    seen: list[str] = sorted({d.get("cls_name", "unknown")
                               for d in dets_left + dets_right})
    return {name: _PALETTE_RGB[i % len(_PALETTE_RGB)]
            for i, name in enumerate(seen)}


# ============================================================
# Misc helpers
# ============================================================

def _resolve_model(override, config_key: str) -> Path:
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
    return json.loads(MANIFEST_PATH.read_text()).get("captures", [])


def _find_entry(captures: list[dict], photo_number: int) -> dict:
    for c in captures:
        if c.get("idx") == photo_number:
            return c
    available = sorted(c.get("idx") for c in captures)
    raise ValueError(
        f"Photo index {photo_number} not in manifest.  Available: {available}")


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


# ============================================================
# Figure 1 — stitched stereo pair  →  fully-vector PDF
# ============================================================
# Strategy
# --------
# • imshow the raw (un-annotated) photo pixels with rasterized=True so the JPEG
#   content is a high-res embedded bitmap.
# • Every annotation (bbox rectangles, confidence text, keypoint markers, number
#   labels, frame labels, legend) is drawn as a native matplotlib artist on top,
#   so it saves as a real vector object in the PDF and is selectable / editable
#   in Illustrator.
#
# Coordinate system
# -----------------
# imshow with default origin='upper' puts (0,0) at the top-left pixel.
# x increases rightward, y increases downward — identical to image pixel coords.
# Right-frame annotations are offset by left_width in x.
# ============================================================

def _draw_bbox_vector(ax, x1, y1, x2, y2, color_rgb, conf,
                      x_offset=0, linewidth=1.8, fontsize=7.5):
    """
    Draw one bounding box as a matplotlib Rectangle patch plus a confidence
    label as ax.text — both are fully vector elements in the saved PDF.
    """
    rx, ry = x1 + x_offset, y1
    rw, rh = x2 - x1, y2 - y1

    rect = Rectangle(
        (rx, ry), rw, rh,
        linewidth=linewidth,
        edgecolor=color_rgb,
        facecolor="none",
        zorder=8,
    )
    ax.add_patch(rect)

    # Confidence label — tiny filled box behind text for legibility
    ax.text(
        rx + 3, ry - 3,
        f"{conf:.2f}",
        color=color_rgb,
        fontsize=fontsize,
        va="bottom",
        ha="left",
        fontweight="bold",
        zorder=11,
        bbox=dict(facecolor="black", alpha=0.45, pad=1.2,
                  edgecolor="none", boxstyle="round,pad=0.15"),
    )


def _draw_keypoint_vector(ax, x, y, number, color_rgb, x_offset=0,
                          radius=7, fontsize=8.5):
    """
    Draw one keypoint as a filled circle + white ring + visit-order number,
    all as vector matplotlib artists.
    """
    px = x + x_offset

    # White ring (slightly larger circle behind the coloured dot)
    ax.plot(px, y, "o",
            markersize=radius * 2 + 2,
            color="white",
            markeredgewidth=0,
            zorder=9,
            )

    # Filled keypoint dot
    ax.plot(px, y, "o",
            markersize=radius * 2,
            color=color_rgb,
            markeredgewidth=0,
            zorder=10,
            )

    # Visit-order number
    ax.text(
        px + radius + 4, y - 1,
        str(number),
        color="white",
        fontsize=fontsize,
        va="center",
        ha="left",
        fontweight="bold",
        zorder=12,
        bbox=dict(facecolor="black", alpha=0.45, pad=1.0,
                  edgecolor="none", boxstyle="round,pad=0.15"),
    )


def _draw_unmatched_kp_vector(ax, x, y, x_offset=0, radius=5):
    """Draw a smaller grey keypoint (no number) for unmatched detections."""
    px = x + x_offset
    ax.plot(px, y, "o",
            markersize=radius * 2 + 2,
            color="white",
            markeredgewidth=0,
            zorder=9)
    ax.plot(px, y, "o",
            markersize=radius * 2,
            color=_KP_UNMATCHED_RGB,
            markeredgewidth=0,
            zorder=10)


def make_stitched_figure(
    left_bgr:        np.ndarray,
    right_bgr:       np.ndarray,
    left_dets:       list[dict],
    right_dets:      list[dict],
    matched_L_kps:   list[tuple],
    matched_R_kps:   list[tuple],
    unmatched_L_kps: list[tuple],
    unmatched_R_kps: list[tuple],
    tsp_L_order:     dict,          # {(x,y): visit_number}
    tsp_R_order:     dict,
    class_color_map: dict,          # {cls_name: rgb_tuple}
    n_matched:       int,
    out_path:        Path,
) -> None:
    """
    Save a side-by-side stereo pair PDF where the photo pixels are rasterized
    but EVERY annotation (boxes, labels, keypoints, numbers) is a live vector
    object editable in Illustrator.
    """
    # ------------------------------------------------------------------
    # 1.  Prepare raw RGB images (no cv2 annotation drawn on them)
    # ------------------------------------------------------------------
    left_rgb  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    hl, wl = left_rgb.shape[:2]
    hr, wr = right_rgb.shape[:2]

    # Pad smaller image vertically so heights match for hconcat
    if hl != hr:
        if hl < hr:
            pad = np.zeros((hr - hl, wl, 3), dtype=np.uint8)
            left_rgb = np.vstack([left_rgb, pad])
            hl = hr
        else:
            pad = np.zeros((hl - hr, wr, 3), dtype=np.uint8)
            right_rgb = np.vstack([right_rgb, pad])
            hr = hl

    combo = np.hstack([left_rgb, right_rgb])
    h_img, w_img = combo.shape[:2]

    # ------------------------------------------------------------------
    # 2.  Figure and axes
    # ------------------------------------------------------------------
    fig_w = 16.0
    fig_h = fig_w * h_img / w_img
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Photo pixels (only rasterized element)
    ax.imshow(combo, rasterized=True, zorder=1)
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)   # y=0 at top, matching image convention
    ax.axis("off")

    # ------------------------------------------------------------------
    # 3.  Bounding boxes — fully vector
    # ------------------------------------------------------------------
    matched_kp_sets = {
        "left":  set(matched_L_kps),
        "right": set(matched_R_kps),
    }

    for side, dets, x_off, kp_set in [
        ("left",  left_dets,  0,   matched_kp_sets["left"]),
        ("right", right_dets, wl,  matched_kp_sets["right"]),
    ]:
        for det in dets:
            x1, y1, x2, y2 = det["box"]
            kp    = det["keypoint"]
            color = class_color_map.get(det.get("cls_name", ""), (0.8, 0.8, 0.8))
            lw    = 2.0 if kp in kp_set else 1.0
            _draw_bbox_vector(ax, x1, y1, x2, y2, color, det["conf"],
                              x_offset=x_off, linewidth=lw)

    # ------------------------------------------------------------------
    # 4.  Unmatched keypoints — vector
    # ------------------------------------------------------------------
    for kp in unmatched_L_kps:
        _draw_unmatched_kp_vector(ax, kp[0], kp[1], x_offset=0)
    for kp in unmatched_R_kps:
        _draw_unmatched_kp_vector(ax, kp[0], kp[1], x_offset=wl)

    # ------------------------------------------------------------------
    # 5.  Matched + numbered keypoints — vector
    # ------------------------------------------------------------------
    for kp in matched_L_kps:
        num = tsp_L_order.get(kp, "?")
        _draw_keypoint_vector(ax, kp[0], kp[1], num, _KP_MATCHED_RGB, x_offset=0)
    for kp in matched_R_kps:
        num = tsp_R_order.get(kp, "?")
        _draw_keypoint_vector(ax, kp[0], kp[1], num, _KP_MATCHED_RGB, x_offset=wl)

    # ------------------------------------------------------------------
    # 6.  Frame divider + "LEFT" / "RIGHT" labels — vector
    # ------------------------------------------------------------------
    ax.axvline(x=wl, color="white", linewidth=1.2, linestyle="--",
               alpha=0.55, zorder=7)

    for label, cx in [("LEFT", wl * 0.5), ("RIGHT", wl + wr * 0.5)]:
        ax.text(cx, 22, label,
                color="white", fontsize=12, ha="center", va="top",
                fontweight="bold", zorder=13,
                bbox=dict(facecolor="black", alpha=0.45, pad=3,
                          edgecolor="none", boxstyle="round,pad=0.3"))

    # ------------------------------------------------------------------
    # 7.  Legend — vector
    # ------------------------------------------------------------------
    legend_handles: list = []

    seen_classes = sorted(class_color_map.keys())
    for cls_name in seen_classes:
        rgb = class_color_map[cls_name]
        legend_handles.append(
            mpatches.Patch(facecolor=rgb, edgecolor="white", linewidth=0.5,
                           label=cls_name.replace("_", " ").title())
        )

    legend_handles.append(
        mlines.Line2D([], [], marker="o", markersize=8, linestyle="none",
                      markerfacecolor=_KP_MATCHED_RGB,
                      markeredgecolor="white", markeredgewidth=0.8,
                      label=f"Matched stem keypoint ({n_matched})")
    )
    legend_handles.append(
        mlines.Line2D([], [], marker="o", markersize=6, linestyle="none",
                      markerfacecolor=_KP_UNMATCHED_RGB,
                      markeredgecolor="white", markeredgewidth=0.8,
                      label="Unmatched keypoint")
    )

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=9,
        facecolor="#111111",
        labelcolor="white",
        framealpha=0.72,
        edgecolor="none",
        zorder=14,
    )

    # ------------------------------------------------------------------
    # 8.  Save
    # ------------------------------------------------------------------
    fig.patch.set_facecolor("black")
    fig.tight_layout(pad=0.0)
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)
    print(f"  [fig1] saved -> {out_path.name}")


# ============================================================
# Figure 2 — 3-D triangulated targets + TSP path  →  PDF
# ============================================================

def _sensor_head_corners(cx, cy, half):
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    return ([x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            [0.0] * 5)


def make_3d_figure(solved_targets, planned_targets, ref_xy, stem, out_path):
    """
    3-D scatter (Z-depth colour-encoded + colorbar) with:
      • Vertical dashed drop-lines from each plant up to z=0 (sensor plane)
      • Shadow × projections at z=0
      • Orange sensor-head footprint box
      • TSP path with mid-segment arrows and visit-order labels
    XY axis limits = full workspace bounds.  No figure title.
    All elements are matplotlib artists → fully vector in PDF.
    """
    if not solved_targets:
        print("  [fig2] no solved targets — skipping")
        return

    xs, ys, zs = [], [], []
    for t in solved_targets:
        txy  = t.get("target_xy_mm")
        xrec = t.get("X_rect_m")
        if txy is None or xrec is None:
            continue
        z = float(xrec[2]) * 1000.0
        if FLIP_Z:
            z = -z
        xs.append(float(txy[0]))
        ys.append(float(txy[1]))
        zs.append(z)

    if not xs:
        print("  [fig2] all targets missing geometry — skipping")
        return

    xy_to_z = {
        (float(t["target_xy_mm"][0]), float(t["target_xy_mm"][1])): z
        for t, z in zip(solved_targets, zs)
        if t.get("target_xy_mm") is not None
    }

    fig = plt.figure(figsize=(10, 8.5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    # Drop-lines and shadow dots
    for x, y, z in zip(xs, ys, zs):
        ax.plot([x, x], [y, y], [z, 0.0],
                color="#b0b0b0", linewidth=0.7, linestyle="--",
                alpha=0.65, zorder=2)
    ax.scatter(xs, ys, [0.0] * len(xs),
               s=18, c="#aaaaaa", marker="x", linewidths=0.8,
               alpha=0.55, zorder=3)

    # Z-encoded scatter
    sc = ax.scatter(xs, ys, zs, s=90, c=zs, cmap="viridis",
                    edgecolors="white", linewidths=0.6,
                    zorder=5, label="Plants")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.45, pad=0.08)
    cbar.set_label("Z depth (mm)", size=9)
    cbar.ax.tick_params(labelsize=8)

    # Sensor-head box
    if ref_xy is not None:
        sx, sy, sz_ = _sensor_head_corners(
            ref_xy[0], ref_xy[1], CAPTURE_BOX_SIZE_MM / 2.0)
        ax.plot(sx, sy, sz_, color="#E87E04", linewidth=2.0,
                label="Sensor head (survey pose)", zorder=6)
        ax.text(ref_xy[0], ref_xy[1], 2.0,
                "capture", fontsize=8.5, color="#E87E04",
                ha="center", va="bottom")

    # TSP path
    plan_xy = [(float(pt["target_xy_mm"][0]), float(pt["target_xy_mm"][1]))
               for pt in planned_targets
               if pt.get("target_xy_mm") is not None]

    if len(plan_xy) >= 2:
        px_ = [p[0] for p in plan_xy]
        py_ = [p[1] for p in plan_xy]
        pz_ = [xy_to_z.get(p, 0.0) for p in plan_xy]

        ax.plot(px_, py_, pz_, color="#1a1a1a", linewidth=1.3,
                alpha=0.85, zorder=4)

        for i in range(len(plan_xy) - 1):
            x0_, y0_, z0_ = px_[i], py_[i], pz_[i]
            dx_ = px_[i + 1] - x0_
            dy_ = py_[i + 1] - y0_
            dz_ = pz_[i + 1] - z0_
            ax.quiver(x0_ + 0.5 * dx_, y0_ + 0.5 * dy_, z0_ + 0.5 * dz_,
                      dx_, dy_, dz_,
                      length=18, normalize=True,
                      color="#1a1a1a", arrow_length_ratio=0.45,
                      linewidth=0, zorder=4)

        for i, p in enumerate(plan_xy, start=1):
            ax.text(p[0], p[1], xy_to_z.get(p, 0.0) - 4,
                    str(i), fontsize=8.5, color="#1a1a1a", fontweight="bold",
                    ha="center", va="top", zorder=7)

    ax.set_xlim(WORKSPACE_X_MIN, WORKSPACE_X_MAX)
    ax.set_ylim(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
    ax.set_xlabel("X base (mm)", labelpad=8, fontsize=10)
    ax.set_ylabel("Y base (mm)", labelpad=8, fontsize=10)
    ax.set_zlabel("Z (mm)",      labelpad=8, fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.view_init(elev=22, azim=-58)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  [fig2] saved -> {out_path.name}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    # 1. Models
    yolo_path   = _resolve_model(YOLO_MODEL_OVERRIDE,   DEFAULT_PLASTIC_MODEL)
    qpoint_path = _resolve_model(QPOINT_MODEL_OVERRIDE, DEFAULT_PLASTIC_QPOINT_MODEL)
    print(f"YOLO model   : {yolo_path.name}")
    print(f"Qpoint model : {qpoint_path.name}")

    # 2. Manifest
    captures   = _load_manifest()
    entry      = _find_entry(captures, PHOTO_NUMBER)
    stem       = re.sub(r"_left\.jpg$", "", Path(entry["left_file"]).name)
    ref_xy     = (float(entry["x_mm"]), float(entry["y_mm"]))
    left_path  = PHOTO_TESTS_DIR / entry["left_file"]
    right_path = PHOTO_TESTS_DIR / entry["right_file"]

    print(f"\nPhoto : {stem}")
    print(f"Pose  : X={ref_xy[0]:.1f} mm  Y={ref_xy[1]:.1f} mm")

    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(
            f"Image file(s) missing.\n  L: {left_path}\n  R: {right_path}")

    # 3. Load raw frames
    left_bgr  = _load_bgr(left_path)
    right_bgr = _load_bgr(right_path)

    # 4. Detect
    print("\nLoading AI detector ...")
    core_L = _WeedCVCore(str(yolo_path), str(qpoint_path), conf=DETECTOR_CONF)
    core_R = _WeedCVCore(str(yolo_path), str(qpoint_path), conf=DETECTOR_CONF)

    print("Running detection ...")
    left_dets  = core_L.detect_with_visuals(left_bgr)
    right_dets = core_R.detect_with_visuals(right_bgr)

    print(f"  Left : {len(left_dets)} detections")
    print(f"  Right: {len(right_dets)} detections")

    class_color_map = _build_class_color_map(left_dets, right_dets)
    print(f"  Classes: {list(class_color_map.keys())}")

    # 5. Stereo matching
    left_kps  = [d["keypoint"] for d in left_dets]
    right_kps = [d["keypoint"] for d in right_dets]

    print("\nMatching stereo points ...")
    if left_kps and right_kps:
        matched_targets, unmatched_L_kps, unmatched_R_kps = \
            match_points(left_kps, right_kps, verbose=True)
    else:
        matched_targets  = []
        unmatched_L_kps  = list(left_kps)
        unmatched_R_kps  = list(right_kps)

    print(f"  Matched: {len(matched_targets)}")
    matched_L_kps = [m["left_px"]  for m in matched_targets]
    matched_R_kps = [m["right_px"] for m in matched_targets]

    # 6. Triangulate + TSP
    out_dir = OUTPUT_BASE_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if matched_targets:
        print("\nTriangulating ...")
        cm             = TriangulationCoarseMover()
        solved_targets = cm.solve_all_from_pose(
            matched_targets, ref_x=ref_xy[0], ref_y=ref_xy[1])

        for i, t in enumerate(solved_targets, 1):
            txy  = t["target_xy_mm"]
            xrec = t["X_rect_m"]
            z_mm = -float(xrec[2]) * 1000.0 if FLIP_Z else float(xrec[2]) * 1000.0
            print(f"  [{i}] X={txy[0]:.1f}  Y={txy[1]:.1f}  Z={z_mm:.1f} mm")

        print("\nPlanning TSP path ...")
        planned_targets = plan_targets(solved_targets, start_xy=ref_xy)

        tsp_L_order = {pt["source_target"]["left_px"]:  i
                       for i, pt in enumerate(planned_targets, 1)}
        tsp_R_order = {pt["source_target"]["right_px"]: i
                       for i, pt in enumerate(planned_targets, 1)}
    else:
        print("\nNo matched stereo pairs — skipping triangulation and 3-D plot.")
        solved_targets  = []
        planned_targets = []
        tsp_L_order     = {kp: i for i, kp in enumerate(matched_L_kps, 1)}
        tsp_R_order     = {kp: i for i, kp in enumerate(matched_R_kps, 1)}

    # 7. Stitched figure (fully vector annotations)
    fig1_path = out_dir / f"{stem}__stitched_cv.pdf"
    make_stitched_figure(
        left_bgr, right_bgr,
        left_dets, right_dets,
        matched_L_kps, matched_R_kps,
        unmatched_L_kps, unmatched_R_kps,
        tsp_L_order, tsp_R_order,
        class_color_map,
        n_matched=len(matched_targets),
        out_path=fig1_path,
    )

    # 8. 3-D figure
    if solved_targets:
        fig2_path = out_dir / f"{stem}__3d_path.pdf"
        make_3d_figure(solved_targets, planned_targets, ref_xy, stem, fig2_path)

    print(f"\nDone.  Outputs in:\n  {out_dir}")


if __name__ == "__main__":
    main()