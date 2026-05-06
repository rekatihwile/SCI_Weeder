"""Manual triangulation debug page — rectified stereo pair click → match → 3D plot.

Routes:
    GET  /triangulate                  — render page
    POST /api/triangulate/capture      — grab rectified frame pair; return b64 + dims
    POST /api/triangulate/rect_to_raw  — map a rectified pixel coord → raw camera coord
    POST /api/triangulate/run          — match + triangulate + 3D matplotlib plot
"""

import base64
import csv
import json
import io
from datetime import datetime
from pathlib import Path

import cv2  # noqa: F401 (used via dashboard_rectify)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from flask import jsonify, render_template, request, send_from_directory

from config import (
    SURVEY_POS_X,
    SURVEY_POS_Y,
)
from dashboard_state import state
from dashboard_camera import ensure_cameras
from dashboard_images import b64_img
from dashboard_rectify import load_rectify_maps, rectify_pair


# ─────────────────────────────────────────────────────────────────────────────
# Rectified → raw coordinate mapping
# ─────────────────────────────────────────────────────────────────────────────

def _rect_to_raw(rect_x, rect_y, side, maps):
    """Map one rectified image pixel back to the raw camera pixel.

    The remap maps used by cv2.remap satisfy:
        dst[ry, rx] = src[ map_y[ry, rx], map_x[ry, rx] ]

    So map_x[ry, rx] / map_y[ry, rx] directly give the raw (x, y) that
    corresponds to rectified pixel (rx, ry).
    """
    mx = maps["left_map_x"]  if side == "left" else maps["right_map_x"]
    my = maps["left_map_y"]  if side == "left" else maps["right_map_y"]
    h, w = mx.shape[:2]
    ry = int(np.clip(round(rect_y), 0, h - 1))
    rx = int(np.clip(round(rect_x), 0, w - 1))

    if mx.ndim == 3:
        cache_key = f"{side}_float_maps"
        if cache_key not in maps:
            maps[cache_key] = cv2.convertMaps(mx, my, cv2.CV_32FC1)
        mx_f, my_f = maps[cache_key]
        return float(mx_f[ry, rx]), float(my_f[ry, rx])

    return float(np.asarray(mx[ry, rx])), float(np.asarray(my[ry, rx]))


# ─────────────────────────────────────────────────────────────────────────────
# 3-D matplotlib plot
# ─────────────────────────────────────────────────────────────────────────────

_DARK_BG    = "#111111"
_DARK_PANE  = "#1c1c1c"
_CACHE_DIR    = Path(__file__).resolve().parents[1] / "cache"
_SURVEY_CACHE_DIR = _CACHE_DIR / "survey"
_EXPORT_DIR = _CACHE_DIR / "triangulate_exports"


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _export_name(prefix, suffix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{suffix}"


def _encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Could not encode PNG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _point_from_det(det):
    if isinstance(det, dict):
        return tuple(map(float, det.get("point", det.get("left_px", det.get("right_px")))))
    return tuple(map(float, det))


def _match_point(match, side, frame_mode=None):
    if frame_mode == "rectified":
        key = f"{side}_px_rect"
        if key in match:
            return tuple(map(float, match[key]))
    return tuple(map(float, match[f"{side}_px"]))


def _match_box(match, side, frame_mode=None):
    rect_key = f"{side}_box_rect"
    raw_key = f"{side}_box"
    if frame_mode == "rectified" and rect_key in match:
        return tuple(map(float, match[rect_key]))
    if raw_key in match:
        return tuple(map(float, match[raw_key]))
    return None


def _display_xy_from_solved(solved):
    xy = solved["target_xy_mm"]
    return float(xy[0]), float(xy[1])


def _plan_targets_nearest(solved_targets, start_xy):
    remaining = list(solved_targets)
    planned = []
    cur_x, cur_y = map(float, start_xy)
    while remaining:
        nxt = min(
            remaining,
            key=lambda s: float(np.hypot(_display_xy_from_solved(s)[0] - cur_x,
                                         _display_xy_from_solved(s)[1] - cur_y)),
        )
        planned.append(nxt)
        remaining.remove(nxt)
        cur_x, cur_y = _display_xy_from_solved(nxt)
    return planned


def _planned_index_by_identity(planned_targets):
    return {id(target): i + 1 for i, target in enumerate(planned_targets)}


def _draw_label(img, text, org, fg=(255, 255, 255), bg=(20, 20, 20), scale=0.48):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(org[0]), int(org[1])
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    pad = 4
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, fg, 1, cv2.LINE_AA)


def _draw_solved_overlay(frame, solved_targets, planned_targets, side, frame_mode=None):
    img = frame.copy()
    order = _planned_index_by_identity(planned_targets)
    for solved in solved_targets:
        match = solved["source_target"]
        x, y = _match_point(match, side, frame_mode=frame_mode)
        xi, yi = int(round(x)), int(round(y))
        box = _match_box(match, side, frame_mode=frame_mode)
        color = (0, 60, 255) if side == "left" else (255, 70, 70)
        if box is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(img, (xi, yi), 6, (255, 0, 200), -1, cv2.LINE_AA)
        cv2.circle(img, (xi, yi), 11, color, 2, cv2.LINE_AA)
        _draw_label(img, f"P{order.get(id(solved), '?')}", (xi + 12, yi - 8), fg=(255, 255, 255))
    return img


def _make_stitched_overlay(left_frame, right_frame, solved_targets, planned_targets, frame_mode=None):
    left = _draw_solved_overlay(left_frame, solved_targets, planned_targets, "left", frame_mode=frame_mode)
    right = _draw_solved_overlay(right_frame, solved_targets, planned_targets, "right", frame_mode=frame_mode)
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
    stitched = np.hstack([left, right])
    split_x = left.shape[1]
    cv2.line(stitched, (split_x, 0), (split_x, stitched.shape[0]), (230, 230, 230), 2)
    _draw_label(stitched, "LEFT", (18, 32), fg=(0, 229, 255), scale=0.65)
    _draw_label(stitched, "RIGHT", (split_x + 18, 32), fg=(255, 120, 120), scale=0.65)
    return stitched


def _make_photo_path_overlay(left_frame, planned_targets, frame_mode=None):
    img = left_frame.copy()
    pts = []
    for solved in planned_targets:
        x, y = _match_point(solved["source_target"], "left", frame_mode=frame_mode)
        pts.append((int(round(x)), int(round(y))))
    if len(pts) >= 2:
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.arrowedLine(img, a, b, (255, 210, 0), 3, cv2.LINE_AA, tipLength=0.04)
    for i, (x, y) in enumerate(pts, start=1):
        cv2.circle(img, (x, y), 7, (255, 0, 200), -1, cv2.LINE_AA)
        _draw_label(img, str(i), (x + 10, y - 10), fg=(255, 255, 255), bg=(40, 40, 40), scale=0.55)
    return img


def _orientation_sanity(solved_targets, frame_mode=None):
    if len(solved_targets) < 2:
        return "Need at least two matched points for orientation sanity check."
    a, b = solved_targets[0], solved_targets[1]
    ax_px, ay_px = _match_point(a["source_target"], "left", frame_mode=frame_mode)
    bx_px, by_px = _match_point(b["source_target"], "left", frame_mode=frame_mode)
    ax_mm, ay_mm = _display_xy_from_solved(a)
    bx_mm, by_mm = _display_xy_from_solved(b)
    image_relation = []
    plot_relation = []
    if ax_px > bx_px:
        image_relation.append("right")
        plot_relation.append("right" if ax_mm > bx_mm else "left")
    elif ax_px < bx_px:
        image_relation.append("left")
        plot_relation.append("left" if ax_mm < bx_mm else "right")
    if ay_px < by_px:
        image_relation.append("top")
        plot_relation.append("top" if ay_mm > by_mm else "bottom")
    elif ay_px > by_px:
        image_relation.append("bottom")
        plot_relation.append("bottom" if ay_mm < by_mm else "top")
    image_text = "-".join(image_relation) or "same image position"
    plot_text = "-".join(plot_relation) or "same plot position"
    ok = image_text == plot_text
    return (
        f"Orientation sanity P1 vs P2: image says P1 is {image_text} of P2; "
        f"top-down plot says P1 is {plot_text} of P2 ({'PASS' if ok else 'CHECK'})."
    )


def _load_survey_cache():
    meta_path = _SURVEY_CACHE_DIR / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            "No survey cache found. Run Survey → 'Scan + Match + Cache' first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    left_frame = cv2.imread(str(_SURVEY_CACHE_DIR / "left.jpg"))
    right_frame = cv2.imread(str(_SURVEY_CACHE_DIR / "right.jpg"))
    if left_frame is None or right_frame is None:
        raise RuntimeError("Survey cache images missing. Re-run the survey.")
    return meta, left_frame, right_frame


def _make_3d_plot(solved_targets, mover, planned_targets=None, left_frame=None, frame_mode=None, start_xy=None, elev=90, azim=-90):
    """Render a dark-themed matplotlib 3-D scatter of triangulated points + cameras.

    The top-down view uses X right and Y down so relative positions match the
    photo: a point that is top-right in the left image appears top-right in the
    plot, too.
    """
    # ── collect raw 3-D positions (metres → mm) ───────────────────────────
    points_3d_by_id = {}
    for s in solved_targets:
        X_rect, _, _, _, _ = mover._solve_geometry(s["source_target"])
        points_3d_by_id[id(s)] = X_rect * 1000.0          # metres → mm
    points_3d = [points_3d_by_id[id(s)] for s in solved_targets]
    planned_targets = planned_targets or list(solved_targets)

    # ── camera positions in the rectified frame (mm) ──────────────────────
    cam1 = np.array([0.0, 0.0, 0.0])               # left camera at origin
    cam2 = mover.T_rect * 1000.0                   # right camera at baseline

    CAMERA_RADIUS_MM = 7.5                          # 15 mm diameter / 2
    theta = np.linspace(0.0, 2.0 * np.pi, 64)

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 8), facecolor=_DARK_BG)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(_DARK_PANE)

    def _disp(arr_mm):
        """Return (xs, ys, zs) for display.

        Camera frame: Y increases downward in image, Z is depth toward ground.
        Negate Y so the plot matches image orientation (bottom of image = low plot Y).
        Negate Z so Z=0 is at camera and positive values extend away from camera.
        """
        a = np.asarray(arr_mm, dtype=float).reshape(-1, 3)
        return a[:, 0].tolist(), (-a[:, 1]).tolist(), (-a[:, 2]).tolist()

    # ── triangulated points ───────────────────────────────────────────────
    if points_3d:
        xs, ys, zs = _disp(points_3d)
        ax.scatter(xs, ys, zs,
                   c="#00e5ff", s=70, marker="o", label="Points",
                   depthshade=True, zorder=5)
        order = _planned_index_by_identity(planned_targets)
        for solved, x, y, z in zip(solved_targets, xs, ys, zs):
            ax.text(x, y, z, f"  P{order.get(id(solved), '?')}", color="#ffffff", fontsize=8)

        path_pts = [points_3d_by_id[id(s)] for s in planned_targets if id(s) in points_3d_by_id]
        if len(path_pts) >= 2:
            px, py, pz = _disp(path_pts)
            ax.plot(px, py, pz, c="#ffd500", lw=2.7, label="TSP path", zorder=6)
            for x0, y0, z0, x1, y1, z1 in zip(px[:-1], py[:-1], pz[:-1], px[1:], py[1:], pz[1:]):
                ax.quiver(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0,
                          color="#ffd500", arrow_length_ratio=0.12, linewidth=1.4)

        if left_frame is not None:
            photo = _make_photo_path_overlay(left_frame, planned_targets, frame_mode=frame_mode)
            photo_rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            inset = fig.add_axes([0.64, 0.08, 0.31, 0.31], facecolor="#111111")
            inset.imshow(photo_rgb)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title("Left photo + planned path", color="#eeeeee", fontsize=9, pad=5)
            for spine in inset.spines.values():
                spine.set_edgecolor("#666666")

    # ── camera 1 disc (left) ──────────────────────────────────────────────
    cx1 = cam1[0] + CAMERA_RADIUS_MM * np.cos(theta)
    cy1 = cam1[1] + CAMERA_RADIUS_MM * np.sin(theta)
    cz1 = np.full(64, -cam1[2])
    ax.plot(cx1, cy1, cz1, c="#ff6600", lw=2, label="Cam L")
    ax.scatter([cam1[0]], [cam1[1]], [-cam1[2]],
               c="#ff6600", s=90, marker="^", zorder=10)

    # ── camera 2 disc (right) ─────────────────────────────────────────────
    cx2 = cam2[0] + CAMERA_RADIUS_MM * np.cos(theta)
    cy2 = cam2[1] + CAMERA_RADIUS_MM * np.sin(theta)
    cz2 = np.full(64, -cam2[2])
    ax.plot(cx2, cy2, cz2, c="#ff0080", lw=2, label="Cam R")
    ax.scatter([cam2[0]], [cam2[1]], [-cam2[2]],
               c="#ff0080", s=90, marker="^", zorder=10)

    # ── baseline connector ────────────────────────────────────────────────
    ax.plot([cam1[0], cam2[0]], [cam1[1], cam2[1]], [-cam1[2], -cam2[2]],
            "w--", lw=1.2, alpha=0.55, label="Baseline")

    if points_3d:
        top_z = min(-p[2] for p in points_3d) - 12.0
        for p in points_3d:
            ax.plot([cam1[0], p[0]], [-cam1[1], -p[1]], [-cam1[2], -p[2]],
                    c="#00e5ff", lw=0.8, alpha=0.16)
            ax.plot([cam2[0], p[0]], [-cam2[1], -p[1]], [-cam2[2], -p[2]],
                    c="#ff0080", lw=0.8, alpha=0.16)

        xs, ys, zs = _disp(points_3d)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad_x = max(20.0, 0.08 * max(1.0, x_max - x_min))
        pad_y = max(20.0, 0.08 * max(1.0, y_max - y_min))
        z_plane = max(zs) + 8.0
        plane = [
            (x_min - pad_x, y_min - pad_y, z_plane),
            (x_max + pad_x, y_min - pad_y, z_plane),
            (x_max + pad_x, y_max + pad_y, z_plane),
            (x_min - pad_x, y_max + pad_y, z_plane),
        ]
        ax.add_collection3d(Poly3DCollection(
            [plane], facecolors=(0.05, 0.05, 0.05, 0.28),
            edgecolors=(0.7, 0.7, 0.7, 0.35), linewidths=0.8,
        ))

    # ── axes styling ──────────────────────────────────────────────────────
    for spine in (ax.xaxis, ax.yaxis, ax.zaxis):
        spine.pane.fill = False
        spine.pane.set_edgecolor("#333333")

    ax.set_xlabel("X (mm)", color="#cccccc", fontsize=10)
    ax.set_ylabel("Y (mm)", color="#cccccc", fontsize=10)
    ax.set_zlabel("Height from camera (mm)", color="#cccccc", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    ax.set_title(
        f"Triangulated {len(points_3d)} point(s) with planned laser path",
        color="#eeeeee", fontsize=12, pad=10,
    )
    if points_3d:
        all_x = [cam1[0], cam2[0]] + [p[0] for p in points_3d]
        all_y = [cam1[1], cam2[1]] + [p[1] for p in points_3d]
        all_z = [-cam1[2], -cam2[2]] + [-p[2] for p in points_3d]
        pad = 20.0
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.set_zlim(min(all_z) - pad, max(all_z) + pad)
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect((max(all_x) - min(all_x) + 2 * pad,
                               max(all_y) - min(all_y) + 2 * pad,
                               max(all_z) - min(all_z) + 2 * pad))
        except Exception:
            pass
    legend = ax.legend(loc="upper right", fontsize=9,
                       facecolor="#222222", edgecolor="#444444")
    if legend is not None:
        for text in legend.get_texts():
            text.set_color("#eeeeee")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=_DARK_BG, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _write_plot_png(plot_b64, path):
    path.write_bytes(base64.b64decode(plot_b64.encode("ascii")))


def _write_graph_csv(path, solved_targets, planned_targets, ref_x, ref_y, frame_mode=None):
    order = _planned_index_by_identity(planned_targets)
    fields = [
        "point_id", "planned_order",
        "left_px_x", "left_px_y", "right_px_x", "right_px_y",
        "left_rect_x", "left_rect_y", "right_rect_x", "right_rect_y",
        "target_x_mm", "target_y_mm",
        "raw_target_x_mm", "raw_target_y_mm",
        "pixel_corr_x_mm", "pixel_corr_y_mm",
        "score", "box_iou", "y_diff_px", "disp_px",
        "ref_x_mm", "ref_y_mm", "frame_mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, solved in enumerate(solved_targets, start=1):
            src = solved["source_target"]
            lx, ly = src["left_px"]
            rx, ry = src["right_px"]
            lrx, lry = src.get("left_px_rect", (None, None))
            rrx, rry = src.get("right_px_rect", (None, None))
            tx, ty = solved["target_xy_mm"]
            raw_tx, raw_ty = solved.get("raw_triangulated_xy_mm", solved["target_xy_mm"])
            corr_x, corr_y = solved.get("pixel_correction_applied_mm", (0.0, 0.0))
            writer.writerow({
                "point_id": idx,
                "planned_order": order.get(id(solved), idx),
                "left_px_x": lx, "left_px_y": ly,
                "right_px_x": rx, "right_px_y": ry,
                "left_rect_x": lrx, "left_rect_y": lry,
                "right_rect_x": rrx, "right_rect_y": rry,
                "target_x_mm": tx, "target_y_mm": ty,
                "raw_target_x_mm": raw_tx, "raw_target_y_mm": raw_ty,
                "pixel_corr_x_mm": corr_x, "pixel_corr_y_mm": corr_y,
                "score": src.get("score"),
                "box_iou": src.get("box_iou"),
                "y_diff_px": src.get("y_diff_px"),
                "disp_px": src.get("disp_px"),
                "ref_x_mm": ref_x,
                "ref_y_mm": ref_y,
                "frame_mode": frame_mode or "",
            })


def _export_triangulation_outputs(
    solved_targets, mover, ref_x, ref_y,
    left_frame=None, right_frame=None, frame_mode=None, prefix="triangulate",
):
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    planned_targets = _plan_targets_nearest(solved_targets, start_xy=(ref_x, ref_y))
    plot_b64 = _make_3d_plot(
        solved_targets, mover, planned_targets=planned_targets,
        left_frame=left_frame, frame_mode=frame_mode, start_xy=(ref_x, ref_y),
    )

    plot_path = _EXPORT_DIR / _export_name(f"{prefix}_plot3d", "png")
    csv_path = _EXPORT_DIR / _export_name(f"{prefix}_graph", "csv")
    manifest_path = _EXPORT_DIR / _export_name(f"{prefix}_manifest", "json")
    _write_plot_png(plot_b64, plot_path)
    _write_graph_csv(csv_path, solved_targets, planned_targets, ref_x, ref_y, frame_mode=frame_mode)

    stitched_path = None
    stitched_b64 = None
    if left_frame is not None and right_frame is not None:
        stitched = _make_stitched_overlay(
            left_frame, right_frame, solved_targets, planned_targets, frame_mode=frame_mode,
        )
        stitched_path = _EXPORT_DIR / _export_name(f"{prefix}_stitched_overlay", "png")
        cv2.imwrite(str(stitched_path), stitched)
        stitched_b64 = _encode_png(stitched)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ref_xy_mm": [float(ref_x), float(ref_y)],
        "frame_mode": frame_mode,
        "orientation_sanity": _orientation_sanity(solved_targets, frame_mode=frame_mode),
        "solved_targets": _json_safe(solved_targets),
        "planned_order": [
            {
                "planned_order": i,
                "target_xy_mm": _json_safe(t["target_xy_mm"]),
                "left_px": _json_safe(t["source_target"]["left_px"]),
                "right_px": _json_safe(t["source_target"]["right_px"]),
            }
            for i, t in enumerate(planned_targets, start=1)
        ],
        "files": {},
    }
    files = {
        "plot_png": plot_path,
        "graph_csv": csv_path,
        "manifest_json": manifest_path,
    }
    if stitched_path is not None:
        files["stitched_png"] = stitched_path
    manifest["files"] = {k: str(v) for k, v in files.items()}
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    def _link(path):
        return f"/api/triangulate/export/{path.name}"

    return {
        "plot_b64": plot_b64,
        "stitched_b64": stitched_b64,
        "planned_targets": planned_targets,
        "orientation_sanity": manifest["orientation_sanity"],
        "files": {
            k: {"path": str(v), "url": _link(v)}
            for k, v in files.items()
        },
    }


def _solve_matches(left_raw_pts, right_raw_pts, ref_x, ref_y):
    from vision.matching import match_points
    from pipeline.steps.match_plan import normalize_matches
    from control.coarse_move import TriangulationCoarseMover

    left_dets = [tuple(map(float, p)) for p in left_raw_pts]
    right_dets = [tuple(map(float, p)) for p in right_raw_pts]

    matched, unmatched_l, unmatched_r = match_points(
        left_dets, right_dets, verbose=False,
    )
    matched = normalize_matches(matched)
    if not matched:
        return matched, unmatched_l, unmatched_r, None, []

    mover = TriangulationCoarseMover()
    mover.fit_epipolar(matched)
    solved = mover.solve_all_from_pose(matched, ref_x, ref_y)
    return matched, unmatched_l, unmatched_r, mover, solved


def _load_disk_cached_plan_scan():
    plan_path = _CACHE_DIR / "latest_plan.json"
    left_path = _CACHE_DIR / "fine_align_debug" / "latest_full_left.jpg"
    right_path = _CACHE_DIR / "fine_align_debug" / "latest_full_right.jpg"

    if not plan_path.exists():
        raise RuntimeError(
            "No in-memory scan and no dev_tools/cache/latest_plan.json fallback found."
        )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    solved = [
        target["raw"]
        for target in plan.get("targets", [])
        if isinstance(target, dict) and isinstance(target.get("raw"), dict)
    ]
    if not solved:
        raise RuntimeError(f"Cached plan has no solved targets: {plan_path}")

    left_frame = cv2.imread(str(left_path)) if left_path.exists() else None
    right_frame = cv2.imread(str(right_path)) if right_path.exists() else None

    return {
        "timestamp": plan.get("created_at"),
        "frame_mode": plan.get("frame_mode", "raw"),
        "left_frame": left_frame,
        "right_frame": right_frame,
        "disk_plan_path": str(plan_path),
        "disk_left_path": str(left_path) if left_frame is not None else None,
        "disk_right_path": str(right_path) if right_frame is not None else None,
        "survey_ref_xy": plan.get("survey_ref_xy"),
        "solved_targets": solved,
    }


def _solve_cached_scan(ref_x, ref_y):
    from vision.matching import match_points
    from pipeline.steps.match_plan import normalize_matches
    from control.coarse_move import TriangulationCoarseMover

    if state.last_scan is None:
        scan = _load_disk_cached_plan_scan()
        solved = scan["solved_targets"]
        matched = normalize_matches([s["source_target"] for s in solved])
        mover = TriangulationCoarseMover()
        if matched:
            mover.fit_epipolar(matched)
        return matched, [], [], mover, solved, scan

    left_dets = list(state.last_scan["left_detections"])
    right_dets = list(state.last_scan["right_detections"])

    if state.last_scan.get("frame_mode") == "rectified":
        def _tag_rectified(dets):
            out = []
            for d in dets:
                item = dict(d)
                if "point" in item and "point_rectified" not in item:
                    item["point_rectified"] = tuple(item["point"])
                if "box" in item and "box_rectified" not in item:
                    item["box_rectified"] = tuple(item["box"])
                out.append(item)
            return out
        left_dets = _tag_rectified(left_dets)
        right_dets = _tag_rectified(right_dets)

    matched, unmatched_l, unmatched_r = match_points(left_dets, right_dets, verbose=False)
    matched = normalize_matches(matched)
    if not matched:
        return matched, unmatched_l, unmatched_r, None, []

    mover = TriangulationCoarseMover()
    mover.fit_epipolar(matched)
    solved = mover.solve_all_from_pose(matched, ref_x, ref_y)
    return matched, unmatched_l, unmatched_r, mover, solved, state.last_scan


def _solved_json(solved):
    return [
        {
            "target_xy_mm":           list(s["target_xy_mm"]),
            "raw_triangulated_xy_mm": list(
                s.get("raw_triangulated_xy_mm", s["target_xy_mm"])
            ),
            "pixel_correction_applied_mm": list(
                s.get("pixel_correction_applied_mm", (0.0, 0.0))
            ),
            "left_px":  list(s["source_target"]["left_px"]),
            "right_px": list(s["source_target"]["right_px"]),
            "score": float(s["source_target"].get("score", 0.0)),
        }
        for s in solved
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Route registration
# ─────────────────────────────────────────────────────────────────────────────

def register_triangulate_routes(app):

    # ── page ─────────────────────────────────────────────────────────────

    @app.route("/triangulate")
    def triangulate_page():
        return render_template("triangulate.html")

    # ── capture rectified pair ────────────────────────────────────────────

    @app.route("/api/triangulate/capture", methods=["POST"])
    def api_triangulate_capture():
        try:
            with state.camera_lock:
                cam = ensure_cameras()
                fl, fr = cam.read_pair()

            if fl is None or fr is None:
                return jsonify({"ok": False, "error": "Camera read returned None."}), 500

            l_rect, r_rect, _ = rectify_pair(fl, fr)
            fh, fw = l_rect.shape[:2]
            state.last_triangulate_capture = {
                "left_frame": l_rect,
                "right_frame": r_rect,
                "frame_mode": "rectified",
                "frame_w": fw,
                "frame_h": fh,
            }

            return jsonify({
                "ok":      True,
                "left":    b64_img(l_rect),
                "right":   b64_img(r_rect),
                "frame_w": fw,
                "frame_h": fh,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    # ── rectified pixel → raw camera pixel ───────────────────────────────

    @app.route("/api/triangulate/rect_to_raw", methods=["POST"])
    def api_triangulate_rect_to_raw():
        try:
            data   = request.get_json(force=True)
            side   = str(data.get("side", "left"))
            rect_x = float(data["rect_x"])
            rect_y = float(data["rect_y"])

            if side not in ("left", "right"):
                return jsonify({"ok": False, "error": f"Invalid side: {side!r}"}), 400

            maps           = load_rectify_maps()
            raw_x, raw_y   = _rect_to_raw(rect_x, rect_y, side, maps)

            return jsonify({"ok": True, "raw_x": raw_x, "raw_y": raw_y})
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    # ── match + triangulate + 3-D plot ────────────────────────────────────

    @app.route("/api/triangulate/run", methods=["POST"])
    def api_triangulate_run():
        try:
            data          = request.get_json(force=True)
            left_raw_pts  = data.get("left_raw_pts",  [])   # [[x, y], ...]
            right_raw_pts = data.get("right_raw_pts", [])
            ref_x         = float(data.get("ref_x", SURVEY_POS_X))
            ref_y         = float(data.get("ref_y", SURVEY_POS_Y))

            if not left_raw_pts or not right_raw_pts:
                return jsonify({
                    "ok":    False,
                    "error": "Need at least 1 left point and 1 right point.",
                }), 400

            matched, unmatched_l, unmatched_r, mover, solved = _solve_matches(
                left_raw_pts, right_raw_pts, ref_x, ref_y,
            )

            if not matched:
                return jsonify({
                    "ok":             True,
                    "matched_count":  0,
                    "unmatched_left": len(unmatched_l),
                    "unmatched_right": len(unmatched_r),
                    "solved":         [],
                    "plot_b64":       None,
                    "message": (
                        "No stereo matches found. "
                        "Check disparity / epipolar constraints. "
                        "Points in the right frame should be slightly to the "
                        "left of their left-frame counterparts with nearly the "
                        "same Y coordinate."
                    ),
                })

            capture = getattr(state, "last_triangulate_capture", {}) or {}
            exports = _export_triangulation_outputs(
                solved, mover, ref_x, ref_y,
                left_frame=capture.get("left_frame"),
                right_frame=capture.get("right_frame"),
                frame_mode=capture.get("frame_mode", "raw"),
                prefix="manual",
            )

            return jsonify({
                "ok":              True,
                "matched_count":   len(matched),
                "unmatched_left":  len(unmatched_l),
                "unmatched_right": len(unmatched_r),
                "solved":          _solved_json(solved),
                "plot_b64":        exports["plot_b64"],
                "stitched_b64":    exports["stitched_b64"],
                "orientation_sanity": exports["orientation_sanity"],
                "files":           exports["files"],
            })

        except Exception as e:
            import traceback
            return jsonify({
                "ok":        False,
                "error":     repr(e),
                "traceback": traceback.format_exc(),
            }), 500

    @app.route("/api/triangulate/run_cached_scan", methods=["POST"])
    def api_triangulate_run_cached_scan():
        try:
            data = request.get_json(force=True) if request.data else {}
            ref_x = float(data.get("ref_x", SURVEY_POS_X))
            ref_y = float(data.get("ref_y", SURVEY_POS_Y))

            matched, unmatched_l, unmatched_r, mover, solved, scan = _solve_cached_scan(ref_x, ref_y)
            if not matched:
                return jsonify({
                    "ok": True,
                    "matched_count": 0,
                    "unmatched_left": len(unmatched_l),
                    "unmatched_right": len(unmatched_r),
                    "solved": [],
                    "plot_b64": None,
                    "stitched_b64": None,
                    "message": "No stereo matches found in cached scan.",
                })

            exports = _export_triangulation_outputs(
                solved, mover, ref_x, ref_y,
                left_frame=scan.get("left_frame"),
                right_frame=scan.get("right_frame"),
                frame_mode=scan.get("frame_mode", "raw"),
                prefix="cached_scan",
            )

            return jsonify({
                "ok": True,
                "matched_count": len(matched),
                "unmatched_left": len(unmatched_l),
                "unmatched_right": len(unmatched_r),
                "solved": _solved_json(solved),
                "plot_b64": exports["plot_b64"],
                "stitched_b64": exports["stitched_b64"],
                "orientation_sanity": exports["orientation_sanity"],
                "files": exports["files"],
                "scan_timestamp": scan.get("timestamp"),
                "frame_mode": scan.get("frame_mode"),
            })

        except Exception as e:
            import traceback
            return jsonify({
                "ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }), 500

    # ── auto-triangulate from survey cache ───────────────────────────────

    @app.route("/api/triangulate/from_survey", methods=["POST"])
    def api_triangulate_from_survey():
        try:
            data = request.get_json(force=True) if request.data else {}
            ref_x = float(data.get("ref_x", SURVEY_POS_X))
            ref_y = float(data.get("ref_y", SURVEY_POS_Y))

            meta, left_frame, right_frame = _load_survey_cache()
            matches = meta.get("matches", [])
            if not matches:
                return jsonify({"ok": False, "error": "Survey cache has no matches. Re-run the survey."}), 400

            frame_mode = meta.get("frame_mode", "raw")

            from control.coarse_move import TriangulationCoarseMover
            mover = TriangulationCoarseMover()
            mover.fit_epipolar(matches)
            solved = mover.solve_all_from_pose(matches, ref_x, ref_y)

            if not solved:
                return jsonify({
                    "ok": True,
                    "matched_count": len(matches),
                    "solved_count": 0,
                    "solved": [],
                    "plot_b64": None,
                    "stitched_b64": None,
                    "message": "Triangulation produced no solved targets.",
                })

            planned = _plan_targets_nearest(solved, (ref_x, ref_y))

            state.last_triangulation = {
                "mover": mover,
                "solved_targets": solved,
                "planned_targets": planned,
                "ref_x": ref_x,
                "ref_y": ref_y,
                "left_frame": left_frame,
                "right_frame": right_frame,
                "frame_mode": frame_mode,
                "meta_timestamp": meta.get("timestamp"),
            }

            plot_b64 = _make_3d_plot(
                solved, mover, planned_targets=planned,
                left_frame=left_frame, frame_mode=frame_mode,
            )
            stitched = _make_stitched_overlay(
                left_frame, right_frame, solved, planned, frame_mode=frame_mode,
            )

            return jsonify({
                "ok": True,
                "matched_count": len(matches),
                "solved_count": len(solved),
                "solved": _solved_json(solved),
                "plot_b64": plot_b64,
                "stitched_b64": _encode_png(stitched),
                "orientation_sanity": _orientation_sanity(solved, frame_mode=frame_mode),
                "scan_timestamp": meta.get("timestamp"),
                "frame_mode": frame_mode,
            })
        except Exception as e:
            import traceback
            return jsonify({"ok": False, "error": repr(e), "traceback": traceback.format_exc()}), 500

    # ── re-render plot at user-chosen angle ───────────────────────────────

    @app.route("/api/triangulate/replot", methods=["POST"])
    def api_triangulate_replot():
        try:
            data = request.get_json(force=True)
            elev = float(data.get("elev", 90))
            azim = float(data.get("azim", -90))

            tri = state.last_triangulation
            if tri is None:
                return jsonify({"ok": False, "error": "No triangulation in memory. Run triangulation first."}), 400

            plot_b64 = _make_3d_plot(
                tri["solved_targets"], tri["mover"],
                planned_targets=tri["planned_targets"],
                left_frame=tri["left_frame"],
                frame_mode=tri["frame_mode"],
                elev=elev, azim=azim,
            )
            return jsonify({"ok": True, "plot_b64": plot_b64})
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    # ── save at user-chosen angle ─────────────────────────────────────────

    @app.route("/api/triangulate/save", methods=["POST"])
    def api_triangulate_save():
        try:
            data = request.get_json(force=True) if request.data else {}
            elev = float(data.get("elev", 90))
            azim = float(data.get("azim", -90))

            tri = state.last_triangulation
            if tri is None:
                return jsonify({"ok": False, "error": "No triangulation in memory. Run triangulation first."}), 400

            mover       = tri["mover"]
            solved      = tri["solved_targets"]
            planned     = tri["planned_targets"]
            ref_x       = tri["ref_x"]
            ref_y       = tri["ref_y"]
            left_frame  = tri["left_frame"]
            right_frame = tri["right_frame"]
            frame_mode  = tri["frame_mode"]

            _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            plot_b64 = _make_3d_plot(
                solved, mover, planned_targets=planned,
                left_frame=left_frame, frame_mode=frame_mode,
                elev=elev, azim=azim,
            )
            plot_path = _EXPORT_DIR / f"survey_{ts}_plot3d.png"
            _write_plot_png(plot_b64, plot_path)

            stitched = _make_stitched_overlay(
                left_frame, right_frame, solved, planned, frame_mode=frame_mode,
            )
            stitched_path = _EXPORT_DIR / f"survey_{ts}_stitched_overlay.png"
            cv2.imwrite(str(stitched_path), stitched)

            csv_path = _EXPORT_DIR / f"survey_{ts}_graph.csv"
            _write_graph_csv(csv_path, solved, planned, ref_x, ref_y, frame_mode=frame_mode)

            files = {
                "plot_png":    {"path": str(plot_path),    "url": f"/api/triangulate/export/{plot_path.name}"},
                "stitched_png": {"path": str(stitched_path), "url": f"/api/triangulate/export/{stitched_path.name}"},
                "graph_csv":   {"path": str(csv_path),     "url": f"/api/triangulate/export/{csv_path.name}"},
            }
            return jsonify({
                "ok": True,
                "plot_b64": plot_b64,
                "stitched_b64": _encode_png(stitched),
                "files": files,
            })
        except Exception as e:
            import traceback
            return jsonify({"ok": False, "error": repr(e), "traceback": traceback.format_exc()}), 500

    @app.route("/api/triangulate/export/<path:filename>", methods=["GET"])
    def api_triangulate_export_file(filename):
        return send_from_directory(_EXPORT_DIR, filename, as_attachment=False)
