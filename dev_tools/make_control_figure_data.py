#!/usr/bin/env python3
"""Generate standalone control-figure detection/triangulation data.

The input survey photos are assumed to already be rectified.  The script does
not remap the images.  It detects targets on the rectified photos, matches them
with the same stereo matcher used by the dashboard/runtime, maps rectified
target pixels back to raw camera pixels, and then calls the shared
TriangulationCoarseMover geometry so the gantry/workspace coordinates match the
dashboard triangulation page.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REMOTE_DASHBOARD_DIR = ROOT / "dev_tools" / "remote"
if str(REMOTE_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_DASHBOARD_DIR))

# Match the dashboard/bringup import path on Jetson systems where torchvision's
# NMS registration can fail before Ultralytics is imported.
_NMS_PATCH_PATH = ROOT / "bringup" / "_nms_patch.py"
if _NMS_PATCH_PATH.exists():
    _spec = importlib.util.spec_from_file_location("_nms_patch", _NMS_PATCH_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

from config import (  # noqa: E402
    AVOID_CLASSES,
    CALIB_NPZ_PATH,
    DEFAULT_QPOINT_MODEL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    RECT_NPZ_PATH,
    SURVEY_CLUSTER_RADIUS_PX,
    SURVEY_CROP_H,
    SURVEY_CROP_MODE,
    SURVEY_CROP_W,
    SURVEY_LEFT_OFFSET_X,
    SURVEY_LEFT_OFFSET_Y,
    SURVEY_MIN_HITS,
    SURVEY_POINT_MODE,
    SURVEY_RIGHT_OFFSET_X,
    SURVEY_RIGHT_OFFSET_Y,
    SURVEY_YOLO_IMGSZ,
    TARGET_CLASSES,
    CALIBRATION_EXPECTS_UNFLIPPED,
    TRI_SIGN_X,
    TRI_SIGN_Y,
    LASER_OFFSET_X_MM,
    LASER_OFFSET_Y_MM,
    TRI_X_GAIN,
    TRI_Y_GAIN,
)
from config.survey_params import resolve_point_mode  # noqa: E402
from control.coarse_move import TriangulationCoarseMover, _survey_crop_box  # noqa: E402
from planning.target_planner import plan_targets  # noqa: E402
from pipeline.steps.match_plan import normalize_matches  # noqa: E402
from vision.detectors.ai_detector import AIDetector  # noqa: E402
from vision.matching import match_points  # noqa: E402

try:
    from dashboard_rectify import load_rectify_maps as dashboard_load_rectify_maps  # noqa: E402
    from dashboard_triangulate import (  # noqa: E402
        _rect_to_raw as dashboard_rect_to_raw,
        _solve_matches as dashboard_solve_matches,
    )
except Exception as exc:  # pragma: no cover - debug/dashboard dependency fallback
    dashboard_load_rectify_maps = None
    dashboard_rect_to_raw = None
    dashboard_solve_matches = None
    DASHBOARD_IMPORT_ERROR = repr(exc)
else:
    DASHBOARD_IMPORT_ERROR = None


CSV_COLUMNS = [
    "id",
    "path_order",
    "plot_label",
    "class_name",
    "confidence_left",
    "confidence_right",
    "u_left_px",
    "v_left_px",
    "u_right_px",
    "v_right_px",
    "disparity_px",
    "x_mm",
    "y_mm",
    "z_mm",
    "gantry_x_mm",
    "gantry_y_mm",
]

ANNOTATION_BGR = (0, 0, 0)
ANNOTATION_RGB01 = (0.0, 0.0, 0.0)


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


def _find_npz_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    return None


def _as_float_xy_maps(map1, map2):
    m1 = np.asarray(map1)
    m2 = np.asarray(map2)
    if m1.ndim == 3 and m1.shape[2] == 2:
        try:
            mx, my = cv2.convertMaps(m1, m2, cv2.CV_32FC1)
            return np.asarray(mx, dtype=np.float32), np.asarray(my, dtype=np.float32)
        except cv2.error:
            return np.asarray(m1[..., 0], dtype=np.float32), np.asarray(m1[..., 1], dtype=np.float32)
    if m1.ndim == 2 and m2.ndim == 2:
        return np.asarray(m1, dtype=np.float32), np.asarray(m2, dtype=np.float32)
    raise RuntimeError(f"Unsupported rectification map shapes: {m1.shape}, {m2.shape}")


def _load_rectified_to_raw_maps():
    if dashboard_load_rectify_maps is not None:
        return dashboard_load_rectify_maps()

    data = np.load(str(RECT_NPZ_PATH))
    left_x_key = _find_npz_key(data, ["map1L", "left_map_x", "map1_left", "left_map1", "mapLx", "mapxL"])
    left_y_key = _find_npz_key(data, ["map2L", "left_map_y", "map2_left", "left_map2", "mapLy", "mapyL"])
    right_x_key = _find_npz_key(data, ["map1R", "right_map_x", "map1_right", "right_map1", "mapRx", "mapxR"])
    right_y_key = _find_npz_key(data, ["map2R", "right_map_y", "map2_right", "right_map2", "mapRy", "mapyR"])
    if None in (left_x_key, left_y_key, right_x_key, right_y_key):
        raise RuntimeError(
            f"Could not find rectification maps in {RECT_NPZ_PATH}; keys={list(data.keys())}"
        )
    left_map_x, left_map_y = _as_float_xy_maps(data[left_x_key], data[left_y_key])
    right_map_x, right_map_y = _as_float_xy_maps(data[right_x_key], data[right_y_key])
    return {
        "left_map_x": left_map_x,
        "left_map_y": left_map_y,
        "right_map_x": right_map_x,
        "right_map_y": right_map_y,
        "keys": {
            "left_x": left_x_key,
            "left_y": left_y_key,
            "right_x": right_x_key,
            "right_y": right_y_key,
        },
    }


def _rect_to_raw(rect_pt, side, maps):
    if dashboard_rect_to_raw is not None:
        return dashboard_rect_to_raw(rect_pt[0], rect_pt[1], side, maps)

    map_x = maps["left_map_x"] if side == "left" else maps["right_map_x"]
    map_y = maps["left_map_y"] if side == "left" else maps["right_map_y"]
    h, w = map_x.shape[:2]
    rx = int(np.clip(round(float(rect_pt[0])), 0, w - 1))
    ry = int(np.clip(round(float(rect_pt[1])), 0, h - 1))
    return float(map_x[ry, rx]), float(map_y[ry, rx])


def _rect_box_to_raw_aabb(box, side, maps):
    x1, y1, x2, y2 = box
    pts = [
        _rect_to_raw((x1, y1), side, maps),
        _rect_to_raw((x2, y1), side, maps),
        _rect_to_raw((x1, y2), side, maps),
        _rect_to_raw((x2, y2), side, maps),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _translate_detections_to_full(dets, crop):
    x0, y0 = crop["x0"], crop["y0"]
    out = []
    for det in dets:
        px, py = det["point"]
        x1, y1, x2, y2 = det["box"]
        out.append({
            **det,
            "point": (int(round(px + x0)), int(round(py + y0))),
            "box": (
                float(x1 + x0),
                float(y1 + y0),
                float(x2 + x0),
                float(y2 + y0),
            ),
        })
    return out


def _detect_side(detector_core, image, side, point_mode, classes_override=TARGET_CLASSES):
    h, w = image.shape[:2]
    if side == "left":
        crop = _survey_crop_box(
            SURVEY_CROP_MODE, SURVEY_CROP_W, SURVEY_CROP_H,
            SURVEY_LEFT_OFFSET_X, SURVEY_LEFT_OFFSET_Y,
            frame_w=w, frame_h=h, side="left",
        )
    else:
        crop = _survey_crop_box(
            SURVEY_CROP_MODE, SURVEY_CROP_W, SURVEY_CROP_H,
            SURVEY_RIGHT_OFFSET_X, SURVEY_RIGHT_OFFSET_Y,
            frame_w=w, frame_h=h, side="right",
        )

    crop_img = image[crop["y0"]:crop["y1"], crop["x0"]:crop["x1"]]
    stable = detector_core.return_burst_stable(
        [crop_img],
        min_stable_views=1,  # Standalone figure input is a single stereo pair.
        group_radius_px=SURVEY_CLUSTER_RADIUS_PX,
        classes_override=classes_override,
        debug_label=f"[FIGURE {side.upper()}]",
        imgsz=SURVEY_YOLO_IMGSZ,
        heatmap_final=(point_mode != "box_center"),
        point_mode=point_mode,
    )
    return _translate_detections_to_full(stable, crop), crop


def _tag_rectified_detections(detections, side, maps):
    tagged = []
    for det in detections:
        rect_point = tuple(map(float, det["point"]))
        rect_box = tuple(map(float, det["box"]))
        tagged.append({
            **det,
            "point_rectified": rect_point,
            "box_rectified": rect_box,
            "point": _rect_to_raw(rect_point, side, maps),
            "box": _rect_box_to_raw_aabb(rect_box, side, maps),
        })
    return tagged


def _class_name(cls_id, class_names):
    if cls_id is None:
        return ""
    return str(class_names.get(int(cls_id), str(cls_id)))


def _match_class_name(match, class_names):
    left_cls = match.get("left_cls")
    right_cls = match.get("right_cls")
    if left_cls == right_cls:
        return _class_name(left_cls, class_names)
    left_name = _class_name(left_cls, class_names)
    right_name = _class_name(right_cls, class_names)
    return f"{left_name}/{right_name}" if left_name or right_name else ""


def _detection_summary(detections, class_names):
    out = []
    for idx, det in enumerate(detections, start=1):
        out.append({
            "id": idx,
            "class_id": det.get("cls"),
            "class_name": _class_name(det.get("cls"), class_names),
            "confidence": det.get("conf"),
            "point_rectified_px": det.get("point_rectified", det.get("point")),
            "point_raw_px": det.get("point"),
            "box_rectified_px": det.get("box_rectified", det.get("box")),
            "box_raw_px": det.get("box"),
            "views": det.get("views"),
            "point_source": det.get("point_source"),
        })
    return out


def _planned_order_map(planned_targets):
    return {id(target): order for order, target in enumerate(planned_targets, start=1)}


def _plot_label_map(order_by_identity):
    orders = [int(v) for v in order_by_identity.values() if v not in (None, "")]
    zero_based = bool(orders) and min(orders) == 0
    return {
        target_identity: int(path_order) + 1 if zero_based else int(path_order)
        for target_identity, path_order in order_by_identity.items()
    }


def _draw_label(img, text, org, fg=ANNOTATION_BGR, scale=0.46):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), font, scale, fg, 1, cv2.LINE_AA)


def _draw_annotations(image, solved_targets, side, plot_labels):
    out = image.copy()
    for solved in solved_targets:
        match = solved["source_target"]
        point = match.get(f"{side}_px_rect", match[f"{side}_px"])
        box = match.get(f"{side}_box_rect", match.get(f"{side}_box"))
        plot_label = plot_labels.get(id(solved), solved["target_id"])
        x, y = int(round(point[0])), int(round(point[1]))
        if box is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), ANNOTATION_BGR, 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 5, ANNOTATION_BGR, -1, cv2.LINE_AA)
        _draw_label(out, str(plot_label), (x + 10, y - 8), fg=ANNOTATION_BGR, scale=0.44)
    return out


def _write_vector_pdf(path, image, solved_targets, side, plot_labels, width_in=2.0, dpi=300):
    """Write photo + vector annotations to PDF.

    The photographic image is necessarily raster, but boxes, circles, and labels
    are vector PDF objects so they can be edited cleanly in Illustrator.
    """
    h, w = image.shape[:2]
    fig_h = width_in * (h / float(w))
    fig = plt.figure(figsize=(width_in, fig_h), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    for solved in solved_targets:
        match = solved["source_target"]
        plot_label = plot_labels.get(id(solved), solved["target_id"])
        point = match.get(f"{side}_px_rect", match[f"{side}_px"])
        box = match.get(f"{side}_box_rect", match.get(f"{side}_box"))
        x, y = float(point[0]), float(point[1])

        if box is not None:
            x1, y1, x2, y2 = [float(v) for v in box]
            ax.add_patch(patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=ANNOTATION_RGB01,
                linewidth=0.8,
            ))

        ax.add_patch(patches.Circle((x, y), radius=5.0, facecolor=ANNOTATION_RGB01, edgecolor="none"))

        ax.text(
            x + 10,
            y - 8,
            str(plot_label),
            color=ANNOTATION_RGB01,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
        )

    fig.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _write_csv(path, solved_targets, planned_targets, mover, gantry_x, gantry_y, class_names):
    order = _planned_order_map(planned_targets)
    plot_labels = _plot_label_map(order)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for solved in solved_targets:
            src = solved["source_target"]
            lpt = src.get("left_px_rect", src["left_px"])
            rpt = src.get("right_px_rect", src["right_px"])
            tx, ty = solved["target_xy_mm"]
            X_rect, _, _, _, _ = mover._solve_geometry(src)
            writer.writerow({
                "id": solved["target_id"],
                "path_order": order.get(id(solved), ""),
                "plot_label": plot_labels.get(id(solved), ""),
                "class_name": _match_class_name(src, class_names),
                "confidence_left": src.get("left_conf", ""),
                "confidence_right": src.get("right_conf", ""),
                "u_left_px": float(lpt[0]),
                "v_left_px": float(lpt[1]),
                "u_right_px": float(rpt[0]),
                "v_right_px": float(rpt[1]),
                "disparity_px": src.get("disp_px", abs(float(lpt[0]) - float(rpt[0]))),
                "x_mm": float(tx),
                "y_mm": float(ty),
                "z_mm": float(X_rect[2] * 1000.0),
                "gantry_x_mm": float(gantry_x),
                "gantry_y_mm": float(gantry_y),
            })


def _trace_target(solved, mover, path_order, plot_label, gantry_x, gantry_y, class_names):
    src = solved["source_target"]
    X_rect, X_mid, X_laser, dx_mm, dy_mm = mover._solve_geometry(src)
    raw_tx, raw_ty = solved.get("raw_triangulated_xy_mm", solved["target_xy_mm"])
    corr_dx, corr_dy = solved.get("pixel_correction_applied_mm", (0.0, 0.0))
    final_x, final_y = solved["target_xy_mm"]
    return {
        "id": solved["target_id"],
        "path_order": path_order,
        "plot_label": plot_label,
        "class_name": _match_class_name(src, class_names),
        "display_rectified_px": {
            "left": src.get("left_px_rect"),
            "right": src.get("right_px_rect"),
        },
        "triangulation_input_raw_px": {
            "left": src.get("left_px"),
            "right": src.get("right_px"),
        },
        "match_diagnostics_rectified": {
            "score": src.get("score"),
            "box_iou": src.get("box_iou"),
            "y_diff_px": src.get("y_diff_px"),
            "disparity_px": src.get("disp_px"),
        },
        "camera_frame": {
            "X_rect_m": X_rect.tolist(),
            "X_rect_mm": (X_rect * 1000.0).tolist(),
            "X_mid_m": X_mid.tolist(),
            "X_laser_m": X_laser.tolist(),
        },
        "robot_frame_transform": {
            "dx_mm": dx_mm,
            "dy_mm": dy_mm,
            "gantry_ref_x_mm": float(gantry_x),
            "gantry_ref_y_mm": float(gantry_y),
            "raw_target_xy_mm": [raw_tx, raw_ty],
            "pixel_correction_mm": [corr_dx, corr_dy],
            "final_target_xy_mm": [final_x, final_y],
        },
    }


def _make_coordinate_trace(
    left_path,
    right_path,
    left_img,
    right_img,
    left_dets,
    right_dets,
    matched,
    solved_targets,
    planned_targets,
    mover,
    plot_labels,
    gantry_x,
    gantry_y,
    class_names,
    rect_maps,
    args,
    dashboard_compare=None,
):
    order = _planned_order_map(planned_targets)
    return {
        "inputs": {
            "left": str(left_path),
            "right": str(right_path),
            "assumed_input_frame": "rectified stereo frame",
            "image_size_px": [int(left_img.shape[1]), int(left_img.shape[0])],
        },
        "display_orientation": {
            "standalone_rotates_or_flips_for_display": False,
            "annotated_png_pdf_frame": "same pixel grid as input image",
            "note": (
                "The input image is treated as a rectified stereo/display frame. "
                "No extra viewing rotation or flip is applied by this script."
            ),
        },
        "working_pipeline_chain": [
            "StereoCameras.read_pair() captures left/right USB frames",
            "StereoCameras.read_pair() rotates both frames 180 degrees with cv2.ROTATE_180",
            "AIDetector/TriangulationCoarseMover.detect_stable_points detect in that production raw/calibration frame",
            "vision.matching.match_points maps raw detections to rectified coordinates only for correspondence scoring",
            "match_points returns raw/calibration-frame left_px/right_px for triangulation",
            "TriangulationCoarseMover._solve_geometry undistorts/rectifies those raw pixels, triangulates, subtracts stereo midpoint and laser offset, then applies TRI_SIGN_X/TRI_SIGN_Y",
            "TriangulationCoarseMover.solve_target_from_pose adds gantry survey pose and pixel-error correction to produce robot/gantry target_xy_mm",
        ],
        "standalone_chain": [
            "Input files are assumed already rectified; detections and annotations are in rectified image coordinates",
            "Rectified detections are mapped back to raw/calibration pixel coordinates using dashboard_triangulate._rect_to_raw and dashboard_rectify.load_rectify_maps",
            "vision.matching.match_points uses explicit rectified coordinates for matching but preserves raw/calibration left_px/right_px for solving",
            "TriangulationCoarseMover.solve_all_from_pose is used unchanged for robot/gantry coordinates",
            "planning.target_planner.plan_targets is used unchanged for path order",
        ],
        "left_right_ordering": {
            "left_input": str(left_path),
            "right_input": str(right_path),
            "triangulation_left_key": "left_px",
            "triangulation_right_key": "right_px",
        },
        "rectification_maps": {
            "path": str(RECT_NPZ_PATH),
            "keys": rect_maps.get("keys", {}),
            "mapper_reused": "dashboard_triangulate._rect_to_raw" if dashboard_rect_to_raw is not None else "local fallback",
        },
        "coarse_mover_config": {
            "calibration_npz": str(CALIB_NPZ_PATH),
            "tri_sign_x": TRI_SIGN_X,
            "tri_sign_y": TRI_SIGN_Y,
            "tri_x_gain": TRI_X_GAIN,
            "tri_y_gain": TRI_Y_GAIN,
            "laser_offset_x_mm": LASER_OFFSET_X_MM,
            "laser_offset_y_mm": LASER_OFFSET_Y_MM,
            "calibration_expects_unflipped": CALIBRATION_EXPECTS_UNFLIPPED,
            "pixel_error_correction_active": bool(mover.pixel_err_model),
        },
        "counts": {
            "left_detections": len(left_dets),
            "right_detections": len(right_dets),
            "matched_targets": len(matched),
            "triangulated_targets": len(solved_targets),
        },
        "planned_tsp_order": [
            {
                "plot_label": i,
                "id": target.get("target_id"),
                "target_xy_mm": target.get("target_xy_mm"),
            }
            for i, target in enumerate(planned_targets, start=1)
        ],
        "targets": [
            _trace_target(
                solved,
                mover,
                order.get(id(solved)),
                plot_labels.get(id(solved)),
                gantry_x,
                gantry_y,
                class_names,
            )
            for solved in solved_targets
        ],
        "dashboard_compare": dashboard_compare,
        "dashboard_import_error": DASHBOARD_IMPORT_ERROR,
        "cli": {
            "gantry_x_mm": args.gantry_x,
            "gantry_y_mm": args.gantry_y,
            "confidence": args.confidence,
            "debug_trace": args.debug_trace,
            "compare_dashboard": args.compare_dashboard,
        },
    }


def _write_debug_trace_text(path, trace):
    lines = [
        "CONTROL FIGURE COORDINATE TRACE",
        "",
        "Frame chain:",
    ]
    for step in trace["working_pipeline_chain"]:
        lines.append(f"  working:    {step}")
    for step in trace["standalone_chain"]:
        lines.append(f"  standalone: {step}")
    lines.extend([
        "",
        f"Display rotates/flips in standalone: {trace['display_orientation']['standalone_rotates_or_flips_for_display']}",
        f"Left/right ordering: {trace['left_right_ordering']['triangulation_left_key']} / {trace['left_right_ordering']['triangulation_right_key']}",
        f"Rectification mapper: {trace['rectification_maps']['mapper_reused']}",
        f"TRI_SIGN_X/Y: {trace['coarse_mover_config']['tri_sign_x']} / {trace['coarse_mover_config']['tri_sign_y']}",
        f"Pixel error correction active: {trace['coarse_mover_config']['pixel_error_correction_active']}",
        "",
        "Targets:",
    ])
    for t in trace["targets"]:
        r = t["robot_frame_transform"]
        lines.append(
            "  "
            f"id={t['id']} label={t['plot_label']} path_order={t['path_order']} "
            f"rectL={t['display_rectified_px']['left']} rawL={t['triangulation_input_raw_px']['left']} "
            f"X_rect_mm={[round(v, 3) for v in t['camera_frame']['X_rect_mm']]} "
            f"dxy=({r['dx_mm']:.3f},{r['dy_mm']:.3f}) "
            f"final_xy=({r['final_target_xy_mm'][0]:.3f},{r['final_target_xy_mm'][1]:.3f})"
        )
    if trace.get("dashboard_compare") is not None:
        lines.extend(["", "Dashboard compare:", json.dumps(trace["dashboard_compare"], indent=2)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compare_with_dashboard_solver(matched, standalone_solved, gantry_x, gantry_y):
    if dashboard_solve_matches is None:
        return {
            "available": False,
            "error": DASHBOARD_IMPORT_ERROR or "dashboard_triangulate._solve_matches unavailable",
        }

    left_raw_pts = [m["left_px"] for m in matched]
    right_raw_pts = [m["right_px"] for m in matched]
    dash_matched, dash_unmatched_l, dash_unmatched_r, _, dash_solved = dashboard_solve_matches(
        left_raw_pts,
        right_raw_pts,
        gantry_x,
        gantry_y,
    )
    deltas = []
    for idx, (ours, dash) in enumerate(zip(standalone_solved, dash_solved), start=1):
        ours_xy = ours["target_xy_mm"]
        dash_xy = dash["target_xy_mm"]
        deltas.append({
            "index": idx,
            "standalone_xy_mm": [float(ours_xy[0]), float(ours_xy[1])],
            "dashboard_xy_mm": [float(dash_xy[0]), float(dash_xy[1])],
            "delta_xy_mm": [float(ours_xy[0] - dash_xy[0]), float(ours_xy[1] - dash_xy[1])],
        })
    max_abs_delta = 0.0
    if deltas:
        max_abs_delta = max(max(abs(v) for v in item["delta_xy_mm"]) for item in deltas)
    return {
        "available": True,
        "method": "dashboard_triangulate._solve_matches on the raw/calibration pixels selected by the standalone matcher",
        "dashboard_matched_count": len(dash_matched),
        "dashboard_unmatched_left": len(dash_unmatched_l),
        "dashboard_unmatched_right": len(dash_unmatched_r),
        "standalone_solved_count": len(standalone_solved),
        "dashboard_solved_count": len(dash_solved),
        "max_abs_delta_mm": float(max_abs_delta),
        "matches_within_1e_6_mm": bool(max_abs_delta <= 1e-6 and len(dash_solved) == len(standalone_solved)),
        "deltas": deltas,
    }


def _read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate standalone detection/triangulation figure data from one rectified survey stereo pair."
    )
    parser.add_argument("--left", required=True, help="Path to rectified left survey image.")
    parser.add_argument("--right", required=True, help="Path to rectified right survey image.")
    parser.add_argument("--outdir", required=True, help="Output directory for PNG/CSV/JSON files.")
    parser.add_argument("--gantry-x", type=float, default=200.0, help="Survey gantry X position in mm.")
    parser.add_argument("--gantry-y", type=float, default=200.0, help="Survey gantry Y position in mm.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.70,
        help="YOLO confidence override for this figure export. Default 0.70 recovers the 5x5 survey pair.",
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Print and save a coordinate-frame trace for each matched target.",
    )
    parser.add_argument(
        "--compare-dashboard",
        action="store_true",
        help="Compare the standalone solve against dashboard_triangulate._solve_matches on the same triangulation pixels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    left_path = Path(args.left)
    right_path = Path(args.right)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    left_img = _read_image(left_path)
    right_img = _read_image(right_path)
    if left_img.shape[:2] != right_img.shape[:2]:
        raise ValueError(f"Left/right image sizes differ: {left_img.shape[:2]} vs {right_img.shape[:2]}")

    point_mode = resolve_point_mode(SURVEY_POINT_MODE)
    detector = AIDetector()
    detector.warmup()
    class_names = dict(getattr(detector.cv_left.yolo, "names", {}) or {})

    if args.confidence is not None:
        detector.cv_left.conf = float(args.confidence)
        detector.cv_right.conf = float(args.confidence)

    left_rect_dets, left_crop = _detect_side(
        detector.cv_left, left_img, "left", point_mode, classes_override=TARGET_CLASSES
    )
    right_rect_dets, right_crop = _detect_side(
        detector.cv_right, right_img, "right", point_mode, classes_override=TARGET_CLASSES
    )

    maps = _load_rectified_to_raw_maps()
    left_dets = _tag_rectified_detections(left_rect_dets, "left", maps)
    right_dets = _tag_rectified_detections(right_rect_dets, "right", maps)

    matched, unmatched_left, unmatched_right = match_points(left_dets, right_dets, verbose=True)
    matched = normalize_matches(matched)

    mover = TriangulationCoarseMover()
    if matched:
        mover.fit_epipolar(matched)
    solved_targets = mover.solve_all_from_pose(matched, args.gantry_x, args.gantry_y)
    for target_id, target in enumerate(solved_targets, start=1):
        target["target_id"] = target_id

    planned_targets = plan_targets(solved_targets, start_xy=(args.gantry_x, args.gantry_y))
    order = _planned_order_map(planned_targets)
    plot_labels = _plot_label_map(order)
    dashboard_compare = (
        _compare_with_dashboard_solver(matched, solved_targets, args.gantry_x, args.gantry_y)
        if args.compare_dashboard
        else None
    )

    left_annotated = _draw_annotations(left_img, solved_targets, "left", plot_labels)
    right_annotated = _draw_annotations(right_img, solved_targets, "right", plot_labels)

    left_out = outdir / "annotated_left.png"
    right_out = outdir / "annotated_right.png"
    left_pdf_out = outdir / "annotated_left.pdf"
    right_pdf_out = outdir / "annotated_right.pdf"
    csv_out = outdir / "triangulated_targets.csv"
    summary_out = outdir / "control_figure_summary.json"
    trace_json_out = outdir / "coordinate_trace.json"
    trace_txt_out = outdir / "coordinate_trace.txt"

    cv2.imwrite(str(left_out), left_annotated)
    cv2.imwrite(str(right_out), right_annotated)
    _write_vector_pdf(left_pdf_out, left_img, solved_targets, "left", plot_labels)
    _write_vector_pdf(right_pdf_out, right_img, solved_targets, "right", plot_labels)
    _write_csv(csv_out, solved_targets, planned_targets, mover, args.gantry_x, args.gantry_y, class_names)

    coordinate_trace = _make_coordinate_trace(
        left_path,
        right_path,
        left_img,
        right_img,
        left_dets,
        right_dets,
        matched,
        solved_targets,
        planned_targets,
        mover,
        plot_labels,
        args.gantry_x,
        args.gantry_y,
        class_names,
        maps,
        args,
        dashboard_compare=dashboard_compare,
    )
    trace_json_out.write_text(json.dumps(_json_safe(coordinate_trace), indent=2), encoding="utf-8")
    _write_debug_trace_text(trace_txt_out, _json_safe(coordinate_trace))
    if args.debug_trace:
        print("\n=== COORDINATE TRACE ===")
        print(trace_txt_out.read_text(encoding="utf-8"))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "left": str(left_path),
            "right": str(right_path),
            "assumed_frame_mode": "rectified",
            "image_size_px": [int(left_img.shape[1]), int(left_img.shape[0])],
        },
        "outputs": {
            "annotated_left": str(left_out),
            "annotated_right": str(right_out),
            "annotated_left_pdf": str(left_pdf_out),
            "annotated_right_pdf": str(right_pdf_out),
            "csv": str(csv_out),
            "summary_json": str(summary_out),
            "coordinate_trace_json": str(trace_json_out),
            "coordinate_trace_txt": str(trace_txt_out),
        },
        "counts": {
            "left_detections": len(left_rect_dets),
            "right_detections": len(right_rect_dets),
            "matched_targets": len(matched),
            "triangulated_targets": len(solved_targets),
            "unmatched_left": len(unmatched_left),
            "unmatched_right": len(unmatched_right),
        },
        "gantry_survey_position_mm": [float(args.gantry_x), float(args.gantry_y)],
        "coordinate_notes": {
            "u_v_pixels": "rectified image coordinates used for detection, matching, CSV, and annotations",
            "triangulation_pixels": "rectified detections mapped back to raw camera pixels before TriangulationCoarseMover.solve_all_from_pose",
            "x_mm_y_mm": "gantry/workspace coordinates from TriangulationCoarseMover target_xy_mm",
            "z_mm": "rectified camera-frame Z from TriangulationCoarseMover._solve_geometry; production gantry motion uses x_mm/y_mm",
            "plot_label": "path-order label used on annotated images/PDFs and in external plotting",
        },
        "config": {
            "calibration_npz": str(CALIB_NPZ_PATH),
            "rectify_maps_npz": str(RECT_NPZ_PATH),
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "survey_min_hits_config": SURVEY_MIN_HITS,
            "effective_min_hits": 1,
            "survey_point_mode": point_mode,
            "survey_yolo_imgsz": SURVEY_YOLO_IMGSZ,
            "confidence_override": args.confidence,
            "target_classes": TARGET_CLASSES,
            "avoid_classes": AVOID_CLASSES,
            "default_qpoint_model": DEFAULT_QPOINT_MODEL,
            "left_crop": left_crop,
            "right_crop": right_crop,
        },
        "model": {
            "yolo_path": str(getattr(detector, "yolo_path", "")),
            "yolo_backend": getattr(detector, "yolo_backend", None),
            "qpoint_path": str(getattr(detector, "qpoint_path", "")),
            "class_names": {str(k): v for k, v in class_names.items()},
        },
        "targets": [
            {
                "id": s["target_id"],
                "path_order": order.get(id(s)),
                "plot_label": plot_labels.get(id(s)),
                "target_xy_mm": s["target_xy_mm"],
                "left_px_rect": s["source_target"].get("left_px_rect"),
                "right_px_rect": s["source_target"].get("right_px_rect"),
                "left_px_raw": s["source_target"].get("left_px"),
                "right_px_raw": s["source_target"].get("right_px"),
                "left_box_rect": s["source_target"].get("left_box_rect"),
                "right_box_rect": s["source_target"].get("right_box_rect"),
                "left_box_raw": s["source_target"].get("left_box"),
                "right_box_raw": s["source_target"].get("right_box"),
                "confidence_left": s["source_target"].get("left_conf"),
                "confidence_right": s["source_target"].get("right_conf"),
                "score": s["source_target"].get("score"),
                "box_iou": s["source_target"].get("box_iou"),
                "y_diff_px": s["source_target"].get("y_diff_px"),
                "disparity_px": s["source_target"].get("disp_px"),
                "class_name": _match_class_name(s["source_target"], class_names),
            }
            for s in solved_targets
        ],
        "detections": {
            "left": _detection_summary(left_dets, class_names),
            "right": _detection_summary(right_dets, class_names),
        },
        "coordinate_trace": _json_safe(coordinate_trace),
    }
    summary_out.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    print("\n=== CONTROL FIGURE DATA ===")
    print(f"Left detections:                 {len(left_rect_dets)}")
    print(f"Right detections:                {len(right_rect_dets)}")
    print(f"Matched/triangulated targets:    {len(solved_targets)}")
    print(f"CSV:                             {csv_out}")
    print(f"Annotated left:                  {left_out}")
    print(f"Annotated right:                 {right_out}")
    print(f"Annotated left PDF:              {left_pdf_out}")
    print(f"Annotated right PDF:             {right_pdf_out}")
    print(f"Coordinate trace JSON:           {trace_json_out}")
    print(f"Coordinate trace text:           {trace_txt_out}")
    print(f"Summary JSON:                    {summary_out}")
    if dashboard_compare is not None:
        print(
            "Dashboard compare:               "
            f"max_abs_delta_mm={dashboard_compare.get('max_abs_delta_mm')} "
            f"matched={dashboard_compare.get('dashboard_matched_count')}"
        )


if __name__ == "__main__":
    main()
