

import importlib.util as _ilu
from pathlib import Path as _Path

# Apply NMS patch before any YOLO/AIDetector import.
_patch_path = _Path(__file__).resolve().parents[2] / "bringup" / "_nms_patch.py"
if _patch_path.exists():
    _spec = _ilu.spec_from_file_location("_nms_patch", _patch_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

from flask import jsonify, render_template, request

from dashboard_state import state
from dashboard_gantry import ensure_gantry
from dashboard_camera import ensure_cameras
from dashboard_images import b64_img
from dashboard_yolo import get_debug_defaults, ensure_detector, get_class_names
from control.fine_align_reid import run_fine_align_reid


# =============================================================================
# Small helpers
# =============================================================================

def _truthy(value):
    return str(value).strip().lower() not in ("0", "false", "no", "off", "none", "")


def _as_text(value, default=""):
    if value is None:
        return default
    return str(value)


def _as_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _optional_float(value):
    text = _as_text(value, "").strip().lower()
    if text in ("", "none", "null"):
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_expected_class(value):
    text = _as_text(value, "").strip().lower()
    if text in ("", "all", "none", "null"):
        return None
    # If multiple classes are provided, use the first for strict Re-ID filtering.
    first = text.split(",", 1)[0].strip()
    try:
        return int(first)
    except Exception:
        return None


# =============================================================================
# Route registration
# =============================================================================

def register_fine_align_routes(app):

    # ------------------------------------------------------------------ #
    # Page                                                                 #
    # ------------------------------------------------------------------ #

    @app.route("/fine_align")
    def fine_align_page():
        defaults = get_debug_defaults("fine")
        return render_template(
            "fine_align.html",
            title="Fine Align / Re-ID Unified Debug",
            kind="fine_align",
            default_mode=defaults["default_mode"],
            default_crop_w=defaults["default_crop_w"],
            default_crop_h=defaults["default_crop_h"],
            default_burst=defaults["default_burst"],
            default_min_hits=defaults["default_min_hits"],
        )

    # ------------------------------------------------------------------ #
    # Load cached plan                                                     #
    # ------------------------------------------------------------------ #

    @app.route("/api/fine_align/load_plan", methods=["POST"])
    def api_fine_align_load_plan():
        try:
            from pipeline.steps.fine_align_debug import load_latest_plan, latest_plan_path

            plan = load_latest_plan()
            return jsonify({
                "ok": True,
                "plan": plan,
                "targets": plan["targets"],
                "path": str(latest_plan_path()),
            })

        except FileNotFoundError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    # ------------------------------------------------------------------ #
    # Coarse move                                                          #
    # ------------------------------------------------------------------ #

    @app.route("/api/fine_align/move_coarse", methods=["POST"])
    def api_fine_align_move_coarse():
        try:
            data = request.get_json(force=True) or {}

            target_id = int(data.get("target_id", 0))
            feed_raw = data.get("feed", None)
            feed = None if feed_raw in (None, "", "None") else float(feed_raw)

            from pipeline.steps.fine_align_debug import (
                load_latest_plan,
                get_cached_target,
                move_to_cached_target,
            )

            plan = load_latest_plan()
            target = get_cached_target(plan, target_id)

            with state.gantry_lock:
                gantry = ensure_gantry()
                result = move_to_cached_target(gantry, target, feed=feed)

            result["ok"] = True
            return jsonify(result)

        except FileNotFoundError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    # ------------------------------------------------------------------ #
    # Re-ID once via unified runtime function                                #
    # ------------------------------------------------------------------ #

    @app.route("/api/fine_align/reid_once", methods=["POST"])
    def api_fine_align_reid_once():
        """Run one fine-align Re-ID pass via control.fine_align_reid.run_fine_align_reid."""
        try:
            data = request.get_json(force=True) or {}
            target_id = _as_int(data.get("target_id", 0), 0)

            from pipeline.steps.fine_align_debug import load_latest_plan, get_cached_target

            plan = load_latest_plan()
            target = get_cached_target(plan, target_id)

            target_class = target.get("class_id")
            requested_class = _parse_expected_class(data.get("classes"))
            expected_cls = requested_class if requested_class is not None else target_class

            crop_w = _as_int(data.get("crop_w", 384), 384)
            crop_h = _as_int(data.get("crop_h", 384), 384)
            burst_count = _as_int(data.get("burst_count", 5), 5)
            min_hits = _as_int(data.get("min_hits", 1), 1)
            point_mode = _as_text(data.get("point_mode"), "box_center")
            use_rectified = _truthy(data.get("rectified", data.get("use_rectified", "1")))
            y_gate_px = _as_float(data.get("y_gate_px", 5), 5)
            min_disp_px = _as_float(data.get("min_disp_px", 10), 10)
            max_disp_px = _as_float(data.get("max_disp_px", 500), 500)
            imgsz_raw = data.get("imgsz")
            imgsz = _as_int(imgsz_raw, 0) if imgsz_raw not in (None, "", "None") else None
            conf_override = _optional_float(data.get("conf"))

            with state.camera_lock:
                cameras = ensure_cameras()
                detector = ensure_detector()
                reid = run_fine_align_reid(
                    cameras=cameras,
                    detector=detector,
                    target=target,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    burst_count=burst_count,
                    min_hits=min_hits,
                    point_mode=point_mode,
                    class_filter=expected_cls,
                    conf_override=conf_override,
                    imgsz=imgsz,
                    use_rectified=use_rectified,
                    y_gate_px=y_gate_px,
                    min_disp_px=min_disp_px,
                    max_disp_px=max_disp_px,
                    return_debug=True,
                )

            timing = dict(reid.get("timing", {}))
            debug_frames = reid.get("debug_frames") or {}
            left_overlay = debug_frames.get("left_overlay")
            if left_overlay is None:
                left_overlay = debug_frames.get("left_full")
            right_overlay = debug_frames.get("right_overlay")
            if right_overlay is None:
                right_overlay = debug_frames.get("right_full")

            return jsonify({
                "ok": bool(reid.get("ok", False)),
                "mode": "fine_align_reid_unified",
                "cv_path": "control.fine_align_reid.run_fine_align_reid",
                "frame_mode": reid.get("frame_mode"),
                "rectified": bool(use_rectified),
                "target_id": target_id,
                "expected_cls": expected_cls,
                "left_count": len(reid.get("left_detections", [])),
                "right_count": len(reid.get("right_detections", [])),
                "matched_count": len(reid.get("matches", [])),
                "timing": timing,
                "capture_s": timing.get("read_burst_s"),
                "yolo_s": float(timing.get("yolo_left_s", 0.0)) + float(timing.get("yolo_right_s", 0.0)),
                "match_s": timing.get("match_s"),
                "crop": reid.get("crop"),
                "left_detections": reid.get("left_detections", []),
                "right_detections": reid.get("right_detections", []),
                "matches": reid.get("matches", []),
                "chosen": reid.get("chosen"),
                "requested_classes": expected_cls,
                "confidence_override": conf_override,
                "imgsz": imgsz,
                "class_names": get_class_names(det=None),
                "left_image": b64_img(left_overlay) if left_overlay is not None else None,
                "right_image": b64_img(right_overlay) if right_overlay is not None else None,
                "error": reid.get("error"),
            })

        except FileNotFoundError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        except KeyError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        except Exception as e:
            import traceback
            return jsonify({
                "ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }), 500
