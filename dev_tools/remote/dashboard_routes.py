"""Flask route registration for the remote dashboard.

This file owns URLs only.

Big logic belongs in:
- dashboard_camera.py
- dashboard_yolo.py
- dashboard_rectify.py
- dashboard_gantry.py
"""

from flask import Response, jsonify, render_template, request

from dashboard_state import state
from dashboard_workspace3d import register_workspace3d_routes
from dashboard_fine_align import register_fine_align_routes
from dashboard_triangulate import register_triangulate_routes
from dashboard_camera_tuner import register_camera_tuner_routes
from dashboard_survey_photos import register_survey_photo_routes

from dashboard_camera import (
    get_preview_frame,
    reset_cameras_sequence,
    close_all,
    mjpeg_generator,
    ensure_cameras,
)

from dashboard_images import (
    b64_img,
    draw_horizontal_lines,
)

from dashboard_rectify import (
    rectify_pair,
)

from dashboard_yolo import (
    get_debug_defaults,
    get_class_names,
    get_detector_info,
    get_segmentation_model_options,
    run_debug_scan,
    run_validation_scan,
    run_cached_match,
    run_survey_and_cache,
)

from dashboard_gantry import (
    ensure_gantry,
    close_gantry,
    gantry_status_payload,
    gantry_position_payload,
    unlock_gantry,
    reset_gantry,
    stop_gantry,
    home_gantry,
    jog_gantry,
    move_absolute_gantry,
    raw_gcode,
)

from dashboard_settings import (
    get_page_settings,
    update_page_settings,
    save_survey_config_to_config_py,
    save_fine_align_config_to_config_py,
)

from dashboard_scout import (
    scout_check_connection,
    scout_stop,
    scout_move_forward,
    scout_move_backward,
    scout_status_payload,
    close_scout,
)


# =============================================================================
# Route registration
# =============================================================================

def register_routes(app):
    register_workspace3d_routes(app)
    register_fine_align_routes(app)
    register_triangulate_routes(app)
    register_camera_tuner_routes(app)
    register_survey_photo_routes(app)

    # =========================================================================
    # Page routes
    # =========================================================================

    @app.route("/")
    def index():
        return render_template("base.html")


    @app.route("/camera")
    def camera_page():
        return render_template("camera.html")


    @app.route("/survey")
    def survey_page():
        defaults = get_debug_defaults("survey")
        return render_template("yolo_debug.html", kind="survey", **defaults)


    @app.route("/fine")
    def fine_page():
        defaults = get_debug_defaults("fine")
        return render_template("yolo_debug.html", kind="fine", **defaults)


    @app.route("/yolo_validation")
    def yolo_validation_page():
        defaults = get_debug_defaults("survey")
        return render_template("yolo_validation.html", **defaults)


    @app.route("/match")
    def match_page():
        return render_template("match.html")


    @app.route("/rectify")
    def rectify_page():
        return render_template("rectify.html")


    @app.route("/gantry")
    def gantry_page():
        return render_template("gantry.html")


    @app.route("/scout")
    def scout_page():
        return render_template("scout.html")


    # =========================================================================
    # Persistent dashboard settings API
    # =========================================================================

    allowed_settings_pages = {
        "survey",
        "fine",
        "yolo_validation",
        "fine_align",
        "match",
        "workspace3d",
        "gantry",
        "camera",
        "survey_photos",
    }

    @app.route("/api/settings/<page_name>", methods=["GET"])
    def api_get_page_settings(page_name):
        try:
            page = str(page_name or "").strip()
            if page not in allowed_settings_pages:
                return jsonify({"ok": False, "error": f"Unknown page: {page}"}), 404

            return jsonify({
                "ok": True,
                "page": page,
                "settings": get_page_settings(page),
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    @app.route("/api/save_survey_config", methods=["POST"])
    def api_save_survey_config():
        try:
            data = request.get_json(silent=True) or {}
            suggested = data.get("suggested_config", data)
            if not isinstance(suggested, dict):
                return jsonify({"ok": False, "error": "No suggested_config provided"}), 400

            updated = save_survey_config_to_config_py(suggested)
            names = [u["var"] for u in updated]
            return jsonify({
                "ok": True,
                "updated": updated,
                "message": f"Saved {len(updated)} value(s) to config.py: {', '.join(names)}",
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    @app.route("/api/save_fine_align_config", methods=["POST"])
    def api_save_fine_align_config():
        try:
            data = request.get_json(silent=True) or {}
            config_values = data.get("config_values", data)
            if not isinstance(config_values, dict):
                return jsonify({"ok": False, "error": "No config_values provided"}), 400

            updated = save_fine_align_config_to_config_py(config_values)
            names = [u["var"] for u in updated]
            return jsonify({
                "ok": True,
                "updated": updated,
                "message": f"Saved {len(updated)} value(s): {', '.join(names)}",
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    @app.route("/api/settings/<page_name>", methods=["POST"])
    def api_post_page_settings(page_name):
        try:
            page = str(page_name or "").strip()
            if page not in allowed_settings_pages:
                return jsonify({"ok": False, "error": f"Unknown page: {page}"}), 404

            data = request.get_json(silent=True) or {}
            values = data.get("settings", data)
            if not isinstance(values, dict):
                values = {}

            settings = update_page_settings(page, values)
            return jsonify({
                "ok": True,
                "page": page,
                "settings": settings,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    # =========================================================================
    # Model / camera API
    # =========================================================================

    @app.route("/api/model_info", methods=["POST"])
    def api_model_info():
        try:
            return jsonify({
                "ok": True,
                "class_names": get_class_names(),
                "model_info": get_detector_info(),
                "model_options": get_segmentation_model_options(),
                "detector_loaded": state.detector is not None,
                "note": "Class names appear after YOLO is loaded by Scan / Save Points."
            })
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    @app.route("/api/preview_frame", methods=["POST"])
    def api_preview_frame():
        try:
            params = request.get_json(force=True)
            with state.camera_lock:
                result = get_preview_frame(params)
            return jsonify(result)
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    @app.route("/api/close_cameras", methods=["POST"])
    def api_close_cameras():
        try:
            with state.camera_lock:
                close_all()
            return jsonify({"ok": True, "message": "Cameras closed."})
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    @app.route("/api/reset_cameras", methods=["POST"])
    def api_reset_cameras():
        try:
            with state.camera_lock:
                result = reset_cameras_sequence()
            result["ok"] = True
            return jsonify(result)
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    # =========================================================================
    # YOLO / matching API
    # =========================================================================

    @app.route("/api/run_scan", methods=["POST"])
    def api_run_scan():
        try:
            params = request.get_json(force=True)
            with state.camera_lock:
                result = run_debug_scan(params)
            result["ok"] = True
            return jsonify(result)
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    @app.route("/api/run_survey", methods=["POST"])
    def api_run_survey():
        return api_run_scan()


    @app.route("/api/run_validation_scan", methods=["POST"])
    def api_run_validation_scan():
        try:
            params = {k: v for k, v in request.form.items()}
            files = request.files.getlist("images")
            if not files:
                return jsonify({"ok": False, "error": "No images uploaded."}), 400

            with state.camera_lock:
                result = run_validation_scan(params, files)
            result["ok"] = True
            return jsonify(result)
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    @app.route("/api/run_survey_full", methods=["POST"])
    def api_run_survey_full():
        try:
            params = request.get_json(force=True)
            with state.camera_lock:
                result = run_survey_and_cache(params)
            result["ok"] = True
            return jsonify(result)
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    @app.route("/api/run_match", methods=["POST"])
    def api_run_match():
        try:
            with state.camera_lock:
                result = run_cached_match()
            result["ok"] = True
            return jsonify(result)
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    # =========================================================================
    # Rectification API
    # =========================================================================

    @app.route("/api/rectify_preview", methods=["POST"])
    def api_rectify_preview():
        try:
            cam = ensure_cameras()

            with state.camera_lock:
                left_raw, right_raw = cam.read_pair()

            left_rect, right_rect, map_keys = rectify_pair(left_raw, right_raw)

            left_raw_lines = draw_horizontal_lines(left_raw)
            right_raw_lines = draw_horizontal_lines(right_raw)
            left_rect_lines = draw_horizontal_lines(left_rect)
            right_rect_lines = draw_horizontal_lines(right_rect)

            return jsonify({
                "ok": True,
                "left_raw": b64_img(left_raw_lines),
                "right_raw": b64_img(right_raw_lines),
                "left_rect": b64_img(left_rect_lines),
                "right_rect": b64_img(right_rect_lines),
                "map_keys": map_keys,
            })

        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500


    # =========================================================================
    # Gantry API
    # =========================================================================

    def _gantry_error_payload(e):
        return jsonify({
            "ok": False,
            "error": repr(e),
        }), 500


    @app.route("/api/gantry/open", methods=["POST"])
    def api_gantry_open():
        try:
            with state.gantry_lock:
                ensure_gantry()
                payload = gantry_status_payload()
            payload["ok"] = True
            payload["message"] = "Gantry opened."
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/status", methods=["POST"])
    def api_gantry_status():
        try:
            with state.gantry_lock:
                payload = gantry_status_payload()
            payload["ok"] = True
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/position", methods=["POST"])
    def api_gantry_position():
        try:
            with state.gantry_lock:
                payload = gantry_position_payload()
            payload["ok"] = True
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/close", methods=["POST"])
    def api_gantry_close():
        try:
            with state.gantry_lock:
                close_gantry()
            return jsonify({
                "ok": True,
                "connected": False,
                "message": "Gantry closed."
            })
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/unlock", methods=["POST"])
    def api_gantry_unlock():
        try:
            with state.gantry_lock:
                payload = unlock_gantry()
            payload["ok"] = True
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/reset", methods=["POST"])
    def api_gantry_reset():
        try:
            with state.gantry_lock:
                payload = reset_gantry()
            payload["ok"] = True
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/stop", methods=["POST"])
    def api_gantry_stop():
        try:
            with state.gantry_lock:
                payload = stop_gantry()
            payload["ok"] = True
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/home", methods=["POST"])
    def api_gantry_home():
        try:
            with state.gantry_lock:
                payload = home_gantry()
            payload["ok"] = True
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/jog", methods=["POST"])
    def api_gantry_jog():
        try:
            data = request.get_json(silent=True) or {}
            dx = float(data.get("dx", 0.0))
            dy = float(data.get("dy", 0.0))

            feed_raw = data.get("feed", None)
            feed = None if feed_raw in (None, "", "None") else float(feed_raw)

            with state.gantry_lock:
                payload = jog_gantry(dx, dy, feed=feed)

            payload["ok"] = True
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/move_absolute", methods=["POST"])
    def api_gantry_move_absolute():
        try:
            data = request.get_json(silent=True) or {}
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))

            feed_raw = data.get("feed", None)
            feed = None if feed_raw in (None, "", "None") else float(feed_raw)

            with state.gantry_lock:
                payload = move_absolute_gantry(x, y, feed=feed)

            payload["ok"] = True
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    @app.route("/api/gantry/raw", methods=["POST"])
    def api_gantry_raw():
        try:
            data = request.get_json(silent=True) or {}
            cmd = data.get("cmd", "")

            with state.gantry_lock:
                payload = raw_gcode(cmd)

            payload["ok"] = True
            return jsonify(payload)
        except Exception as e:
            return _gantry_error_payload(e)


    # =========================================================================
    # Scout API
    # =========================================================================

    def _scout_error_payload(e):
        return jsonify({"ok": False, "error": repr(e)}), 500

    @app.route("/api/scout/status", methods=["GET"])
    def api_scout_status():
        try:
            return jsonify(scout_status_payload())
        except Exception as e:
            return _scout_error_payload(e)

    @app.route("/api/scout/check", methods=["POST"])
    def api_scout_check():
        try:
            with state.scout_lock:
                result = scout_check_connection()
            result.setdefault("ok", False)
            return jsonify(result)
        except Exception as e:
            return _scout_error_payload(e)

    @app.route("/api/scout/stop", methods=["POST"])
    def api_scout_stop():
        try:
            with state.scout_lock:
                result = scout_stop()
            result.setdefault("ok", False)
            return jsonify(result)
        except Exception as e:
            return _scout_error_payload(e)

    @app.route("/api/scout/move", methods=["POST"])
    def api_scout_move():
        try:
            data = request.get_json(silent=True) or {}
            distance_m = float(data.get("distance_m", 0.25))
            speed_mps  = float(data.get("speed_mps",  0.10))
            direction  = str(data.get("direction", "forward")).strip().lower()
            timeout_s  = data.get("timeout_s", None)
            if timeout_s is not None:
                timeout_s = float(timeout_s)
            dry_run = bool(data.get("dry_run", True))

            if distance_m <= 0 or speed_mps <= 0:
                return jsonify({"ok": False, "error": "distance_m and speed_mps must be > 0"}), 400
            if direction not in ("forward", "backward"):
                return jsonify({"ok": False, "error": "direction must be 'forward' or 'backward'"}), 400

            with state.scout_lock:
                move_fn = scout_move_backward if direction == "backward" else scout_move_forward
                result = move_fn(
                    distance_m=distance_m,
                    speed_mps=speed_mps,
                    timeout_s=timeout_s,
                    dry_run=dry_run,
                )
            result.setdefault("ok", False)
            return jsonify(result)
        except Exception as e:
            return _scout_error_payload(e)

    @app.route("/api/scout/close", methods=["POST"])
    def api_scout_close():
        try:
            with state.scout_lock:
                close_scout()
            return jsonify({"ok": True, "message": "Scout closed."})
        except Exception as e:
            return _scout_error_payload(e)

    # =========================================================================
    # MJPEG stream
    # =========================================================================

    @app.route("/stream/<side>")
    def stream(side):
        if side not in ("left", "right"):
            return "bad side", 400

        rectified = request.args.get("rectified", "0").strip().lower() in ("1", "true", "yes", "on")

        return Response(
            mjpeg_generator(side, rectified=rectified),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
