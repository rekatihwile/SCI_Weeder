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
    run_debug_scan,
    run_cached_match,
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
)


# =============================================================================
# Route registration
# =============================================================================

def register_routes(app):
    register_workspace3d_routes(app)
    register_fine_align_routes(app)

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


    @app.route("/match")
    def match_page():
        return render_template("match.html")


    @app.route("/rectify")
    def rectify_page():
        return render_template("rectify.html")


    @app.route("/gantry")
    def gantry_page():
        return render_template("gantry.html")


    # =========================================================================
    # Persistent dashboard settings API
    # =========================================================================

    allowed_settings_pages = {
        "survey",
        "fine",
        "fine_align",
        "match",
        "workspace3d",
        "gantry",
        "camera",
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