import importlib.util as _ilu
import os as _os

# Apply pure-PyTorch NMS patch before any ultralytics import.
# torchvision C++ extensions are broken in this venv.
_nms_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "bringup", "_nms_patch.py")
if _os.path.exists(_nms_path):
    _spec = _ilu.spec_from_file_location("_nms_patch", _nms_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

import json
import math
import time
from datetime import datetime

from config import (
    HOMING,
    MOCK_GANTRY,
    DETECTOR_MODE,
    GRBL_PORT,
    SURVEY_POS_X,
    SURVEY_POS_Y,
    TRIANGULATION_ONLY_MODE,
    FULL_AUTO,
    SHOW_TRIANGULATION_PLOT,
    SHOW_MATCH_DEBUG_WINDOW,
    SAVE_MATCH_DEBUG_IMAGE,
    HAS_DISPLAY,
    MANUAL_DISPLAY_SCALE,
    AI_DISPLAY_SCALE,
    AI_CONFIDENCE,
    SURVEY_BURST_COUNT,
    SURVEY_POINT_MODE,
    SURVEY_MIN_HITS,
    SURVEY_CLUSTER_RADIUS_PX,
    SURVEY_TARGET_CLASSES,
    FINE_ALIGN_SETTLE_FRAMES,
    RECORD_TRIAL,
    TRIAL_RECORDINGS_DIR,
    OVERRIDE_BURST_NUMBER,
    OVERRIDE_BURST_COUNT,
    OVERRIDE_POINT_MODE,
    OVERRIDE_POINT_MODE_VALUE,
    WORKSPACE_X_MIN,
    WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Y_MAX,
    ENABLE_EXPERIMENT_LOGGING,
    EXPERIMENT_TRIAL_ID,
    EXPERIMENT_TRIAL_TYPE,
    EXPERIMENT_LAYOUT_TYPE,
    EXPECTED_WEED_COUNT,
    EXPECTED_KALE_COUNT,
    EXPERIMENT_NOTES,
)

from control.coarse_move import TriangulationCoarseMover, is_in_workspace
from control.fine_align import (
    fine_align_target,
    close_fine_align_window,
)
from control.strike import fire_target
from hardware.cameras import StereoCameras
from hardware.gantry import Gantry
from hardware.mock_gantry import MockGantry
from planning.target_planner import plan_targets
from pipeline.steps.match_plan import run_match_and_plan
from pipeline.steps.survey import run_survey_detection
from ui.terminal import (
    clear_current_target_line,
    print_workspace_plan,
    show_current_target,
    print_target_result,
    print_skip_target,
    print_global_survey_ready,
    print_global_survey_results,
)
from ui.workspace_plot import show_workspace_triangulation_map
from ui.triangulation_debug import show_match_debug_view
from vision.detectors.ai_detector import AIDetector
from vision.detectors.manual_detector_local import ManualDetectorLocal
from vision.matching import match_points


def build_detector():
    if DETECTOR_MODE == "manual":
        return ManualDetectorLocal(display_scale=MANUAL_DISPLAY_SCALE)
    if DETECTOR_MODE == "ai":
        return AIDetector(
            display_scale=AI_DISPLAY_SCALE,
            conf=AI_CONFIDENCE,
        )
    raise ValueError(f"Unknown DETECTOR_MODE: {DETECTOR_MODE}")


def _resolve_burst_count(default_count):
    if OVERRIDE_BURST_NUMBER:
        return max(1, int(OVERRIDE_BURST_COUNT))
    return max(1, int(default_count))


def _resolve_point_mode(default_mode):
    mode = OVERRIDE_POINT_MODE_VALUE if OVERRIDE_POINT_MODE else default_mode
    mode = str(mode or "box_center").strip().lower()
    if mode == "heatmap":
        mode = "qpoint"
    if mode not in ("box_center", "qpoint"):
        raise ValueError(f"Unknown point mode: {mode}")
    return mode


def _flush_camera_buffer(cameras, n=8):
    for _ in range(n):
        cameras.read_pair()


def _save_manifest(manifest, timestamp, recording_dir=None):
    if recording_dir is not None:
        recording_dir.mkdir(parents=True, exist_ok=True)
        path = recording_dir / "trial_summary.json"
    else:
        TRIAL_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = TRIAL_RECORDINGS_DIR / f"manifest_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Saved → {path}")


def _compact_target_list(target_queue):
    out = []
    for idx, target in enumerate(target_queue or [], start=1):
        src = target.get("source_target", {})
        xy = target.get("target_xy_mm")
        out.append({
            "id": idx,
            "target_xy_mm": list(xy) if xy is not None else None,
            "left_px": src.get("left_px"),
            "right_px": src.get("right_px"),
            "class_id": src.get("left_cls", src.get("right_cls")),
            "confidence": src.get("left_conf", src.get("right_conf", src.get("conf"))),
        })
    return out


def _compact_hits(actual_hits):
    return [
        {
            "id": idx,
            "planned_xy_mm": hit.get("planned_xy_mm"),
            "final_xy_mm": hit.get("final_xy_mm"),
            "left_px": hit.get("left_px"),
            "right_px": hit.get("right_px"),
        }
        for idx, hit in enumerate(actual_hits or [], start=1)
    ]


def _gantry_xy(gantry):
    if gantry is None:
        return None
    try:
        xy = gantry.get_estimated_xy()
        return [float(xy[0]), float(xy[1])]
    except Exception:
        return None


def _metrics_snapshot(logger):
    if logger is None or not getattr(logger, "run", None):
        return {}
    return {
        k: v
        for k, v in logger.run.items()
        if k.endswith("_time_s") or k in ("run_id", "num_targets_fired", "num_targets_attempted")
    }


def _active_target_xy(active_target):
    if not active_target:
        return None
    xy = active_target.get("target_xy_mm")
    return list(xy) if xy is not None else None


def _update_recording_context(
    cameras,
    state_name,
    gantry=None,
    target_queue=None,
    current_target_id=None,
    active_target=None,
    actual_hits=None,
    logger=None,
):
    if cameras is None:
        return
    cameras.set_recording_context(
        state_name=state_name,
        current_target_id=current_target_id,
        current_target_index=current_target_id,
        gantry_position_mm=_gantry_xy(gantry),
        planned_target_list=_compact_target_list(target_queue),
        active_target_workspace_mm=_active_target_xy(active_target),
        hit_targets_so_far=_compact_hits(actual_hits),
        metrics_timestamps=_metrics_snapshot(logger),
    )


def _attach_recording_metrics(logger, cameras):
    if logger is None or cameras is None:
        return
    stats = cameras.get_recording_stats()
    if stats:
        logger.run.update(stats)


def _add_run_total(logger, key, value):
    if logger is None or value is None:
        return
    logger.run[key] = round((logger.run.get(key) or 0.0) + float(value), 3)


def _print_final_targets(actual_hits):
    if not actual_hits:
        return
    print("\n=== FINAL ALIGNED TARGETS (PD Locked) ===")
    print(f"  {'#':>3}  {'Planned (mm)':^26}  {'Final (mm)':^26}")
    for i, hit in enumerate(actual_hits, start=1):
        cx, cy = hit.get("planned_xy_mm", [0.0, 0.0])
        fx, fy = hit.get("final_xy_mm", [0.0, 0.0])
        print(f"  {i:>3}  ({cx:>10.2f}, {cy:>10.2f})  →  ({fx:>10.2f}, {fy:>10.2f})")
    print(f"  {len(actual_hits)} target(s) locked.\n")


def _save_metrics(logger, status, cameras=None):
    if logger is None:
        return
    try:
        _attach_recording_metrics(logger, cameras)
        logger.end_run(status=status)
        logger.save_csvs()
        logger.save_json()
        logger.print_summary()
    except Exception as e:
        print(f"[METRICS] Warning: could not save metrics: {e}")


def main():
    gantry = None
    cameras = None
    state = "INIT"
    detector = None
    coarse_mover = None
    logger = None
    model_load_time_s = 0.0
    warmup_info = {}

    left_detections = []
    right_detections = []
    left_points = []
    right_points = []
    matched_targets = []
    absolute_targets = []
    target_queue = []
    actual_hits = []

    trial_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "trial_timestamp": trial_timestamp,
        "survey_position_mm": [SURVEY_POS_X, SURVEY_POS_Y],
        "targets": [],
    }

    try:
        while state != "DONE":
            if state == "INIT":
                if MOCK_GANTRY:
                    gantry = MockGantry()
                else:
                    gantry = Gantry(GRBL_PORT)
                cameras = StereoCameras()
                t_model = time.perf_counter()
                detector = build_detector()
                model_load_time_s = round(time.perf_counter() - t_model, 3)
                coarse_mover = TriangulationCoarseMover()
                coarse_mover.clear_actual_targets_log()
                if ENABLE_EXPERIMENT_LOGGING:
                    from metrics.experiment_logger import ExperimentLogger
                    logger = ExperimentLogger(config={
                        "trial_id": EXPERIMENT_TRIAL_ID,
                        "trial_type": EXPERIMENT_TRIAL_TYPE,
                        "layout_type": EXPERIMENT_LAYOUT_TYPE,
                        "expected_weed_count": EXPECTED_WEED_COUNT,
                        "expected_kale_count": EXPECTED_KALE_COUNT,
                        "workspace_width_mm": WORKSPACE_X_MAX - WORKSPACE_X_MIN,
                        "workspace_height_mm": WORKSPACE_Y_MAX - WORKSPACE_Y_MIN,
                        "notes": EXPERIMENT_NOTES,
                    })
                state = "HOME"

            elif state == "HOME":
                if DETECTOR_MODE == "ai" and hasattr(detector, "warmup"):
                    warmup_info = detector.warmup()
                cameras.open()
                _update_recording_context(cameras, "HOME", gantry, target_queue, None, None, actual_hits, logger)
                if HOMING:
                    gantry.home()
                if logger is not None:
                    logger.start_run(run_metadata={
                        "model_load_time_s": model_load_time_s,
                        **warmup_info,
                    })
                    _attach_recording_metrics(logger, cameras)
                    _update_recording_context(cameras, "HOME", gantry, target_queue, None, None, actual_hits, logger)
                state = "SURVEY"

            elif state == "SURVEY":
                _update_recording_context(cameras, "SURVEY", gantry, target_queue, None, None, actual_hits, logger)
                gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
                print_global_survey_ready(SURVEY_POS_X, SURVEY_POS_Y)
                state = "SURVEY_CONFIRM"

            elif state == "SURVEY_CONFIRM":
                state = "DETECT"

            elif state == "DETECT":
                _update_recording_context(cameras, "DETECT", gantry, target_queue, None, None, actual_hits, logger)

                if logger is not None:
                    logger.start_section("survey")
                left_detections, right_detections = run_survey_detection(cameras, detector, coarse_mover)
                if logger is not None:
                    logger.end_section("survey")
                    logger.run["num_detections_left"] = len(left_detections)
                    logger.run["num_detections_right"] = len(right_detections)
                    logger.run.update(getattr(coarse_mover, "last_survey_timing", {}))

                def _to_pt(d):
                    return d["point"] if isinstance(d, dict) else d

                left_points = [_to_pt(d) for d in left_detections]
                right_points = [_to_pt(d) for d in right_detections]
                state = "MATCH"

            elif state == "MATCH":
                _update_recording_context(cameras, "MATCH", gantry, target_queue, None, None, actual_hits, logger)
                if logger is not None:
                    logger.start_section("stereo_matching")
                matched_targets, _, _ = match_points(
                    left_detections,
                    right_detections,
                    verbose=True,
                )
                if logger is not None:
                    logger.end_section("stereo_matching")
                    logger.run["num_stereo_matches"] = len(matched_targets)

                print_global_survey_results(len(left_points), len(right_points), len(matched_targets))
                if FULL_AUTO:
                    print("[AUTO] Accepting survey result automatically.")
                    state = "PLAN"
                else:
                    user = input("Enter = accept global survey | r = rescan | q = quit: ").strip().lower()
                    if user == "r":
                        state = "DETECT"
                    elif user == "q":
                        state = "DONE"
                    else:
                        state = "PLAN"

            elif state == "PLAN":
                _update_recording_context(cameras, "PLAN", gantry, target_queue, None, None, actual_hits, logger)
                if logger is not None:
                    logger.start_section("triangulation")
                matched_targets, absolute_targets, target_queue = run_match_and_plan(
                    left_detections,
                    right_detections,
                    coarse_mover,
                    start_xy=gantry.get_estimated_xy(),
                )
                if logger is not None:
                    logger.end_section("triangulation")

                if logger is not None:
                    logger.start_section("planning")
                _update_recording_context(cameras, "PLAN", gantry, target_queue, None, None, actual_hits, logger)
                if logger is not None:
                    logger.end_section("planning")
                    logger.run["num_targets_planned"] = len(target_queue)
                    logger.compute_path_metrics(target_queue, start_xy=gantry.get_estimated_xy())

                coarse_mover.all_planned_targets = target_queue

                coarse_mover.save_workspace_targets(
                    target_queue,
                    filename="predicted_workspace_targets.json",
                )

                if SHOW_TRIANGULATION_PLOT:
                    show_workspace_triangulation_map(
                        target_queue,
                        survey_xy=(SURVEY_POS_X, SURVEY_POS_Y),
                        save_path=coarse_mover.planning_dir / "triangulation_overview.png",
                        show_window=HAS_DISPLAY,
                    )

                if (SAVE_MATCH_DEBUG_IMAGE or SHOW_MATCH_DEBUG_WINDOW) and \
                        coarse_mover.last_survey_frameL is not None and \
                        coarse_mover.last_survey_frameR is not None:
                    show_match_debug_view(
                        coarse_mover.last_survey_frameL,
                        coarse_mover.last_survey_frameR,
                        left_points,
                        right_points,
                        matched_targets,
                        absolute_targets,
                        save_path=(coarse_mover.planning_dir / "triangulation_debug_view.png") if SAVE_MATCH_DEBUG_IMAGE else None,
                        show_window=SHOW_MATCH_DEBUG_WINDOW,
                    )

                state = "EXECUTE"

            elif state == "EXECUTE":
                total = len(target_queue)
                prev_xy = gantry.get_estimated_xy()

                for i, solved in enumerate(target_queue, start=1):
                    tx, ty = solved["target_xy_mm"]
                    src = solved.get("source_target", {})
                    travel_dist = round(math.hypot(tx - prev_xy[0], ty - prev_xy[1]), 2)
                    _update_recording_context(cameras, "TARGET", gantry, target_queue, i, solved, actual_hits, logger)

                    if logger is not None:
                        logger.start_target(i, {
                            "x_target_mm": tx,
                            "y_target_mm": ty,
                            "x_commanded_mm": tx,
                            "y_commanded_mm": ty,
                            "travel_distance_mm": travel_dist,
                            "class_id": src.get("left_cls"),
                            "confidence": src.get("conf"),
                        })

                    if not is_in_workspace(tx, ty):
                        print_skip_target(i, total, solved, f"Outside workspace bounds ({tx:.1f}, {ty:.1f}) mm.")
                        if logger is not None:
                            logger.end_target(i, {"status": "skipped_out_of_bounds"})
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": None,
                            "reid_protocol": None,
                            "status": "skipped_out_of_bounds",
                        })
                        continue

                    if coarse_mover.is_duplicate_of_actual(
                        solved["target_xy_mm"],
                        actual_hits,
                        tol_mm=8.0,
                    ):
                        print_skip_target(i, total, solved, "Already covered by a previous PD lock.")
                        if logger is not None:
                            logger.end_target(i, {"status": "skipped_duplicate"})
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": None,
                            "reid_protocol": None,
                            "status": "skipped_duplicate",
                        })
                        continue

                    show_current_target(i, total, solved)
                    cameras.clear_overlay()
                    cameras.set_recording_status([
                        f"Target {i}/{total}",
                        f"Coarse: ({tx:.1f}, {ty:.1f}) mm",
                        "Moving to target...",
                    ])

                    if logger is not None:
                        logger.start_target_section(i, "travel")
                    _update_recording_context(cameras, "TRAVEL", gantry, target_queue, i, solved, actual_hits, logger)
                    moved = coarse_mover.move_to_absolute_target(gantry, solved)
                    if logger is not None:
                        logger.end_target_section(i, "travel")
                    _update_recording_context(cameras, "POST_TRAVEL", gantry, target_queue, i, solved, actual_hits, logger)

                    if not moved:
                        if logger is not None:
                            logger.end_target(i, {"status": "skipped_move_failed"})
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": None,
                            "reid_protocol": None,
                            "status": "skipped_move_failed",
                        })
                        continue

                    prev_xy = (tx, ty)

                    if TRIANGULATION_ONLY_MODE:
                        gantry.sync_estimate_to_machine()
                        final_xy = gantry.get_estimated_xy()
                        actual_entry = {
                            "planned_xy_mm": [float(tx), float(ty)],
                            "selected_local_xy_mm": [float(tx), float(ty)],
                            "final_xy_mm": [float(final_xy[0]), float(final_xy[1])],
                            "left_px": src.get("left_px"),
                            "right_px": src.get("right_px"),
                            "score": float(src.get("score", 1.0)),
                        }
                        actual_hits.append(actual_entry)
                        coarse_mover.append_actual_target(
                            solved, solved, final_xy,
                            filename="actual_pd_targets.json",
                        )
                        print_target_result(i, total, solved, actual_entry)
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": list(final_xy),
                            "reid_protocol": None,
                            "status": "triangulation_only",
                        })
                        if logger is not None:
                            logger.end_target(i, {
                                "x_final_mm": float(final_xy[0]),
                                "y_final_mm": float(final_xy[1]),
                                "fired": False,
                                "status": "triangulation_only",
                            })
                        if not FULL_AUTO:
                            user = input("Triangulation only: Enter = next | q = quit: ").strip().lower()
                            if user == "q":
                                state = "DONE"
                                break
                        continue

                    cameras.set_recording_status([
                        f"Target {i}/{total}",
                        f"Coarse: ({tx:.1f}, {ty:.1f}) mm",
                        "Fine aligning...",
                    ])

                    if logger is not None:
                        logger.start_target_section(i, "pd")
                    _update_recording_context(cameras, "FINE_ALIGN", gantry, target_queue, i, solved, actual_hits, logger)
                    aligned, actual_entry = fine_align_target(
                        gantry,
                        cameras,
                        detector,
                        coarse_mover,
                        solved,
                        actual_hits,
                        settle_frames=FINE_ALIGN_SETTLE_FRAMES,
                        show_debug=HAS_DISPLAY,
                        target_idx=i,
                        total_targets=total,
                    )
                    if logger is not None:
                        logger.end_target_section(i, "pd")
                        fa_timing = dict(getattr(fine_align_target, "last_timing", {}))
                        if fa_timing:
                            logger.update_target(i, fa_timing)
                            _add_run_total(logger, "total_fine_align_reid_yolo_time_s", fa_timing.get("fine_align_reid_yolo_time_s"))
                            _add_run_total(logger, "total_fine_align_reid_time_s", fa_timing.get("fine_align_reid_total_time_s"))
                            _add_run_total(logger, "total_fine_align_pd_lk_time_s", fa_timing.get("fine_align_pd_lk_time_s"))
                            _add_run_total(logger, "total_final_snap_time_s", fa_timing.get("final_snap_time_s"))

                    if aligned:
                        actual_hits.append(actual_entry)
                        _update_recording_context(cameras, "LOCKED", gantry, target_queue, i, solved, actual_hits, logger)
                        print_target_result(i, total, solved, actual_entry)

                        if logger is not None:
                            logger.start_target_section(i, "fire")
                        _update_recording_context(cameras, "FIRE", gantry, target_queue, i, solved, actual_hits, logger)
                        fire_target(gantry, solved, cameras=cameras)
                        if logger is not None:
                            logger.end_target_section(i, "fire")
                        _update_recording_context(cameras, "FIRED", gantry, target_queue, i, solved, actual_hits, logger)

                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": actual_entry.get("final_xy_mm"),
                            "reid_protocol": actual_entry.get("reid_protocol"),
                            "status": "locked_fired",
                        })
                        if logger is not None:
                            final_xy = actual_entry.get("final_xy_mm") or [tx, ty]
                            logger.end_target(i, {
                                "x_final_mm": float(final_xy[0]),
                                "y_final_mm": float(final_xy[1]),
                                "fired": True,
                                "pd_converged": True,
                                "status": "locked_fired",
                            })
                    else:
                        _update_recording_context(cameras, "FAILED_FINE_ALIGN", gantry, target_queue, i, solved, actual_hits, logger)
                        cameras.set_recording_status([
                            f"Target {i}/{total}",
                            f"Coarse: ({tx:.1f}, {ty:.1f}) mm",
                            "FAILED: fine align",
                        ])
                        print_skip_target(i, total, solved, "Fine align failed")
                        manifest["targets"].append({
                            "target_id": i,
                            "survey_pixel_left": src.get("left_px"),
                            "survey_pixel_right": src.get("right_px"),
                            "coarse_triangulated_mm": [tx, ty],
                            "fine_align_final_mm": None,
                            "reid_protocol": None,
                            "status": "failed_fine_align",
                        })
                        if logger is not None:
                            logger.end_target(i, {
                                "pd_converged": False,
                                "fired": False,
                                "status": "failed_fine_align",
                            })

                close_fine_align_window()
                clear_current_target_line()
                print("\n  All targets complete.")
                _print_final_targets(actual_hits)
                if cameras is not None:
                    cameras.stop_recording()
                _save_metrics(logger, "complete", cameras)
                state = "DONE"

        print("\n=== DONE ===")

    except KeyboardInterrupt:
        clear_current_target_line()
        print("\nInterrupted by user.")
        _save_metrics(logger, "user_aborted", cameras)

    except Exception as e:
        clear_current_target_line()
        print(f"\nERROR: {e}")
        _save_metrics(logger, "failed", cameras)

    finally:
        recording_dir = cameras.get_recording_dir() if cameras is not None else None
        _save_manifest(manifest, trial_timestamp, recording_dir=recording_dir)
        if cameras is not None:
            cameras.stop_recording()
            cameras.close()
        if gantry is not None:
            gantry.close()


if __name__ == "__main__":
    main()
