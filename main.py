import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

from config import (
    HOMING,
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
    CAMERA_SETTINGS,
    SURVEY_BURST_COUNT,
    SURVEY_POINT_MODE,
    SURVEY_MIN_HITS,
    SURVEY_CLUSTER_RADIUS_PX,
    SURVEY_CROP_MODE,
    SURVEY_CROP_HALF_X_PX,
    SURVEY_CROP_HALF_Y_PX,
    SURVEY_YOLO_IMGSZ,
    SURVEY_TARGET_CLASSES,
    SURVEY_CAN_TARGET_CLASSES,
    SURVEY_CANT_TARGET_CLASSES,
    SURVEY_DETECT_ALL_CLASSES,
    SURVEY_DETECT_CLASS_IDS,
    FINE_ALIGN_REID_BURST_COUNT,
    FINE_ALIGN_REID_CROP_HALF_PX,
    FINE_ALIGN_MAX_TIME_SEC,
    FINE_ALIGN_DEADZONE_PX,
    FINE_ALIGN_ENABLE_SNAP,
    FINE_ALIGN_SETTLE_FRAMES,
    YOLO_WARMUP,
    YOLO_WARMUP_IMGSZ,
    YOLO_WARMUP_ITERS,
    RECORD_TRIAL,
    TRIAL_RECORDINGS_DIR,
    AUTO_RENDER_TRIAL,
    AUTO_RENDER_DELETE_RAW,
    RUN_PIPELINE_CYCLES,
    OVERRIDE_BURST_ENABLED,
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
from planning.target_planner import plan_targets
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
    if OVERRIDE_BURST_ENABLED:
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


def _normalise_class_selector(spec):
    """Normalise None/int/list-like class selectors into a set[int] or None."""
    if spec is None:
        return None
    if isinstance(spec, int):
        return {int(spec)}

    out = set()
    for v in spec:
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return out if out else set()


def _filter_survey_detections_by_class(detections, can_target_set, cant_target_set):
    """Apply class allow/deny policy to rich survey detections.

    Deny list always wins over allow list.
    """
    if not detections:
        return [], {"kept": 0, "dropped": 0, "dropped_by_cant": 0, "dropped_by_can": 0}

    kept = []
    dropped_by_cant = 0
    dropped_by_can = 0

    for det in detections:
        if not isinstance(det, dict):
            kept.append(det)
            continue

        cls_id = det.get("cls")
        try:
            cls_id = int(cls_id) if cls_id is not None else None
        except (TypeError, ValueError):
            cls_id = None

        if cant_target_set and cls_id is not None and cls_id in cant_target_set:
            dropped_by_cant += 1
            continue

        if can_target_set is not None:
            if cls_id is None or cls_id not in can_target_set:
                dropped_by_can += 1
                continue

        kept.append(det)

    dropped = len(detections) - len(kept)
    return kept, {
        "kept": len(kept),
        "dropped": dropped,
        "dropped_by_cant": dropped_by_cant,
        "dropped_by_can": dropped_by_can,
    }


def _class_counts(detections):
    counts = Counter()
    for det in detections or []:
        if not isinstance(det, dict):
            continue
        cls_id = det.get("cls")
        try:
            cls_id = int(cls_id)
        except (TypeError, ValueError):
            continue
        counts[cls_id] += 1
    return dict(sorted(counts.items()))


def _matched_class_counts(matched_targets):
    counts = Counter()
    for target in matched_targets or []:
        cls_id = target.get("left_cls", target.get("right_cls"))
        try:
            cls_id = int(cls_id)
        except (TypeError, ValueError):
            continue
        counts[cls_id] += 1
    return dict(sorted(counts.items()))


def _det_point(det):
    if isinstance(det, dict):
        return tuple(map(int, det["point"]))
    return tuple(map(int, det))


def _match_entry_from_detections(left_det, right_det=None, right_px=None, score=0.25, repair_mode=None):
    lp = _det_point(left_det)
    rp = _det_point(right_det) if right_det is not None else tuple(map(int, right_px))
    dx = float(rp[0] - lp[0])
    dy = float(rp[1] - lp[1])

    entry = {
        "left_px": lp,
        "right_px": rp,
        "score": float(score),
        "y_diff_px": float(abs(dy)),
        "disp_px": float(abs(dx)),
        "dx_px": dx,
        "dy_px": dy,
    }
    if repair_mode:
        entry["survey_repair"] = repair_mode

    if isinstance(left_det, dict):
        entry["left_cls"] = left_det.get("cls")
        entry["left_conf"] = left_det.get("conf")
        entry["left_views"] = left_det.get("views")
        if "box" in left_det:
            entry["left_box"] = left_det.get("box")
    if isinstance(right_det, dict):
        entry["right_cls"] = right_det.get("cls")
        entry["right_conf"] = right_det.get("conf")
        entry["right_views"] = right_det.get("views")
        if "box" in right_det:
            entry["right_box"] = right_det.get("box")
    elif isinstance(left_det, dict) and "box" in left_det:
        lx1, ly1, lx2, ly2 = left_det["box"]
        entry["right_box"] = (lx1 + dx, ly1 + dy, lx2 + dx, ly2 + dy)
        entry["synthetic_right"] = True

    return entry


def _repair_survey_matches(matched_targets, left_detections, right_detections, expected_count=None):
    """Recover sparse dark-lab survey misses using the fitted stereo offset.

    The normal constellation matcher is still the source of truth. This pass is
    intentionally conservative: it only runs while the survey is short of the
    configured expected count, first rematches real unmatched detections, then
    fills at most the remaining gap from strong left-only detections.
    """
    if not expected_count:
        return matched_targets
    expected_count = int(expected_count)
    if len(matched_targets) < 4:
        return matched_targets

    dxs = [float(m.get("dx_px", m["right_px"][0] - m["left_px"][0])) for m in matched_targets]
    dys = [float(m.get("dy_px", m["right_px"][1] - m["left_px"][1])) for m in matched_targets]
    med_dx = float(median(dxs))
    med_dy = float(median(dys))

    def _offset_error(match):
        dx = float(match.get("dx_px", match["right_px"][0] - match["left_px"][0]))
        dy = float(match.get("dy_px", match["right_px"][1] - match["left_px"][1]))
        return math.hypot(dx - med_dx, dy - med_dy)

    def _view_count(match):
        views = 0
        for key in ("left_views", "right_views"):
            try:
                views += int(match.get(key) or 0)
            except (TypeError, ValueError):
                pass
        return views

    kept = []
    dropped = []
    for m in matched_targets:
        dx = float(m.get("dx_px", m["right_px"][0] - m["left_px"][0]))
        dy = float(m.get("dy_px", m["right_px"][1] - m["left_px"][1]))
        score = float(m.get("score", 1.0))
        dx_dev = abs(dx - med_dx)
        dy_dev = abs(dy - med_dy)
        if (score < 0.25 and (dx_dev > 10.0 or dy_dev > 8.0)) or dx_dev > 16.0 or dy_dev > 12.0:
            dropped.append(m)
        else:
            kept.append(m)

    matched_left = {tuple(m["left_px"]) for m in kept}
    matched_right = {tuple(m["right_px"]) for m in kept}

    unmatched_left = [d for d in left_detections if _det_point(d) not in matched_left]
    unmatched_right = [d for d in right_detections if _det_point(d) not in matched_right]

    # First, repair with actual right detections near the median stereo offset.
    pair_candidates = []
    for left_det in unmatched_left:
        lp = _det_point(left_det)
        pred = (lp[0] + med_dx, lp[1] + med_dy)
        for right_det in unmatched_right:
            rp = _det_point(right_det)
            dist = math.hypot(rp[0] - pred[0], rp[1] - pred[1])
            if dist <= 20.0:
                pair_candidates.append((dist, left_det, right_det))

    used_left = set()
    used_right = set()
    repaired = []
    trimmed = []
    for dist, left_det, right_det in sorted(pair_candidates, key=lambda x: x[0]):
        lp = _det_point(left_det)
        rp = _det_point(right_det)
        if lp in used_left or rp in used_right:
            continue
        score = max(0.15, 0.65 * (1.0 - dist / 20.0))
        entry = _match_entry_from_detections(left_det, right_det, score=score, repair_mode="offset_rematch")

        if len(kept) + len(repaired) >= expected_count:
            duplicate_idxs = []
            for idx, match in enumerate(kept):
                left_dist = math.hypot(lp[0] - match["left_px"][0], lp[1] - match["left_px"][1])
                right_dist = math.hypot(rp[0] - match["right_px"][0], rp[1] - match["right_px"][1])
                if left_dist < 35.0 or right_dist < 35.0:
                    duplicate_idxs.append(idx)
            if not duplicate_idxs:
                continue
            replace_idx = max(
                duplicate_idxs,
                key=lambda idx: (
                    _offset_error(kept[idx]),
                    -_view_count(kept[idx]),
                    -float(kept[idx].get("score", 0.0)),
                ),
            )
            if _offset_error(entry) + 3.0 >= _offset_error(kept[replace_idx]):
                continue
            trimmed.append(kept.pop(replace_idx))

        repaired.append(entry)
        used_left.add(lp)
        used_right.add(rp)
        if len(kept) + len(repaired) >= expected_count and len(kept) >= expected_count:
            break

    kept.extend(repaired)
    right_by_pt = {_det_point(d): d for d in right_detections}
    matched_left = {tuple(m["left_px"]) for m in kept}

    # If a matched pair uses the right detection for a nearby duplicate left
    # point, swap in the unmatched left point when it fits the survey offset
    # better. This catches low-light duplicate boxes that steal a real target.
    for left_det in left_detections:
        lp = _det_point(left_det)
        if lp in matched_left or lp in used_left:
            continue

        best_replacement = None
        for idx, match in enumerate(kept):
            right_det = right_by_pt.get(tuple(match["right_px"]))
            if right_det is None:
                continue
            left_dist = math.hypot(lp[0] - match["left_px"][0], lp[1] - match["left_px"][1])
            pred = (lp[0] + med_dx, lp[1] + med_dy)
            right_dist = math.hypot(match["right_px"][0] - pred[0], match["right_px"][1] - pred[1])
            if left_dist >= 35.0 and right_dist >= 20.0:
                continue
            entry = _match_entry_from_detections(
                left_det,
                right_det,
                score=max(0.15, float(match.get("score", 0.0))),
                repair_mode="left_duplicate_swap",
            )
            improvement = _offset_error(match) - _offset_error(entry)
            if improvement < 3.0:
                continue
            if best_replacement is None or improvement > best_replacement[0]:
                best_replacement = (improvement, idx, entry)

        if best_replacement is None:
            continue
        _, replace_idx, entry = best_replacement
        trimmed.append(kept[replace_idx])
        kept[replace_idx] = entry
        repaired.append(entry)
        matched_left = {tuple(m["left_px"]) for m in kept}

    left_by_pt = {_det_point(d): d for d in left_detections}
    matched_right = {tuple(m["right_px"]) for m in kept}
    for right_det in right_detections:
        rp = _det_point(right_det)
        if rp in matched_right or rp in used_right:
            continue

        best_replacement = None
        for idx, match in enumerate(kept):
            left_det = left_by_pt.get(tuple(match["left_px"]))
            if left_det is None:
                continue
            right_dist = math.hypot(rp[0] - match["right_px"][0], rp[1] - match["right_px"][1])
            pred = (match["left_px"][0] + med_dx, match["left_px"][1] + med_dy)
            pred_dist = math.hypot(rp[0] - pred[0], rp[1] - pred[1])
            if right_dist >= 35.0 and pred_dist >= 20.0:
                continue
            entry = _match_entry_from_detections(
                left_det,
                right_det,
                score=max(0.15, float(match.get("score", 0.0))),
                repair_mode="right_duplicate_swap",
            )
            improvement = _offset_error(match) - _offset_error(entry)
            if improvement < 3.0:
                continue
            if best_replacement is None or improvement > best_replacement[0]:
                best_replacement = (improvement, idx, entry)

        if best_replacement is None:
            continue
        _, replace_idx, entry = best_replacement
        trimmed.append(kept[replace_idx])
        kept[replace_idx] = entry
        repaired.append(entry)
        matched_right = {tuple(m["right_px"]) for m in kept}

    matched_left = {tuple(m["left_px"]) for m in kept}
    matched_right = {tuple(m["right_px"]) for m in kept}

    # If the right detector still missed a plant, seed a coarse target from a
    # strong unmatched left detection. Fine-align Re-ID will reacquire it locally.
    synthetic = []
    if len(kept) < expected_count:
        remaining_left = [d for d in left_detections if _det_point(d) not in matched_left]

        def _nearest_existing_left(pt):
            if not matched_left:
                return float("inf")
            return min(math.hypot(pt[0] - mp[0], pt[1] - mp[1]) for mp in matched_left)

        def _nearest_existing_right(pt):
            if not matched_right:
                return float("inf")
            return min(math.hypot(pt[0] - mp[0], pt[1] - mp[1]) for mp in matched_right)

        candidates = []
        for left_det in remaining_left:
            if not isinstance(left_det, dict):
                continue
            views = int(left_det.get("views", 1) or 1)
            lp = _det_point(left_det)
            if views < 3:
                continue
            if _nearest_existing_left(lp) < 35.0:
                continue
            pred_rp = (int(round(lp[0] + med_dx)), int(round(lp[1] + med_dy)))
            if _nearest_existing_right(pred_rp) < 28.0:
                continue
            candidates.append((-views, lp, pred_rp, left_det))

        for _, _, pred_rp, left_det in sorted(candidates):
            synthetic.append(
                _match_entry_from_detections(
                    left_det,
                    right_px=pred_rp,
                    score=0.22,
                    repair_mode="left_only_offset_fill",
                )
            )
            kept.append(synthetic[-1])
            matched_left.add(tuple(synthetic[-1]["left_px"]))
            matched_right.add(tuple(synthetic[-1]["right_px"]))
            if len(kept) >= expected_count:
                break

    def _remove_worse_duplicate(i, j):
        # For duplicate-looking pairs, stereo-offset consistency is more
        # trustworthy than raw match score under this low-light survey noise.
        return max(
            (i, j),
            key=lambda idx: (
                _offset_error(kept[idx]),
                -_view_count(kept[idx]),
                -float(kept[idx].get("score", 0.0)),
            ),
        )

    while len(kept) > expected_count:
        duplicate_candidates = []
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                li = kept[i]["left_px"]
                lj = kept[j]["left_px"]
                ri = kept[i]["right_px"]
                rj = kept[j]["right_px"]
                left_dist = math.hypot(li[0] - lj[0], li[1] - lj[1])
                right_dist = math.hypot(ri[0] - rj[0], ri[1] - rj[1])
                if left_dist < 35.0 or right_dist < 35.0:
                    duplicate_candidates.append((
                        min(left_dist, right_dist),
                        _remove_worse_duplicate(i, j),
                    ))

        if duplicate_candidates:
            _, worst_idx = min(duplicate_candidates, key=lambda item: item[0])
        else:
            worst_idx = max(
                range(len(kept)),
                key=lambda i: (
                    _offset_error(kept[i]),
                    -_view_count(kept[i]),
                    -float(kept[i].get("score", 0.0)),
                ),
            )
        trimmed.append(kept.pop(worst_idx))

    if dropped or repaired or synthetic or trimmed:
        print(
            "[SURVEY MATCH REPAIR] "
            f"offset dx={med_dx:.1f}px dy={med_dy:.1f}px; "
            f"dropped={len(dropped)} rematched={len(repaired)} synthetic={len(synthetic)} "
            f"trimmed={len(trimmed)} "
            f"total={len(matched_targets)}->{len(kept)}"
        )
        if synthetic:
            for m in synthetic:
                print(
                    f"[SURVEY MATCH REPAIR] synthetic right L={m['left_px']} "
                    f"R={m['right_px']} cls={m.get('left_cls')}"
                )
        if trimmed:
            for m in trimmed:
                print(
                    f"[SURVEY MATCH REPAIR] trimmed L={m['left_px']} R={m['right_px']} "
                    f"score={float(m.get('score', 0.0)):.3f} "
                    f"offset_err={_offset_error(m):.1f}px"
                )

    kept.sort(key=lambda m: (m["left_px"][0], m["left_px"][1]))
    return kept


def _append_manifest_target(manifest, target_id, src, coarse_xy, status, final_xy=None, reid_protocol=None):
    manifest["targets"].append({
        "target_id": target_id,
        "class_id": src.get("left_cls", src.get("right_cls")),
        "survey_pixel_left": src.get("left_px"),
        "survey_pixel_right": src.get("right_px"),
        "coarse_triangulated_mm": [coarse_xy[0], coarse_xy[1]],
        "fine_align_final_mm": final_xy,
        "reid_protocol": reid_protocol,
        "status": status,
    })


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
            "id": hit.get("target_id", idx),
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


def _auto_render(recording_dir):
    """Spawn render_trial_video.py on recording_dir, then optionally delete raw frames."""
    if recording_dir is None or not recording_dir.exists():
        print("[AUTO_RENDER] No recording directory found; skipping render.")
        return
    renderer = Path(__file__).resolve().parent / "dev_tools" / "render_trial_video.py"
    cmd = [sys.executable, str(renderer), str(recording_dir)]
    print(f"[AUTO_RENDER] Rendering {recording_dir.name} ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[AUTO_RENDER] Renderer exited with code {result.returncode}; raw frames kept.")
        return
    if AUTO_RENDER_DELETE_RAW:
        for sub in ("left", "right"):
            d = recording_dir / sub
            if d.is_dir():
                shutil.rmtree(d)
                print(f"[AUTO_RENDER] Deleted raw frames: {d}")


def _camera_env_metadata():
    left = CAMERA_SETTINGS.get("left") if isinstance(CAMERA_SETTINGS, dict) else None
    right = CAMERA_SETTINGS.get("right") if isinstance(CAMERA_SETTINGS, dict) else None
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}
    return {
        "camera_left_auto_exposure": left.get("auto_exposure"),
        "camera_left_exposure": left.get("exposure"),
        "camera_left_gain": left.get("gain"),
        "camera_left_auto_wb": left.get("auto_wb"),
        "camera_left_white_balance": left.get("white_balance"),
        "camera_right_auto_exposure": right.get("auto_exposure"),
        "camera_right_exposure": right.get("exposure"),
        "camera_right_gain": right.get("gain"),
        "camera_right_auto_wb": right.get("auto_wb"),
        "camera_right_white_balance": right.get("white_balance"),
    }


def _run_variable_metadata(total_cycles):
    return {
        "pipeline_cycles_requested": int(total_cycles),
        "ai_confidence": float(AI_CONFIDENCE),
        "survey_burst_count_cfg": int(SURVEY_BURST_COUNT),
        "survey_min_hits_cfg": int(SURVEY_MIN_HITS),
        "survey_cluster_radius_px_cfg": float(SURVEY_CLUSTER_RADIUS_PX),
        "survey_crop_mode_cfg": SURVEY_CROP_MODE,
        "survey_crop_half_x_px_cfg": SURVEY_CROP_HALF_X_PX,
        "survey_crop_half_y_px_cfg": SURVEY_CROP_HALF_Y_PX,
        "survey_yolo_imgsz_cfg": SURVEY_YOLO_IMGSZ,
        "fine_align_reid_burst_count_cfg": int(FINE_ALIGN_REID_BURST_COUNT),
        "fine_align_reid_crop_half_px_cfg": int(FINE_ALIGN_REID_CROP_HALF_PX),
        "fine_align_max_time_sec_cfg": float(FINE_ALIGN_MAX_TIME_SEC),
        "fine_align_settle_frames_cfg": int(FINE_ALIGN_SETTLE_FRAMES),
        "fine_align_deadzone_px_cfg": float(FINE_ALIGN_DEADZONE_PX),
        "fine_align_snap_enabled_cfg": bool(FINE_ALIGN_ENABLE_SNAP),
        "yolo_warmup_enabled_cfg": bool(YOLO_WARMUP),
        "yolo_warmup_imgsz_cfg": YOLO_WARMUP_IMGSZ,
        "yolo_warmup_iters_cfg": int(YOLO_WARMUP_ITERS),
        **_camera_env_metadata(),
    }


def main():
    gantry = None
    cameras = None
    state = "INIT"
    detector = None
    coarse_mover = None
    logger = None
    model_load_time_s = 0.0
    warmup_info = {}
    total_cycles = max(1, int(RUN_PIPELINE_CYCLES))
    cycle_index = 1
    global_target_id = 1
    run_started = False

    left_detections = []
    right_detections = []
    left_points = []
    right_points = []
    matched_targets = []
    target_queue = []
    actual_hits = []
    cycle_hits = []

    trial_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "trial_timestamp": trial_timestamp,
        "pipeline_cycles_requested": total_cycles,
        "survey_position_mm": [SURVEY_POS_X, SURVEY_POS_Y],
        "survey_target_policy": {
            "detect_all_classes": bool(SURVEY_DETECT_ALL_CLASSES),
            "detect_class_ids": SURVEY_DETECT_CLASS_IDS,
            "can_target_classes": SURVEY_CAN_TARGET_CLASSES,
            "cant_target_classes": SURVEY_CANT_TARGET_CLASSES,
            "cant_target_priority": True,
            "legacy_survey_target_classes": SURVEY_TARGET_CLASSES,
        },
        "cycles": [],
        "targets": [],
    }

    try:
        while state != "DONE":
            if state == "INIT":
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
                cycle_hits = []
                if not run_started:
                    cameras.open()
                    run_started = True
                _update_recording_context(cameras, "HOME", gantry, target_queue, None, None, cycle_hits, logger)
                if HOMING:
                    gantry.home()
                if cycle_index == 1 and DETECTOR_MODE == "ai" and hasattr(detector, "warmup"):
                    warmup_info = detector.warmup()
                if logger is not None and cycle_index == 1:
                    logger.start_run(run_metadata={
                        "model_load_time_s": model_load_time_s,
                        **warmup_info,
                        **_run_variable_metadata(total_cycles),
                    })
                    _attach_recording_metrics(logger, cameras)
                    _update_recording_context(cameras, "HOME", gantry, target_queue, None, None, cycle_hits, logger)
                state = "SURVEY"

            elif state == "SURVEY":
                _update_recording_context(cameras, "SURVEY", gantry, target_queue, None, None, cycle_hits, logger)
                gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
                print_global_survey_ready(SURVEY_POS_X, SURVEY_POS_Y)
                state = "SURVEY_CONFIRM"

            elif state == "SURVEY_CONFIRM":
                state = "DETECT"

            elif state == "DETECT":
                _update_recording_context(cameras, "DETECT", gantry, target_queue, None, None, cycle_hits, logger)
                _flush_camera_buffer(cameras, n=8)
                survey_burst_count = _resolve_burst_count(SURVEY_BURST_COUNT)
                survey_point_mode = _resolve_point_mode(SURVEY_POINT_MODE)
                if DETECTOR_MODE == "ai" and SURVEY_DETECT_ALL_CLASSES:
                    survey_classes_for_detection = SURVEY_DETECT_CLASS_IDS
                else:
                    survey_classes_for_detection = SURVEY_TARGET_CLASSES
                print(f"[CV CONFIG] SURVEY burst_count={survey_burst_count} point_mode={survey_point_mode}")
                print(
                    f"[CV CONFIG] SURVEY classes_detect={survey_classes_for_detection} "
                    f"can_target={SURVEY_CAN_TARGET_CLASSES} cant_target={SURVEY_CANT_TARGET_CLASSES}"
                )

                if logger is not None:
                    logger.start_section("survey")
                left_detections, right_detections = coarse_mover.detect_stable_points(
                    cameras,
                    detector,
                    detector_mode=DETECTOR_MODE,
                    burst_count=survey_burst_count,
                    min_hits=SURVEY_MIN_HITS,
                    cluster_radius_px=SURVEY_CLUSTER_RADIUS_PX,
                    survey_classes=survey_classes_for_detection,
                    point_mode=survey_point_mode,
                )
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
                _update_recording_context(cameras, "MATCH", gantry, target_queue, None, None, cycle_hits, logger)

                can_target_set = _normalise_class_selector(SURVEY_CAN_TARGET_CLASSES)
                cant_target_set = _normalise_class_selector(SURVEY_CANT_TARGET_CLASSES)

                left_before = len(left_detections)
                right_before = len(right_detections)
                left_classes_before = _class_counts(left_detections)
                right_classes_before = _class_counts(right_detections)
                left_detections, left_stats = _filter_survey_detections_by_class(
                    left_detections, can_target_set, cant_target_set
                )
                right_detections, right_stats = _filter_survey_detections_by_class(
                    right_detections, can_target_set, cant_target_set
                )
                left_points = [d["point"] if isinstance(d, dict) else d for d in left_detections]
                right_points = [d["point"] if isinstance(d, dict) else d for d in right_detections]

                print(
                    f"[SURVEY FILTER] LEFT {left_before}->{left_stats['kept']} "
                    f"(drop cant={left_stats['dropped_by_cant']} can={left_stats['dropped_by_can']})"
                )
                print(
                    f"[SURVEY FILTER] RIGHT {right_before}->{right_stats['kept']} "
                    f"(drop cant={right_stats['dropped_by_cant']} can={right_stats['dropped_by_can']})"
                )
                print(f"[SURVEY FILTER] LEFT pre-filter classes {left_classes_before}")
                print(f"[SURVEY FILTER] RIGHT pre-filter classes {right_classes_before}")
                print(f"[SURVEY FILTER] LEFT classes {_class_counts(left_detections)}")
                print(f"[SURVEY FILTER] RIGHT classes {_class_counts(right_detections)}")

                if logger is not None:
                    logger.start_section("stereo_matching")
                matched_targets, _, _ = match_points(
                    left_detections,
                    right_detections,
                    verbose=True,
                )
                matched_targets = _repair_survey_matches(
                    matched_targets,
                    left_detections,
                    right_detections,
                    expected_count=EXPECTED_WEED_COUNT,
                )
                if logger is not None:
                    logger.end_section("stereo_matching")
                    logger.run["num_stereo_matches"] = len(matched_targets)
                print(f"[SURVEY MATCH] matched classes {_matched_class_counts(matched_targets)}")

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
                _update_recording_context(cameras, "PLAN", gantry, target_queue, None, None, cycle_hits, logger)
                if logger is not None:
                    logger.start_section("triangulation")
                coarse_mover.fit_epipolar(matched_targets)
                absolute_targets = coarse_mover.solve_all_from_pose(
                    matched_targets,
                    ref_x=SURVEY_POS_X,
                    ref_y=SURVEY_POS_Y,
                )
                if logger is not None:
                    logger.end_section("triangulation")

                if logger is not None:
                    logger.start_section("planning")
                target_queue = plan_targets(
                    absolute_targets,
                    start_xy=gantry.get_estimated_xy(),
                )
                _update_recording_context(cameras, "PLAN", gantry, target_queue, None, None, cycle_hits, logger)
                if logger is not None:
                    logger.end_section("planning")
                    logger.run["num_targets_planned"] = len(target_queue)
                    logger.compute_path_metrics(target_queue, start_xy=gantry.get_estimated_xy())

                coarse_mover.all_planned_targets = target_queue

                coarse_mover.save_workspace_targets(
                    target_queue,
                    filename=f"predicted_workspace_targets_cycle_{cycle_index:02d}.json",
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
                cycle_start_hits = len(actual_hits)

                for cycle_target_idx, solved in enumerate(target_queue, start=1):
                    i = global_target_id
                    global_target_id += 1
                    tx, ty = solved["target_xy_mm"]
                    src = solved.get("source_target", {})
                    travel_dist = round(math.hypot(tx - prev_xy[0], ty - prev_xy[1]), 2)
                    _update_recording_context(cameras, "TARGET", gantry, target_queue, i, solved, cycle_hits, logger)

                    if logger is not None:
                        logger.start_target(i, {
                            "x_target_mm": tx,
                            "y_target_mm": ty,
                            "x_commanded_mm": tx,
                            "y_commanded_mm": ty,
                            "travel_distance_mm": travel_dist,
                            "class_id": src.get("left_cls"),
                            "confidence": src.get("conf"),
                            "notes": f"cycle={cycle_index},cycle_target={cycle_target_idx}/{total}",
                        })

                    if not is_in_workspace(tx, ty):
                        print_skip_target(i, total, solved, f"Outside workspace bounds ({tx:.1f}, {ty:.1f}) mm.")
                        if logger is not None:
                            logger.end_target(i, {"status": "skipped_out_of_bounds"})
                        _append_manifest_target(
                            manifest,
                            target_id=i,
                            src=src,
                            coarse_xy=(tx, ty),
                            status="skipped_out_of_bounds",
                        )
                        continue

                    if coarse_mover.is_duplicate_of_actual(
                        solved["target_xy_mm"],
                        cycle_hits,
                        tol_mm=8.0,
                    ):
                        print_skip_target(i, total, solved, "Already covered by a previous PD lock.")
                        if logger is not None:
                            logger.end_target(i, {"status": "skipped_duplicate"})
                        _append_manifest_target(
                            manifest,
                            target_id=i,
                            src=src,
                            coarse_xy=(tx, ty),
                            status="skipped_duplicate",
                        )
                        continue

                    show_current_target(i, total, solved)
                    cameras.clear_overlay()
                    cameras.set_recording_status([
                        f"Cycle {cycle_index}/{total_cycles} Target {cycle_target_idx}/{total}",
                        f"Coarse: ({tx:.1f}, {ty:.1f}) mm",
                        "Moving to target...",
                    ])

                    if logger is not None:
                        logger.start_target_section(i, "travel")
                    _update_recording_context(cameras, "TRAVEL", gantry, target_queue, i, solved, cycle_hits, logger)
                    moved = coarse_mover.move_to_absolute_target(gantry, solved)
                    if logger is not None:
                        logger.end_target_section(i, "travel")
                    _update_recording_context(cameras, "POST_TRAVEL", gantry, target_queue, i, solved, cycle_hits, logger)

                    if not moved:
                        if logger is not None:
                            logger.end_target(i, {"status": "skipped_move_failed"})
                        _append_manifest_target(
                            manifest,
                            target_id=i,
                            src=src,
                            coarse_xy=(tx, ty),
                            status="skipped_move_failed",
                        )
                        continue

                    prev_xy = (tx, ty)

                    if TRIANGULATION_ONLY_MODE:
                        gantry.sync_estimate_to_machine()
                        final_xy = gantry.get_estimated_xy()
                        actual_entry = {
                            "target_id": int(i),
                            "planned_xy_mm": [float(tx), float(ty)],
                            "selected_local_xy_mm": [float(tx), float(ty)],
                            "final_xy_mm": [float(final_xy[0]), float(final_xy[1])],
                            "left_px": src.get("left_px"),
                            "right_px": src.get("right_px"),
                            "score": float(src.get("score", 1.0)),
                        }
                        cycle_hits.append(actual_entry)
                        actual_hits.append(actual_entry)
                        coarse_mover.append_actual_target(
                            solved, solved, final_xy,
                            filename="actual_pd_targets.json",
                        )
                        print_target_result(i, total, solved, actual_entry)
                        _append_manifest_target(
                            manifest,
                            target_id=i,
                            src=src,
                            coarse_xy=(tx, ty),
                            final_xy=list(final_xy),
                            reid_protocol=None,
                            status="triangulation_only",
                        )
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
                        f"Cycle {cycle_index}/{total_cycles} Target {cycle_target_idx}/{total}",
                        f"Coarse: ({tx:.1f}, {ty:.1f}) mm",
                        "Fine aligning...",
                    ])

                    if logger is not None:
                        logger.start_target_section(i, "pd")
                    _update_recording_context(cameras, "FINE_ALIGN", gantry, target_queue, i, solved, cycle_hits, logger)
                    aligned, actual_entry = fine_align_target(
                        gantry,
                        cameras,
                        detector,
                        coarse_mover,
                        solved,
                        cycle_hits,
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
                        if actual_entry is not None:
                            actual_entry["target_id"] = int(i)
                        cycle_hits.append(actual_entry)
                        actual_hits.append(actual_entry)
                        _update_recording_context(cameras, "LOCKED", gantry, target_queue, i, solved, cycle_hits, logger)
                        print_target_result(i, total, solved, actual_entry)

                        if logger is not None:
                            logger.start_target_section(i, "fire")
                        _update_recording_context(cameras, "FIRE", gantry, target_queue, i, solved, cycle_hits, logger)
                        fire_target(gantry, solved, cameras=cameras)
                        if logger is not None:
                            logger.end_target_section(i, "fire")
                        _update_recording_context(cameras, "FIRED", gantry, target_queue, i, solved, cycle_hits, logger)

                        _append_manifest_target(
                            manifest,
                            target_id=i,
                            src=src,
                            coarse_xy=(tx, ty),
                            final_xy=actual_entry.get("final_xy_mm"),
                            reid_protocol=actual_entry.get("reid_protocol"),
                            status="locked_fired",
                        )
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
                        _update_recording_context(cameras, "FAILED_FINE_ALIGN", gantry, target_queue, i, solved, cycle_hits, logger)
                        cameras.set_recording_status([
                            f"Cycle {cycle_index}/{total_cycles} Target {cycle_target_idx}/{total}",
                            f"Coarse: ({tx:.1f}, {ty:.1f}) mm",
                            "FAILED: fine align",
                        ])
                        print_skip_target(i, total, solved, "Fine align failed")
                        _append_manifest_target(
                            manifest,
                            target_id=i,
                            src=src,
                            coarse_xy=(tx, ty),
                            status="failed_fine_align",
                        )
                        if logger is not None:
                            logger.end_target(i, {
                                "pd_converged": False,
                                "fired": False,
                                "status": "failed_fine_align",
                            })

                close_fine_align_window()
                clear_current_target_line()
                cycle_fired = len(actual_hits) - cycle_start_hits
                cycle_summary = {
                    "cycle_index": int(cycle_index),
                    "targets_planned": int(total),
                    "targets_fired": int(cycle_fired),
                }
                manifest["cycles"].append(cycle_summary)
                print(
                    f"\n  Cycle {cycle_index}/{total_cycles} complete: "
                    f"planned={total} fired={cycle_fired}"
                )

                if cycle_index < total_cycles:
                    cycle_index += 1
                    left_detections = []
                    right_detections = []
                    left_points = []
                    right_points = []
                    matched_targets = []
                    target_queue = []
                    state = "HOME"
                else:
                    print("\n  All cycles complete.")
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

    if AUTO_RENDER_TRIAL and RECORD_TRIAL:
        _auto_render(recording_dir)


if __name__ == "__main__":
    main()
