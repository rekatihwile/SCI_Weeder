import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path


def _bbox_area(box):
    if not box or len(box) < 4:
        return None
    return round(max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1])), 3)


class ExperimentLogger:
    def __init__(self, output_dir="experiments/metrics", config=None):
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.run = {}
        self.targets = {}
        self.section_starts = {}
        self.target_section_starts = {}
        self._run_start_perf = None

    def start_run(self, run_metadata=None):
        self._run_start_perf = time.perf_counter()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        w = self.config.get("workspace_width_mm")
        h = self.config.get("workspace_height_mm")
        ew = self.config.get("expected_weed_count") or 0
        ek = self.config.get("expected_kale_count") or 0
        self.run = {
            "run_id": run_id,
            "trial_id": self.config.get("trial_id", ""),
            "trial_type": self.config.get("trial_type", ""),
            "layout_type": self.config.get("layout_type", ""),
            "workspace_width_mm": w,
            "workspace_height_mm": h,
            "workspace_area_mm2": float(w) * float(h) if w and h else None,
            "expected_weed_count": ew,
            "expected_kale_count": ek,
            "expected_total_plants": ew + ek,
            "num_detections_left": None,
            "num_detections_right": None,
            "num_stereo_matches": None,
            "num_targets_planned": None,
            "num_targets_attempted": 0,
            "num_targets_fired": 0,
            "num_targets_hit": None,
            "num_weeds_missed": None,
            "num_false_positives": None,
            "survey_time_s": None,
            "model_load_time_s": None,
            "warmup_time_s": None,
            "survey_camera_read_time_s": None,
            "survey_yolo_time_s": None,
            "survey_grouping_time_s": None,
            "detection_time_s": None,
            "stereo_matching_time_s": None,
            "triangulation_time_s": None,
            "planning_time_s": None,
            "total_travel_time_s": 0.0,
            "total_pd_time_s": 0.0,
            "total_fine_align_reid_yolo_time_s": 0.0,
            "total_fine_align_reid_time_s": 0.0,
            "total_fine_align_pd_lk_time_s": 0.0,
            "total_final_snap_time_s": 0.0,
            "total_fire_time_s": 0.0,
            "recording_frame_save_time_s": 0.0,
            "recording_frames_saved": 0,
            "recording_frames_dropped": 0,
            "total_state_overhead_time_s": None,
            "total_run_time_s": None,
            "planned_path_length_mm": None,
            "actual_path_length_mm": None,
            "experiment_grid_enabled": None,
            "trial_filter_enabled": None,
            "trial_filter_mode": None,
            "random_seed": None,
            "grid_rows": None,
            "grid_cols": None,
            "grid_x_min_mm": None,
            "grid_x_max_mm": None,
            "grid_y_min_mm": None,
            "grid_y_max_mm": None,
            "cell_width_mm": None,
            "cell_height_mm": None,
            "survey_origin_x_mm": None,
            "survey_origin_y_mm": None,
            "survey_origin_cell_id": None,
            "survey_origin_cell_row": None,
            "survey_origin_cell_col": None,
            "surveyed_target_count": None,
            "requested_active_cell_count": None,
            "selected_target_count": None,
            "rejected_target_count": None,
            "occupied_cell_count": None,
            "occupied_cell_ids": None,
            "selected_cell_ids": None,
            "rejected_cell_ids": None,
            "mean_selected_radius_mm": None,
            "max_selected_radius_mm": None,
            "mean_selected_distance_from_cell_center_mm": None,
            "selected_spread_mm": None,
            "selected_convex_hull_area_mm2": None,
            "total_treatment_time_s": None,
            "area_rate_mm2_per_s": None,
            "area_rate_m2_per_min": None,
            "weeds_per_min": None,
            "run_status": "running",
            "notes": self.config.get("notes", ""),
        }
        if run_metadata:
            self.run.update(run_metadata)

    def end_run(self, status="complete"):
        if not self.run:
            return
        self.run["run_status"] = status
        if self._run_start_perf is not None:
            elapsed = round(time.perf_counter() - self._run_start_perf, 3)
            self.run["total_run_time_s"] = elapsed
            area = self.run.get("workspace_area_mm2")
            if area and elapsed > 0:
                rate = area / elapsed
                self.run["area_rate_mm2_per_s"] = round(rate, 3)
                self.run["area_rate_m2_per_min"] = round(rate * 60 / 1e6, 4)
            fired = self.run.get("num_targets_fired") or 0
            if elapsed > 0:
                self.run["weeds_per_min"] = round(fired * 60 / elapsed, 3)
        self.compute_treatment_totals()

    def start_section(self, name):
        self.section_starts[name] = time.perf_counter()

    def end_section(self, name):
        if name not in self.section_starts:
            return
        dt = round(time.perf_counter() - self.section_starts.pop(name), 3)
        self.run[f"{name}_time_s"] = dt

    def start_target(self, target_id, target_data=None):
        entry = {
            "run_id": self.run.get("run_id"),
            "target_id": target_id,
            "detection_id": target_id,
            "class_name": None,
            "class_id": None,
            "confidence": None,
            "weed_bbox_area_px2": None,
            "weed_mask_area_px2": None,
            "x_target_mm": None,
            "y_target_mm": None,
            "z_target_mm": None,
            "cell_id": None,
            "cell_row": None,
            "cell_col": None,
            "cell_center_x_mm": None,
            "cell_center_y_mm": None,
            "distance_from_cell_center_mm": None,
            "radius_from_survey_mm": None,
            "angle_from_survey_deg": None,
            "ring_index": None,
            "axis_label": None,
            "quadrant_label": None,
            "was_selected_by_trial_filter": None,
            "selection_reason": None,
            "x_commanded_mm": None,
            "y_commanded_mm": None,
            "x_final_mm": None,
            "y_final_mm": None,
            "position_error_mm": None,
            "travel_distance_mm": None,
            "travel_time_s": None,
            "reid_time_s": None,
            "fine_align_time_s": None,
            "pd_time_s": None,
            "fire_time_s": None,
            "per_target_total_time_s": None,
            "per_target_treatment_time_s": None,
            "pd_iterations": None,
            "pd_converged": None,
            "fired": False,
            "hit_success": None,
            "false_positive": None,
            "missed": None,
            "status": "pending",
            "notes": "",
        }
        if target_data:
            entry.update(target_data)
        if target_id in self.targets:
            existing = self.targets[target_id]
            existing.update({k: v for k, v in entry.items() if v is not None})
            entry = existing
        self.targets[target_id] = entry
        self.target_section_starts[target_id] = {"_start": time.perf_counter()}
        self.run["num_targets_attempted"] = (self.run.get("num_targets_attempted") or 0) + 1

    def register_survey_targets(self, targets):
        for idx, target in enumerate(targets or [], start=1):
            target_id = target.get("target_id", idx)
            xy = target.get("target_xy_mm") or (None, None)
            src = target.get("source_target", {})
            entry = {
                "run_id": self.run.get("run_id"),
                "target_id": target_id,
                "detection_id": target_id,
                "class_name": src.get("class_name"),
                "class_id": src.get("left_cls", src.get("right_cls")),
                "confidence": src.get("left_conf", src.get("right_conf", src.get("conf"))),
                "weed_bbox_area_px2": _bbox_area(src.get("left_box", src.get("right_box"))),
                "weed_mask_area_px2": src.get("weed_mask_area_px2"),
                "x_target_mm": xy[0],
                "y_target_mm": xy[1],
                "z_target_mm": target.get("z_target_mm"),
                "status": (
                    "selected_by_trial_filter"
                    if target.get("was_selected_by_trial_filter", True)
                    else "rejected_by_trial_filter"
                ),
            }
            for key in (
                "cell_id", "cell_row", "cell_col",
                "cell_center_x_mm", "cell_center_y_mm",
                "distance_from_cell_center_mm",
                "radius_from_survey_mm", "angle_from_survey_deg",
                "ring_index", "axis_label", "quadrant_label",
                "was_selected_by_trial_filter", "selection_reason",
            ):
                entry[key] = target.get(key)
            if target_id in self.targets:
                self.targets[target_id].update(entry)
            else:
                self.targets[target_id] = entry

    def update_target(self, target_id, target_data):
        if target_id in self.targets:
            self.targets[target_id].update(target_data)
            if "fine_align_reid_total_time_s" in target_data:
                self.targets[target_id]["reid_time_s"] = target_data.get("fine_align_reid_total_time_s")
            if "fine_align_pd_lk_time_s" in target_data:
                self.targets[target_id]["fine_align_time_s"] = target_data.get("fine_align_pd_lk_time_s")

    def log_reid_debug(self, target_id, reid_debug):
        if target_id not in self.targets or not isinstance(reid_debug, dict):
            return

        entry = self.targets[target_id]
        rejects = dict(reid_debug.get("reid_filter_rejects") or {})
        chosen = dict(reid_debug.get("reid_chosen_detail") or {})
        timing = dict(reid_debug.get("reid_timing") or {})

        entry["reid_debug"] = dict(reid_debug)
        entry.update({
            "reid_ok": bool(reid_debug.get("reid_ok", False)),
            "reid_error": reid_debug.get("reid_error"),
            "reid_filter_mode": reid_debug.get("reid_filter_mode"),
            "reid_left_count": int(reid_debug.get("reid_left_count", 0) or 0),
            "reid_right_count": int(reid_debug.get("reid_right_count", 0) or 0),
            "reid_match_count": int(reid_debug.get("reid_match_count", 0) or 0),
            "reid_expected_cls": reid_debug.get("reid_expected_cls"),
            "reid_point_mode": reid_debug.get("reid_point_mode"),
            "reid_burst_count": reid_debug.get("reid_burst_count"),
            "reid_chosen": bool(reid_debug.get("reid_chosen", False)),
            "reid_chosen_pd_err_px": chosen.get("pd_err_px"),
            "reid_chosen_tri_dist_mm": chosen.get("tri_dist_mm"),
            "reid_chosen_geo_score": chosen.get("geo_score"),
            "reid_reject_crop": int(rejects.get("crop", 0) or 0),
            "reid_reject_duplicate": int(rejects.get("duplicate", 0) or 0),
            "reid_reject_pd": int(rejects.get("pd", 0) or 0),
            "reid_reject_epipolar": int(rejects.get("epipolar", 0) or 0),
            "reid_reject_max_tri_dist": int(rejects.get("max_tri_dist", 0) or 0),
            "reid_timing_total_s": timing.get("reid_total_time_s", timing.get("total_s")),
            "reid_timing_yolo_s": timing.get("reid_yolo_time_s"),
            "reid_timing_match_s": timing.get("reid_matching_time_s", timing.get("match_s")),
        })

    def start_target_section(self, target_id, section_name):
        if target_id not in self.target_section_starts:
            self.target_section_starts[target_id] = {}
        self.target_section_starts[target_id][section_name] = time.perf_counter()

    def end_target_section(self, target_id, section_name):
        ts = self.target_section_starts.get(target_id, {})
        if section_name not in ts:
            return
        dt = round(time.perf_counter() - ts.pop(section_name), 3)
        if target_id in self.targets:
            self.targets[target_id][f"{section_name}_time_s"] = dt
        run_key = f"total_{section_name}_time_s"
        if run_key in self.run:
            self.run[run_key] = round((self.run[run_key] or 0.0) + dt, 3)

    def end_target(self, target_id, final_data=None):
        if target_id not in self.targets:
            return
        if final_data:
            self.targets[target_id].update(final_data)
        start = self.target_section_starts.get(target_id, {}).get("_start")
        if start is not None:
            self.targets[target_id]["per_target_total_time_s"] = round(
                time.perf_counter() - start, 3
            )
        t = self.targets[target_id]
        treatment_parts = [
            t.get("travel_time_s"),
            t.get("fine_align_reid_total_time_s") or t.get("reid_time_s"),
            t.get("fine_align_pd_lk_time_s") or t.get("fine_align_time_s"),
            t.get("fire_time_s"),
        ]
        if any(v is not None for v in treatment_parts):
            t["per_target_treatment_time_s"] = round(sum(float(v or 0.0) for v in treatment_parts), 3)
        if self.targets[target_id].get("fired"):
            self.run["num_targets_fired"] = (self.run.get("num_targets_fired") or 0) + 1

    def compute_treatment_totals(self):
        if not self.run:
            return
        travel = self.run.get("total_travel_time_s") or 0.0
        reid = self.run.get("total_fine_align_reid_time_s") or 0.0
        pd_lk = self.run.get("total_fine_align_pd_lk_time_s") or 0.0
        fire = self.run.get("total_fire_time_s") or 0.0
        self.run["total_treatment_time_s"] = round(travel + reid + pd_lk + fire, 3)

    def compute_path_metrics(self, planned_targets, start_xy=None):
        if not planned_targets:
            return
        coords = [t["target_xy_mm"] for t in planned_targets]
        total = 0.0
        prev = start_xy if start_xy else coords[0]
        for xy in coords:
            total += math.hypot(xy[0] - prev[0], xy[1] - prev[1])
            prev = xy
        self.run["planned_path_length_mm"] = round(total, 2)

    def save_csvs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_path = self.output_dir / "run_summary.csv"
        tgt_path = self.output_dir / "target_summary.csv"

        run_fields = list(self.run.keys())
        run_exists = run_path.exists()
        with open(run_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_fields, extrasaction="ignore")
            if not run_exists:
                writer.writeheader()
            writer.writerow(self.run)

        tgt_fields = [
            "run_id", "target_id", "detection_id", "class_name", "class_id",
            "confidence", "weed_bbox_area_px2", "weed_mask_area_px2",
            "x_target_mm", "y_target_mm", "z_target_mm",
            "cell_id", "cell_row", "cell_col",
            "cell_center_x_mm", "cell_center_y_mm",
            "distance_from_cell_center_mm", "radius_from_survey_mm",
            "angle_from_survey_deg", "ring_index", "axis_label", "quadrant_label",
            "was_selected_by_trial_filter", "selection_reason",
            "x_commanded_mm", "y_commanded_mm", "x_final_mm", "y_final_mm",
            "position_error_mm", "travel_distance_mm", "travel_time_s",
            "reid_time_s", "fine_align_time_s",
            "pd_time_s", "fire_time_s", "per_target_total_time_s",
            "per_target_treatment_time_s",
            "pd_iterations", "pd_converged", "fired",
            "fine_align_reid_yolo_time_s", "fine_align_reid_total_time_s",
            "fine_align_pd_lk_time_s", "final_snap_time_s",
            "fine_align_snap_used", "fine_align_snap_move_px",
            "reid_ok", "reid_error", "reid_filter_mode",
            "reid_left_count", "reid_right_count", "reid_match_count",
            "reid_expected_cls", "reid_point_mode", "reid_burst_count",
            "reid_chosen", "reid_chosen_pd_err_px", "reid_chosen_tri_dist_mm", "reid_chosen_geo_score",
            "reid_reject_crop", "reid_reject_duplicate", "reid_reject_pd",
            "reid_reject_epipolar", "reid_reject_max_tri_dist",
            "reid_timing_total_s", "reid_timing_yolo_s", "reid_timing_match_s",
            "hit_success", "false_positive", "missed", "status", "notes",
        ]
        tgt_exists = tgt_path.exists()
        with open(tgt_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tgt_fields, extrasaction="ignore")
            if not tgt_exists:
                writer.writeheader()
            for t in self.targets.values():
                writer.writerow({k: t.get(k) for k in tgt_fields})

    def save_json(self):
        json_dir = self.output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        run_id = self.run.get("run_id", "unknown")
        path = json_dir / f"{run_id}.json"
        with open(path, "w") as f:
            json.dump(
                {"run": self.run, "targets": {str(k): v for k, v in self.targets.items()}},
                f, indent=2, default=str,
            )
        return path

    def print_summary(self):
        r = self.run
        run_id = r.get("run_id", "?")
        total  = r.get("total_run_time_s") or 0.0
        survey = r.get("survey_time_s") or 0.0
        model  = r.get("model_load_time_s") or 0.0
        warmup = r.get("warmup_time_s") or 0.0
        survey_read = r.get("survey_camera_read_time_s") or 0.0
        survey_yolo = r.get("survey_yolo_time_s") or 0.0
        survey_group = r.get("survey_grouping_time_s") or 0.0
        match  = r.get("stereo_matching_time_s") or 0.0
        tri    = r.get("triangulation_time_s") or 0.0
        plan   = r.get("planning_time_s") or 0.0
        travel = r.get("total_travel_time_s") or 0.0
        pd     = r.get("total_pd_time_s") or 0.0
        reid_yolo = r.get("total_fine_align_reid_yolo_time_s") or 0.0
        pd_lk = r.get("total_fine_align_pd_lk_time_s") or 0.0
        snap = r.get("total_final_snap_time_s") or 0.0
        fire   = r.get("total_fire_time_s") or 0.0
        rec_save = r.get("recording_frame_save_time_s") or 0.0
        rate   = r.get("area_rate_m2_per_min") or 0.0
        wpm    = r.get("weeds_per_min") or 0.0
        attempted = r.get("num_targets_attempted") or 0
        fired     = r.get("num_targets_fired") or 0

        print("\n=== EXPERIMENT METRICS SAVED ===")
        print(f"  Run ID:        {run_id}")
        print(f"  Status:        {r.get('run_status', '?')}")
        print(f"  Total time:    {total:.2f} s")
        print(f"  Model load:    {model:.2f} s")
        print(f"  Warmup:        {warmup:.2f} s")
        print(f"  Survey:        {survey:.2f} s  (read {survey_read:.2f}, YOLO {survey_yolo:.2f}, group {survey_group:.2f})")
        print(f"  Matching:      {match:.2f} s")
        print(f"  Triangulation: {tri:.2f} s")
        print(f"  Planning:      {plan:.2f} s")
        print(f"  Travel:        {travel:.2f} s")
        print(f"  PD align:      {pd:.2f} s  (Re-ID YOLO {reid_yolo:.2f}, PD/LK {pd_lk:.2f}, snap {snap:.2f})")
        print(f"  Fire:          {fire:.2f} s")
        print(f"  Recording:     save {rec_save:.2f} s, frames {r.get('recording_frames_saved') or 0}")
        print(f"  Area rate:     {rate:.4f} m²/min")
        print(f"  Weeds/min:     {wpm:.2f}")
        print(f"  Targets:       {fired}/{attempted} fired")
        print(f"  Files:")
        print(f"    {self.output_dir / 'run_summary.csv'}")
        print(f"    {self.output_dir / 'target_summary.csv'}")
        print(f"    {self.output_dir / 'json' / (run_id + '.json')}")
        print()
