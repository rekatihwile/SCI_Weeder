import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path


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
            "x_target_mm": None,
            "y_target_mm": None,
            "z_target_mm": None,
            "x_commanded_mm": None,
            "y_commanded_mm": None,
            "x_final_mm": None,
            "y_final_mm": None,
            "position_error_mm": None,
            "travel_distance_mm": None,
            "travel_time_s": None,
            "pd_time_s": None,
            "fire_time_s": None,
            "per_target_total_time_s": None,
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
        self.targets[target_id] = entry
        self.target_section_starts[target_id] = {"_start": time.perf_counter()}
        self.run["num_targets_attempted"] = (self.run.get("num_targets_attempted") or 0) + 1

    def update_target(self, target_id, target_data):
        if target_id in self.targets:
            self.targets[target_id].update(target_data)

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
        if self.targets[target_id].get("fired"):
            self.run["num_targets_fired"] = (self.run.get("num_targets_fired") or 0) + 1

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
            "confidence", "x_target_mm", "y_target_mm", "z_target_mm",
            "x_commanded_mm", "y_commanded_mm", "x_final_mm", "y_final_mm",
            "position_error_mm", "travel_distance_mm", "travel_time_s",
            "pd_time_s", "fire_time_s", "per_target_total_time_s",
            "pd_iterations", "pd_converged", "fired",
            "fine_align_reid_yolo_time_s", "fine_align_reid_total_time_s",
            "fine_align_pd_lk_time_s", "final_snap_time_s",
            "fine_align_snap_used", "fine_align_snap_move_px",
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
