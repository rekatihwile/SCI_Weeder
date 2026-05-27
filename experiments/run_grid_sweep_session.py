import argparse
import contextlib
import io
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as runtime_config
import pipeline.runtime as runtime_pipeline
from pipeline.runtime import close_runtime_session, run_runtime


JSON_DIR = ROOT / "experiments" / "metrics" / "json"

# Simple controls for experiment sessions.
# Example: NUM_WEEDS = [5, 6, 7, 8, 9, 10, 11, 12]
NUM_WEEDS = [14,15]
NUM_TRIALS = 5


def _parse_counts(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _should_skip(cell_count, repeat, start_count, start_repeat):
    if start_count is None:
        return False
    if cell_count < start_count:
        return True
    if cell_count == start_count and repeat < start_repeat:
        return True
    return False


def _valid_run_json(path, trial_id=None):
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    run = data.get("run") or {}
    if trial_id is not None and run.get("trial_id") != trial_id:
        return None
    return data


def _latest_metrics(before, trial_id=None):
    after = set(JSON_DIR.glob("*.json"))
    new_paths = sorted(after - before, key=lambda p: p.stat().st_mtime)
    for path in reversed(new_paths):
        if _valid_run_json(path, trial_id=trial_id) is not None:
            return path
    return None


def _read_summary(path):
    if path is None or not path.exists():
        return {}
    data = _valid_run_json(path)
    if data is None:
        return {}
    run = data.get("run") or {}
    return {
        "run_id": run.get("run_id"),
        "status": run.get("status", run.get("run_status")),
        "requested": run.get("requested_active_cell_count"),
        "eligible": run.get("eligible_target_count"),
        "selected": run.get("selected_target_count"),
        "rejected": run.get("rejected_target_count"),
        "planned_path_mm": run.get("planned_path_length_mm"),
        "total_s": run.get("total_run_time_s"),
        "cells": run.get("selected_cell_ids"),
    }


def _write_results(log_dir, sweep_id, results):
    payload = {
        "sweep_id": sweep_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_count": len(results),
        "results": results,
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)


def _load_existing_results(log_dir):
    path = log_dir / "summary.json"
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        with open(path, "r") as f:
            return list((json.load(f) or {}).get("results") or [])
    except (OSError, json.JSONDecodeError):
        return []


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def _apply_trial_overrides(cell_count, repeat, seed, sweep_id):
    trial_id = f"{sweep_id}_cells{cell_count:02d}_rep{repeat:02d}"
    notes = f"Session grid sweep {sweep_id}: requested_count={cell_count}, repeat={repeat}, seed={seed}"

    # Used by run_match_and_plan via config module.
    runtime_config.TRIAL_FILTER_ENABLED = True
    runtime_config.TRIAL_FILTER_MODE = "random_cells"
    runtime_config.REQUESTED_ACTIVE_CELL_COUNT = int(cell_count)
    runtime_config.RANDOM_SEED = int(seed)
    runtime_config.DRY_RUN_GRID_FILTER = False

    # Used by pipeline/runtime.py module globals imported from config.
    runtime_pipeline.EXPERIMENT_TRIAL_ID = trial_id
    runtime_pipeline.EXPERIMENT_TRIAL_TYPE = "grid_sweep"
    runtime_pipeline.EXPERIMENT_LAYOUT_TYPE = "grid"
    runtime_pipeline.EXPERIMENT_NOTES = notes
    runtime_pipeline.TRIAL_FILTER_ENABLED = True
    runtime_pipeline.TRIAL_FILTER_MODE = "random_cells"
    runtime_pipeline.RANDOM_SEED = int(seed)
    runtime_pipeline.DRY_RUN_GRID_FILTER = False

    return trial_id


def _restore_runtime_defaults(orig):
    runtime_config.TRIAL_FILTER_ENABLED = orig["cfg_TRIAL_FILTER_ENABLED"]
    runtime_config.TRIAL_FILTER_MODE = orig["cfg_TRIAL_FILTER_MODE"]
    runtime_config.REQUESTED_ACTIVE_CELL_COUNT = orig["cfg_REQUESTED_ACTIVE_CELL_COUNT"]
    runtime_config.RANDOM_SEED = orig["cfg_RANDOM_SEED"]
    runtime_config.DRY_RUN_GRID_FILTER = orig["cfg_DRY_RUN_GRID_FILTER"]

    runtime_pipeline.EXPERIMENT_TRIAL_ID = orig["rt_EXPERIMENT_TRIAL_ID"]
    runtime_pipeline.EXPERIMENT_TRIAL_TYPE = orig["rt_EXPERIMENT_TRIAL_TYPE"]
    runtime_pipeline.EXPERIMENT_LAYOUT_TYPE = orig["rt_EXPERIMENT_LAYOUT_TYPE"]
    runtime_pipeline.EXPERIMENT_NOTES = orig["rt_EXPERIMENT_NOTES"]
    runtime_pipeline.TRIAL_FILTER_ENABLED = orig["rt_TRIAL_FILTER_ENABLED"]
    runtime_pipeline.TRIAL_FILTER_MODE = orig["rt_TRIAL_FILTER_MODE"]
    runtime_pipeline.RANDOM_SEED = orig["rt_RANDOM_SEED"]
    runtime_pipeline.DRY_RUN_GRID_FILTER = orig["rt_DRY_RUN_GRID_FILTER"]


def main():
    parser = argparse.ArgumentParser(description="Run repeated grid-filter trials in one process with persistent cameras.")
    parser.add_argument("--counts", default=None, help="count range/list, e.g. 1-10 or 1,3,5. Default comes from NUM_WEEDS.")
    parser.add_argument("--repeats", type=int, default=None, help="number of repeats per weed count. Default comes from NUM_TRIALS.")
    parser.add_argument("--log-dir", default="experiments/metrics/grid_sweep_logs")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--start-count", type=int, default=None)
    parser.add_argument("--start-repeat", type=int, default=1)
    parser.add_argument("--dry-run-grid-filter", action="store_true")
    args = parser.parse_args()

    counts = list(NUM_WEEDS) if args.counts is None else _parse_counts(args.counts)
    repeats = int(NUM_TRIALS) if args.repeats is None else int(args.repeats)
    if repeats <= 0:
        raise SystemExit("--repeats must be positive")

    sweep_id = args.sweep_id or datetime.now().strftime("grid_sweep_session_%Y%m%d_%H%M%S")
    log_dir = ROOT / args.log_dir / sweep_id
    log_dir.mkdir(parents=True, exist_ok=True)
    total = len(counts) * repeats
    results = _load_existing_results(log_dir)
    seed_rng = random.SystemRandom()
    seed_base = args.seed_base if args.seed_base is not None else seed_rng.randrange(1, 2_147_000_000)
    used_seeds_by_count = {}

    orig = {
        "cfg_TRIAL_FILTER_ENABLED": runtime_config.TRIAL_FILTER_ENABLED,
        "cfg_TRIAL_FILTER_MODE": runtime_config.TRIAL_FILTER_MODE,
        "cfg_REQUESTED_ACTIVE_CELL_COUNT": runtime_config.REQUESTED_ACTIVE_CELL_COUNT,
        "cfg_RANDOM_SEED": runtime_config.RANDOM_SEED,
        "cfg_DRY_RUN_GRID_FILTER": runtime_config.DRY_RUN_GRID_FILTER,
        "rt_EXPERIMENT_TRIAL_ID": runtime_pipeline.EXPERIMENT_TRIAL_ID,
        "rt_EXPERIMENT_TRIAL_TYPE": runtime_pipeline.EXPERIMENT_TRIAL_TYPE,
        "rt_EXPERIMENT_LAYOUT_TYPE": runtime_pipeline.EXPERIMENT_LAYOUT_TYPE,
        "rt_EXPERIMENT_NOTES": runtime_pipeline.EXPERIMENT_NOTES,
        "rt_TRIAL_FILTER_ENABLED": runtime_pipeline.TRIAL_FILTER_ENABLED,
        "rt_TRIAL_FILTER_MODE": runtime_pipeline.TRIAL_FILTER_MODE,
        "rt_RANDOM_SEED": runtime_pipeline.RANDOM_SEED,
        "rt_DRY_RUN_GRID_FILTER": runtime_pipeline.DRY_RUN_GRID_FILTER,
    }

    session = {}

    print(f"[GridSweepSession] sweep_id={sweep_id}", flush=True)
    print(f"[GridSweepSession] counts={counts} repeats={repeats} total={total}", flush=True)
    print(f"[GridSweepSession] logs={log_dir}", flush=True)
    print("[GridSweepSession] Camera/gantry/detector are reused across trials in this process.", flush=True)

    try:
        idx = 0
        for cell_count in counts:
            used_seeds = used_seeds_by_count.setdefault(cell_count, set())
            for repeat in range(1, repeats + 1):
                idx += 1
                if _should_skip(cell_count, repeat, args.start_count, args.start_repeat):
                    continue

                seed = seed_base + cell_count * 1000 + repeat
                while seed in used_seeds:
                    seed = seed_rng.randrange(1, 2_147_000_000)
                used_seeds.add(seed)
                trial_id = _apply_trial_overrides(cell_count, repeat, seed, sweep_id)

                log_path = log_dir / f"{idx:03d}_cells{cell_count:02d}_rep{repeat:02d}.log"
                print(
                    f"\n[GridSweepSession] {idx}/{total} start "
                    f"count={cell_count} repeat={repeat} seed={seed}",
                    flush=True,
                )

                json_before = set(JSON_DIR.glob("*.json"))
                t0 = time.perf_counter()
                rc = 0

                with open(log_path, "w") as lf:
                    tee = _Tee(sys.stdout, lf)
                    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                        try:
                            run_status = run_runtime(
                                use_real_gantry=True,
                                execute_targets=not args.dry_run_grid_filter,
                                dry_run_grid_filter=args.dry_run_grid_filter,
                                session=session,
                                keep_resources_open=True,
                            )
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            rc = 1
                            print(f"[GridSweepSession] trial exception: {exc}")

                dt = time.perf_counter() - t0
                metrics_path = _latest_metrics(json_before, trial_id=trial_id)
                summary = _read_summary(metrics_path)

                result = {
                    "index": idx,
                    "cell_count": cell_count,
                    "repeat": repeat,
                    "seed": seed,
                    "returncode": rc,
                    "elapsed_s": round(dt, 3),
                    "log_path": str(log_path),
                    "metrics_path": str(metrics_path) if metrics_path else None,
                    **summary,
                }
                results.append(result)
                _write_results(log_dir, sweep_id, results)

                print(
                    f"[GridSweepSession] {idx}/{total} done rc={rc} "
                    f"status={summary.get('status')} selected={summary.get('selected')} "
                    f"eligible={summary.get('eligible')} time={dt:.1f}s run={summary.get('run_id')}",
                    flush=True,
                )

                if run_status == "user_aborted" or summary.get("status") == "user_aborted":
                    print("[GridSweepSession] user abort detected; ending sweep cleanly.", flush=True)
                    raise SystemExit(130)

                if rc != 0 and not args.continue_on_error:
                    raise SystemExit(f"Trial failed; see {log_path}")

    finally:
        _restore_runtime_defaults(orig)
        close_runtime_session(session)
        _write_results(log_dir, sweep_id, results)
        print("[GridSweepSession] restored in-memory runtime/config overrides", flush=True)

    print(f"[GridSweepSession] complete: {len(results)}/{total} trial(s)", flush=True)
    print(f"[GridSweepSession] summary: {log_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
