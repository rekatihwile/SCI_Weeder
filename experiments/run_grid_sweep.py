import argparse
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment.py"
RUNTIME_FLAGS_PATH = ROOT / "config" / "runtime_flags.py"
DEFAULT_PYTHON = Path("/home/eli/venvs/laserweeder_cv412/bin/python")


def _replace_assignment(text, name, value):
    rendered = repr(value)
    pattern = rf"^{name}\s*=.*$"
    repl = f"{name} = {rendered}"
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n == 0:
        raise RuntimeError(f"Could not find config assignment for {name}")
    return new_text


def _write_trial_config(base_text, cell_count, repeat, seed, sweep_id):
    text = base_text
    text = _replace_assignment(text, "EXPERIMENT_TRIAL_ID", f"{sweep_id}_cells{cell_count:02d}_rep{repeat:02d}")
    text = _replace_assignment(text, "EXPERIMENT_TRIAL_TYPE", "grid_sweep")
    text = _replace_assignment(text, "EXPERIMENT_LAYOUT_TYPE", "grid")
    text = _replace_assignment(
        text,
        "EXPERIMENT_NOTES",
        f"Automated grid sweep {sweep_id}: requested_count={cell_count}, repeat={repeat}, seed={seed}",
    )
    text = _replace_assignment(text, "TRIAL_FILTER_ENABLED", True)
    text = _replace_assignment(text, "TRIAL_FILTER_MODE", "random_cells")
    text = _replace_assignment(text, "REQUESTED_ACTIVE_CELL_COUNT", int(cell_count))
    text = _replace_assignment(text, "RANDOM_SEED", int(seed))
    text = _replace_assignment(text, "DRY_RUN_GRID_FILTER", False)
    CONFIG_PATH.write_text(text)


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
    json_dir = ROOT / "experiments" / "metrics" / "json"
    after = set(json_dir.glob("*.json"))
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
        "status": run.get("run_status"),
        "requested": run.get("requested_active_cell_count"),
        "eligible": run.get("eligible_target_count"),
        "selected": run.get("selected_target_count"),
        "rejected": run.get("rejected_target_count"),
        "planned_path_mm": run.get("planned_path_length_mm"),
        "total_s": run.get("total_run_time_s"),
        "cells": run.get("selected_cell_ids"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run repeated real grid-filter trials.")
    parser.add_argument("--counts", default="1-10", help="count range/list, e.g. 1-10 or 1,3,5")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--log-dir", default="experiments/metrics/grid_sweep_logs")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--sweep-id", default=None, help="resume/write into an existing sweep log directory")
    parser.add_argument("--start-count", type=int, default=None)
    parser.add_argument("--start-repeat", type=int, default=1)
    parser.add_argument("--disable-recording", action="store_true", help="temporarily set RECORD_TRIAL=False during the sweep")
    args = parser.parse_args()

    counts = _parse_counts(args.counts)
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    py = Path(args.python)
    if not py.exists():
        raise SystemExit(f"Python interpreter not found: {py}")

    sweep_id = args.sweep_id or datetime.now().strftime("grid_sweep_%Y%m%d_%H%M%S")
    log_dir = ROOT / args.log_dir / sweep_id
    log_dir.mkdir(parents=True, exist_ok=True)
    base_text = CONFIG_PATH.read_text()
    runtime_flags_text = RUNTIME_FLAGS_PATH.read_text()
    seed_base = args.seed_base if args.seed_base is not None else int(time.time()) % 1_000_000
    total = len(counts) * args.repeats
    results = _load_existing_results(log_dir)

    print(f"[GridSweep] sweep_id={sweep_id}", flush=True)
    print(f"[GridSweep] counts={counts} repeats={args.repeats} total={total}", flush=True)
    print(f"[GridSweep] logs={log_dir}", flush=True)
    print("[GridSweep] FIRE is controlled by config/runtime_flags.py; verify it before running live laser trials.", flush=True)
    if args.disable_recording:
        RUNTIME_FLAGS_PATH.write_text(_replace_assignment(runtime_flags_text, "RECORD_TRIAL", False))
        print("[GridSweep] RECORD_TRIAL temporarily set to False for this sweep.", flush=True)

    try:
        idx = 0
        for cell_count in counts:
            for repeat in range(1, args.repeats + 1):
                idx += 1
                if _should_skip(cell_count, repeat, args.start_count, args.start_repeat):
                    continue
                seed = seed_base + cell_count * 1000 + repeat
                trial_id = f"{sweep_id}_cells{cell_count:02d}_rep{repeat:02d}"
                _write_trial_config(base_text, cell_count, repeat, seed, sweep_id)
                log_path = log_dir / f"{idx:03d}_cells{cell_count:02d}_rep{repeat:02d}.log"
                json_before = set((ROOT / "experiments" / "metrics" / "json").glob("*.json"))
                print(
                    f"\n[GridSweep] {idx}/{total} start "
                    f"count={cell_count} repeat={repeat} seed={seed}",
                    flush=True,
                )
                t0 = time.perf_counter()
                with open(log_path, "w") as log:
                    proc = subprocess.Popen(
                        [str(py), "main.py"],
                        cwd=str(ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )

                    def _tee(pipe, file_obj):
                        for line in pipe:
                            file_obj.write(line)
                            file_obj.flush()
                            sys.stdout.write(line)
                            sys.stdout.flush()

                    tee_thread = threading.Thread(target=_tee, args=(proc.stdout, log), daemon=True)
                    tee_thread.start()
                    proc.wait()
                    tee_thread.join()
                dt = time.perf_counter() - t0
                metrics_path = _latest_metrics(json_before, trial_id=trial_id)
                summary = _read_summary(metrics_path)
                result = {
                    "index": idx,
                    "cell_count": cell_count,
                    "repeat": repeat,
                    "seed": seed,
                    "returncode": proc.returncode,
                    "elapsed_s": round(dt, 3),
                    "log_path": str(log_path),
                    "metrics_path": str(metrics_path) if metrics_path else None,
                    **summary,
                }
                results.append(result)
                _write_results(log_dir, sweep_id, results)
                print(
                    f"[GridSweep] {idx}/{total} done rc={proc.returncode} "
                    f"status={summary.get('status')} selected={summary.get('selected')} "
                    f"eligible={summary.get('eligible')} time={dt:.1f}s "
                    f"run={summary.get('run_id')}",
                    flush=True,
                )
                if proc.returncode != 0 and not args.continue_on_error:
                    raise SystemExit(f"Trial failed; see {log_path}")
    finally:
        CONFIG_PATH.write_text(base_text)
        RUNTIME_FLAGS_PATH.write_text(runtime_flags_text)
        print("[GridSweep] restored original config/experiment.py and config/runtime_flags.py", flush=True)
        _write_results(log_dir, sweep_id, results)

    print(f"[GridSweep] complete: {len(results)}/{total} trial(s)", flush=True)
    print(f"[GridSweep] summary: {log_dir / 'summary.json'}", flush=True)


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


def _should_skip(cell_count, repeat, start_count, start_repeat):
    if start_count is None:
        return False
    if cell_count < start_count:
        return True
    if cell_count == start_count and repeat < start_repeat:
        return True
    return False


if __name__ == "__main__":
    main()
