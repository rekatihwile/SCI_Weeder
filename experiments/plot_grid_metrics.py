import argparse
import json
import sys
from pathlib import Path


def _load_runs(metrics_dir):
    paths = list(Path(metrics_dir).rglob("*.json"))
    runs = []
    for path in paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"warning: skip {path}: {exc}")
            continue
        if "run" in data:
            runs.append((path, data.get("run") or {}, list((data.get("targets") or {}).values())))
        else:
            runs.append((path, data, data.get("targets") or []))
    return runs


def _vals(runs, x_key, y_key):
    out = []
    for _, run, _ in runs:
        if run.get(x_key) is not None and run.get(y_key) is not None:
            out.append((float(run[x_key]), float(run[y_key])))
    return out


def _scatter(plt, runs, x_key, y_key, out):
    pts = _vals(runs, x_key, y_key)
    if not pts:
        print(f"warning: skip {out.name}: missing {x_key}/{y_key}")
        return
    x, y = zip(*pts)
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _stacked_time(plt, runs, out):
    rows = []
    for _, run, _ in runs:
        n = run.get("selected_target_count")
        if n is None:
            continue
        rows.append((
            int(n),
            float(run.get("total_travel_time_s") or 0),
            float(run.get("total_fine_align_reid_time_s") or 0),
            float(run.get("total_fine_align_pd_lk_time_s") or 0),
            float(run.get("total_fire_time_s") or 0),
        ))
    if not rows:
        print(f"warning: skip {out.name}: missing selected_target_count")
        return
    rows.sort()
    labels = [str(r[0]) for r in rows]
    travel = [r[1] for r in rows]
    reid = [r[2] for r in rows]
    align = [r[3] for r in rows]
    fire = [r[4] for r in rows]
    plt.figure()
    bottom = [0] * len(rows)
    for name, values in (("travel", travel), ("re-ID", reid), ("fine-align", align), ("fire", fire)):
        plt.bar(labels, values, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, values)]
    plt.xlabel("selected_target_count")
    plt.ylabel("time_s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _heatmap(plt, runs, field, out, reducer="mean"):
    buckets = {}
    rows = cols = None
    for _, run, targets in runs:
        rows = rows or run.get("grid_rows")
        cols = cols or run.get("grid_cols")
        for target in targets:
            r = target.get("cell_row")
            c = target.get("cell_col")
            v = target.get(field)
            if r is None or c is None or v is None:
                continue
            buckets.setdefault((int(r), int(c)), []).append(float(v))
    if not buckets or not rows or not cols:
        print(f"warning: skip {out.name}: missing cell data/{field}")
        return
    grid = [[None for _ in range(int(cols))] for _ in range(int(rows))]
    for (r, c), values in buckets.items():
        grid[r][c] = sum(values) / len(values)
    plt.figure()
    plt.imshow([[v if v is not None else float("nan") for v in row] for row in grid], origin="lower")
    plt.colorbar(label=field if reducer == "mean" else reducer)
    plt.xlabel("cell_col")
    plt.ylabel("cell_row")
    plt.title(field)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _residual_plot(plt, runs, out):
    rows = []
    for _, run, _ in runs:
        n = run.get("selected_target_count")
        t = run.get("total_treatment_time_s")
        p = run.get("planned_path_length_mm")
        if n is not None and t is not None and p is not None:
            rows.append((float(n), float(t), float(p)))
    if len(rows) < 2:
        print(f"warning: skip {out.name}: need at least two runs")
        return
    mean_x = sum(r[0] for r in rows) / len(rows)
    mean_y = sum(r[1] for r in rows) / len(rows)
    denom = sum((r[0] - mean_x) ** 2 for r in rows) or 1.0
    slope = sum((r[0] - mean_x) * (r[1] - mean_y) for r in rows) / denom
    intercept = mean_y - slope * mean_x
    x = [r[2] for r in rows]
    y = [r[1] - (intercept + slope * r[0]) for r in rows]
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("planned_path_length_mm")
    plt.ylabel("residual_treatment_time_s")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="experiments/metrics")
    parser.add_argument("--out-dir", default="figure_outputs/grid_metrics")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable: {exc}")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(args.metrics_dir)
    if not runs:
        print("warning: no JSON metrics found")
        return 0

    _stacked_time(plt, runs, out_dir / "stacked_treatment_time_vs_selected_count.png")
    _scatter(plt, runs, "planned_path_length_mm", "total_treatment_time_s",
             out_dir / "path_length_vs_treatment_time.png")
    _residual_plot(plt, runs, out_dir / "residual_time_vs_path_length.png")
    _heatmap(plt, runs, "fine_align_reid_total_time_s", out_dir / "heatmap_reid_time.png")
    _heatmap(plt, runs, "position_error_mm", out_dir / "heatmap_position_error.png")
    _heatmap(plt, runs, "hit_success", out_dir / "heatmap_hit_success.png")
    print(f"wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
