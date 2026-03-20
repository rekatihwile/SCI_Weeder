from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import (
    GRBL_PORT,
    TRAINING_PHOTOS_DIR,
    WORKSPACE_X_MIN,
    WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Y_MAX,
    X_SUBSECTIONS,
    Y_SUBSECTIONS,
    PHOTO_SETTLE_SEC,
)
from hardware.cameras import StereoCameras
from hardware.gantry import Gantry
from planning.target_planner import plan_targets


MOVE_FEED_MM_MIN = 12000
WARMUP_FRAMES = 4


def ensure_training_dirs(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "left").mkdir(parents=True, exist_ok=True)
    (folder / "right").mkdir(parents=True, exist_ok=True)
    (folder / "combined").mkdir(parents=True, exist_ok=True)


def safe_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def get_single_key() -> str:
    if not sys.stdin.isatty():
        try:
            return input().strip().lower() or "enter"
        except EOFError:
            return "q"

    if os.name == "nt":
        import msvcrt
        key = msvcrt.getwch()
        if key == "\r":
            return "enter"
        if key == " ":
            return "space"
        return key.lower()

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if key in ("\r", "\n"):
        return "enter"
    if key == " ":
        return "space"
    return key.lower()


def build_grid_points(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    cols: int,
    rows: int,
) -> list[dict]:
    dx = (x_max - x_min) / cols
    dy = (y_max - y_min) / rows

    points = []
    for row in range(rows):
        y = y_min + (row + 0.5) * dy
        for col in range(cols):
            x = x_min + (col + 0.5) * dx
            points.append(
                {
                    "row": row,
                    "col": col,
                    "x_mm": round(x, 3),
                    "y_mm": round(y, 3),
                }
            )
    return points


def get_next_global_index(folder: Path) -> int:
    ensure_training_dirs(folder)

    max_idx = 0
    pattern = re.compile(r"pt_(\d+)_")

    for sub in ["left", "right", "combined"]:
        subdir = folder / sub
        if not subdir.exists():
            continue

        for file in subdir.iterdir():
            if not file.is_file():
                continue
            match = pattern.match(file.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))

    return max_idx + 1


def save_capture(folder: Path, global_idx: int, point: dict, frame_l, frame_r) -> dict:
    ensure_training_dirs(folder)

    left_dir = folder / "left"
    right_dir = folder / "right"
    combined_dir = folder / "combined"

    stem = f"pt_{global_idx:06d}_r{point['row']+1}_c{point['col']+1}"

    left_path = left_dir / f"{stem}_left.jpg"
    right_path = right_dir / f"{stem}_right.jpg"
    combined_path = combined_dir / f"{stem}_combined.jpg"

    combined = cv2.hconcat([frame_l, frame_r])

    ok_l = cv2.imwrite(str(left_path), frame_l)
    ok_r = cv2.imwrite(str(right_path), frame_r)
    ok_c = cv2.imwrite(str(combined_path), combined)

    if not (ok_l and ok_r and ok_c):
        raise RuntimeError("Failed to save one or more image files")

    return {
        "idx": global_idx,
        "row": point["row"],
        "col": point["col"],
        "x_mm": point["x_mm"],
        "y_mm": point["y_mm"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "left_file": str(left_path.relative_to(folder)),
        "right_file": str(right_path.relative_to(folder)),
        "combined_file": str(combined_path.relative_to(folder)),
    }


def warmup_and_read_pair(cameras: StereoCameras):
    for _ in range(WARMUP_FRAMES):
        cameras.read_pair()
    return cameras.read_pair()


def print_plan(points: list[dict], cols: int, rows: int, title: str):
    print(f"\n=== {title} | {rows} x {cols} GRID ===")
    for i, p in enumerate(points, start=1):
        print(
            f"{i:>2}. row={p['row']+1} col={p['col']+1} "
            f"X={p['x_mm']:.1f} mm Y={p['y_mm']:.1f} mm"
        )
    print()


def build_planned_points(points: list[dict]) -> list[dict]:
    if not points:
        return []

    start_xy = (points[0]["x_mm"], points[0]["y_mm"])
    planned = plan_targets(points, start_xy=start_xy)
    return planned


def run_one_grid(
    gantry: Gantry,
    cameras: StereoCameras,
    out_dir: Path,
    manifest: dict,
    manifest_path: Path,
    latest_path: Path,
    planned_points: list[dict],
    home_point: dict,
    start_idx: int,
    feed: float,
    settle: float,
    cycle_num: int,
) -> int:
    current_idx = start_idx

    manifest["last_cycle"] = {
        "cycle_num": cycle_num,
        "started": datetime.now().isoformat(timespec="seconds"),
        "num_points": len(planned_points),
    }
    safe_write_json(manifest_path, manifest)

    for point_num, point in enumerate(planned_points, start=1):
        print(
            f"Cycle {cycle_num} | moving to point {point_num}/{len(planned_points)} "
            f"(row {point['row']+1}, col {point['col']+1})..."
        )

        gantry.move_absolute(point["x_mm"], point["y_mm"], feed=feed)

        print(f"Settling for {settle:.2f} s before capture...")
        time.sleep(settle)

        frame_l, frame_r = warmup_and_read_pair(cameras)

        capture_entry = save_capture(out_dir, current_idx, point, frame_l, frame_r)
        capture_entry["cycle_num"] = cycle_num

        manifest["captures"].append(capture_entry)

        safe_write_json(manifest_path, manifest)
        safe_write_json(latest_path, capture_entry)

        print(
            f"Saved idx {current_idx} -> "
            f"{capture_entry['left_file']}, "
            f"{capture_entry['right_file']}, "
            f"{capture_entry['combined_file']}"
        )

        current_idx += 1

    print(
        f"Returning to home grid point "
        f"X={home_point['x_mm']:.1f}, Y={home_point['y_mm']:.1f}..."
    )
    gantry.move_absolute(home_point["x_mm"], home_point["y_mm"], feed=feed)
    time.sleep(settle)

    manifest["last_cycle"]["finished"] = datetime.now().isoformat(timespec="seconds")
    safe_write_json(manifest_path, manifest)

    print(f"Cycle {cycle_num} complete.\n")
    return current_idx


def main():
    parser = argparse.ArgumentParser(
        description="Home gantry, open cameras, and repeatedly capture full-grid stereo photo sets with TSP-style ordering."
    )
    parser.add_argument("--x-min", type=float, default=WORKSPACE_X_MIN)
    parser.add_argument("--x-max", type=float, default=WORKSPACE_X_MAX)
    parser.add_argument("--y-min", type=float, default=WORKSPACE_Y_MIN)
    parser.add_argument("--y-max", type=float, default=WORKSPACE_Y_MAX)
    parser.add_argument("--cols", type=int, default=X_SUBSECTIONS)
    parser.add_argument("--rows", type=int, default=Y_SUBSECTIONS)
    parser.add_argument("--feed", type=float, default=MOVE_FEED_MM_MIN)
    parser.add_argument("--settle", type=float, default=PHOTO_SETTLE_SEC)
    args = parser.parse_args()

    if args.cols < 1 or args.rows < 1:
        raise ValueError("rows and cols must both be >= 1")

    out_dir = Path(TRAINING_PHOTOS_DIR)
    ensure_training_dirs(out_dir)

    next_idx = get_next_global_index(out_dir)

    grid_points = build_grid_points(
        args.x_min,
        args.x_max,
        args.y_min,
        args.y_max,
        cols=args.cols,
        rows=args.rows,
    )
    planned_points = build_planned_points(grid_points)

    if not planned_points:
        raise RuntimeError("No grid points were generated.")

    home_point = grid_points[0]

    manifest_path = out_dir / "manifest.json"
    latest_path = out_dir / "latest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {"captures": []}
    else:
        manifest = {"captures": []}

    if "captures" not in manifest:
        manifest["captures"] = []

    manifest["session"] = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "grbl_port": GRBL_PORT,
        "bounds_mm": {
            "x_min": args.x_min,
            "x_max": args.x_max,
            "y_min": args.y_min,
            "y_max": args.y_max,
        },
        "grid": {
            "cols": args.cols,
            "rows": args.rows,
        },
        "settle_sec": args.settle,
        "path_mode": "nearest_neighbor",
        "home_point": {
            "x_mm": home_point["x_mm"],
            "y_mm": home_point["y_mm"],
            "row": home_point["row"],
            "col": home_point["col"],
        },
    }
    safe_write_json(manifest_path, manifest)

    print("\nSaving photos to:")
    print(out_dir.resolve())
    print("\nGRBL port:", GRBL_PORT)
    print("Next image index:", next_idx)
    print(f"Grid: {args.rows} rows x {args.cols} cols")
    print(f"Settle time before capture: {args.settle:.2f} s")
    print_plan(grid_points, args.cols, args.rows, "ORIGINAL GRID ORDER")
    print_plan(planned_points, args.cols, args.rows, "PLANNED TSP-STYLE ORDER")

    gantry = None
    cameras = None

    try:
        gantry = Gantry(GRBL_PORT)
        cameras = StereoCameras()
        cameras.open()

        print("Homing gantry first...")
        gantry.home()
        time.sleep(1.0)

        print(
            f"Moving to home grid point "
            f"X={home_point['x_mm']:.1f}, Y={home_point['y_mm']:.1f}..."
        )
        gantry.move_absolute(home_point["x_mm"], home_point["y_mm"], feed=args.feed)
        time.sleep(args.settle)

        current_idx = next_idx
        cycle_num = 1

        while True:
            print("Press [space] or [enter] to capture one full grid.")
            print("Press [q] to quit.\n")

            key = get_single_key()
            print()

            if key == "q":
                print("Quit requested.")
                break

            if key not in ("space", "enter"):
                print(f"Ignoring key '{key}'.")
                continue

            current_idx = run_one_grid(
                gantry=gantry,
                cameras=cameras,
                out_dir=out_dir,
                manifest=manifest,
                manifest_path=manifest_path,
                latest_path=latest_path,
                planned_points=planned_points,
                home_point=home_point,
                start_idx=current_idx,
                feed=args.feed,
                settle=args.settle,
                cycle_num=cycle_num,
            )

            cycle_num += 1

    finally:
        try:
            ensure_training_dirs(out_dir)
            manifest["session"]["finished"] = datetime.now().isoformat(timespec="seconds")
            safe_write_json(manifest_path, manifest)
        except Exception:
            pass

        if cameras is not None:
            cameras.close()
        if gantry is not None:
            gantry.close()


if __name__ == "__main__":
    main()