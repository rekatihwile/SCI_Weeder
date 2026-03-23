import json
import re
import sys
import time
from pathlib import Path
from statistics import mean
import csv
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_MODEL, DEFAULT_QPOINT_MODEL, MODEL_MAP
from vision.detectors.ai_detector import AIDetector
from vision.matching import match_points
from control.coarse_move import TriangulationCoarseMover


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "training_photos"
LEFT_DIR = DATA_DIR / "left"
RIGHT_DIR = DATA_DIR / "right"
MANIFEST_PATH = DATA_DIR / "manifest.json"

BASE_RESULTS_DIR = Path(__file__).resolve().parent / "Psuedo_Matching_Results"
BASE_RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# USER SETTINGS
# ============================================================

# "single" -> one chosen pair
# "range"  -> only photo numbers in [PHOTO_NUMBER_START, PHOTO_NUMBER_END]
# "all"    -> all pairs
RUN_MODE = "all"

MODEL_NAME = MODEL_MAP.get(DEFAULT_MODEL, DEFAULT_MODEL)
QPOINT_MODEL_NAME = MODEL_MAP.get(DEFAULT_QPOINT_MODEL, DEFAULT_QPOINT_MODEL)

DETECTOR_CONF = 0.20
MIN_STABLE_VIEWS = 1

# Single-pair selection
SELECT_BY_STEM = None
PHOTO_NUMBER = 333
PHOTO_ROW = None
PHOTO_COL = None

# Range selection
PHOTO_NUMBER_START = 330
PHOTO_NUMBER_END = 345

# Optional cap after selection
MAX_PAIRS = None

# Ranking/export settings
TOP_K_TO_TRIANGULATE = 50

# Plot settings
CAPTURE_BOX_SIZE_MM = 50.0
FLIP_Z = True


def _run_label():
    if RUN_MODE == "single":
        if SELECT_BY_STEM:
            mode_text = f"single_{SELECT_BY_STEM}"
        else:
            mode_text = f"single_{PHOTO_NUMBER}"
    elif RUN_MODE == "range":
        mode_text = f"range_{PHOTO_NUMBER_START}_{PHOTO_NUMBER_END}"
    else:
        mode_text = "all"

    timestamp = datetime.now().strftime("%Y-%m-%d__%H%M%S")
    detector_stem = Path(MODEL_NAME).stem
    qpoint_stem = Path(QPOINT_MODEL_NAME).stem
    return f"{timestamp}__{detector_stem}__{qpoint_stem}__{mode_text}"


RUN_DIR = BASE_RESULTS_DIR / _run_label()
RANKINGS_DIR = RUN_DIR / "rankings"
KEYPOINTS_DIR = RUN_DIR / "keypoints"
PLOTS_DIR = RUN_DIR / "plots"
COMBINED_DIR = RUN_DIR / "combined"
CSV_PER_STEM_DIR = RUN_DIR / "csv_per_stem"
CSV_SUMMARY_DIR = RUN_DIR / "csv_summary"

for d in [RUN_DIR, RANKINGS_DIR, KEYPOINTS_DIR, PLOTS_DIR, COMBINED_DIR, CSV_PER_STEM_DIR, CSV_SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def _load_frame(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _pair_images():
    left_images = sorted(LEFT_DIR.glob("*.jpg"))
    pairs = []

    for left_path in left_images:
        stem = left_path.name.replace("_left.jpg", "")
        right_path = RIGHT_DIR / f"{stem}_right.jpg"
        if right_path.exists():
            pairs.append((left_path, right_path))

    return pairs


def _parse_stem(path_or_name):
    return Path(path_or_name).stem.replace("_left", "").replace("_right", "")


def _stem_to_number(stem):
    m = re.match(r"pt_(\d+)_", stem)
    return int(m.group(1)) if m else None


def _pair_for_stem(stem):
    left_path = LEFT_DIR / f"{stem}_left.jpg"
    right_path = RIGHT_DIR / f"{stem}_right.jpg"
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(f"Missing stereo pair for stem: {stem}")
    return left_path, right_path


def _load_manifest():
    if not MANIFEST_PATH.exists():
        return {}

    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except Exception:
        return {}

    out = {}
    for c in data.get("captures", []):
        left_file = c.get("left_file", "")
        stem = Path(left_file).stem.replace("_left", "")
        out[stem] = {
            "x_mm": c.get("x_mm"),
            "y_mm": c.get("y_mm"),
            "row": c.get("row"),
            "col": c.get("col"),
        }
    return out


def _select_single_pair(manifest_lookup):
    if SELECT_BY_STEM:
        left_path, right_path = _pair_for_stem(SELECT_BY_STEM)
        return SELECT_BY_STEM, left_path, right_path

    candidates = []
    for stem, meta in manifest_lookup.items():
        idx = _stem_to_number(stem)
        if PHOTO_NUMBER is not None and idx != PHOTO_NUMBER:
            continue
        if PHOTO_ROW is not None and meta.get("row") != PHOTO_ROW:
            continue
        if PHOTO_COL is not None and meta.get("col") != PHOTO_COL:
            continue

        try:
            left_path, right_path = _pair_for_stem(stem)
            candidates.append((stem, left_path, right_path))
        except FileNotFoundError:
            pass

    if not candidates:
        raise FileNotFoundError(
            f"No stereo pair matched PHOTO_NUMBER={PHOTO_NUMBER}, "
            f"PHOTO_ROW={PHOTO_ROW}, PHOTO_COL={PHOTO_COL}"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def _select_range_pairs():
    pairs = []
    for left_path, right_path in _pair_images():
        stem = _parse_stem(left_path)
        num = _stem_to_number(stem)
        if num is None:
            continue
        if PHOTO_NUMBER_START <= num <= PHOTO_NUMBER_END:
            pairs.append((left_path, right_path))
    return pairs


def _build_detector():
    return AIDetector(
        conf=DETECTOR_CONF,
        min_stable_views=MIN_STABLE_VIEWS,
        yolo_path=MODEL_NAME,
        qpoint_path=QPOINT_MODEL_NAME,
    )


def _save_run_info():
    out_path = RUN_DIR / "run_info.txt"
    with open(out_path, "w") as f:
        f.write(f"detector_model: {MODEL_NAME}\n")
        f.write(f"qpoint_model: {QPOINT_MODEL_NAME}\n")
        f.write(f"run_mode: {RUN_MODE}\n")
        f.write(f"top_k_to_triangulate: {TOP_K_TO_TRIANGULATE}\n")
        f.write(f"detector_conf: {DETECTOR_CONF}\n")
        f.write(f"min_stable_views: {MIN_STABLE_VIEWS}\n")
        f.write(f"flip_z: {FLIP_Z}\n")
        f.write(f"capture_box_size_mm: {CAPTURE_BOX_SIZE_MM}\n")

        if RUN_MODE == "range":
            f.write(f"photo_number_start: {PHOTO_NUMBER_START}\n")
            f.write(f"photo_number_end: {PHOTO_NUMBER_END}\n")

        if RUN_MODE == "single":
            f.write(f"selected_stem: {SELECT_BY_STEM}\n")
            f.write(f"photo_number: {PHOTO_NUMBER}\n")
            f.write(f"photo_row: {PHOTO_ROW}\n")
            f.write(f"photo_col: {PHOTO_COL}\n")

    return out_path


def detect_pairs(pairs):
    detector = _build_detector()

    if MAX_PAIRS is not None:
        pairs = pairs[:MAX_PAIRS]

    if not pairs:
        print(f"No image pairs found in {DATA_DIR}")
        return []

    detections = []
    t0 = time.time()
    total = len(pairs)

    for idx, (left_path, right_path) in enumerate(pairs, start=1):
        left_frame = _load_frame(left_path)
        right_frame = _load_frame(right_path)

        left_points = detector.cv_left.detect_points(left_frame)
        right_points = detector.cv_right.detect_points(right_frame)

        elapsed = time.time() - t0
        eta = (elapsed / idx) * (total - idx) if idx > 0 else 0.0

        matched_targets = []
        if left_points and right_points:
            matched_targets, _, _ = match_points(left_points, right_points, verbose=False)

        avg_score = mean([m.get("score", 0.0) for m in matched_targets]) if matched_targets else 0.0
        stem = _parse_stem(left_path)
        photo_num = _stem_to_number(stem)

        detections.append({
            "id": idx,
            "stem": stem,
            "photo_num": photo_num,
            "left_path": str(left_path),
            "right_path": str(right_path),
            "left_points": left_points,
            "right_points": right_points,
            "matched_targets": matched_targets,
            "num_matches": len(matched_targets),
            "avg_score": avg_score,
        })

        print(
            f"Pair {idx}/{total} | {left_path.name} | "
            f"L={len(left_points)} R={len(right_points)} "
            f"matches={len(matched_targets)} avg_score={avg_score:.3f} "
            f"| ETA {eta:.1f}s"
        )

    return detections


def _rank_detections(detections):
    return sorted(
        detections,
        key=lambda d: (d["num_matches"], d["avg_score"], -(d["photo_num"] or 10**9)),
        reverse=True,
    )


def _save_ranked_summary_csv(ranked):
    out_path = RANKINGS_DIR / "ranked_pairs.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "photo_num",
            "stem",
            "num_matches",
            "avg_score",
            "num_left_points",
            "num_right_points",
            "left_path",
            "right_path",
            "detector_model",
            "qpoint_model",
        ])

        for rank, d in enumerate(ranked, start=1):
            writer.writerow([
                rank,
                d["photo_num"],
                d["stem"],
                d["num_matches"],
                d["avg_score"],
                len(d["left_points"]),
                len(d["right_points"]),
                d["left_path"],
                d["right_path"],
                MODEL_NAME,
                QPOINT_MODEL_NAME,
            ])

    return out_path


def _render_keypoint_image(entry, rank):
    left = _load_frame(entry["left_path"])
    right = _load_frame(entry["right_path"])

    for i, p in enumerate([m["left_px"] for m in entry["matched_targets"]], start=1):
        cv2.circle(left, (int(p[0]), int(p[1])), 8, (0, 0, 255), 2)
        cv2.putText(left, str(i), (int(p[0]) + 6, int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for i, p in enumerate([m["right_px"] for m in entry["matched_targets"]], start=1):
        cv2.circle(right, (int(p[0]), int(p[1])), 8, (0, 255, 0), 2)
        cv2.putText(right, str(i), (int(p[0]) + 6, int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))

    combo = cv2.hconcat([left, right])

    header = (
        f"rank={rank} | stem={entry['stem']} | detector={Path(MODEL_NAME).stem} | "
        f"qpoint={Path(QPOINT_MODEL_NAME).stem} | matches={entry['num_matches']} | avg={entry['avg_score']:.3f}"
    )
    cv2.putText(combo, header, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = KEYPOINTS_DIR / f"rank_{rank:03d}__{entry['stem']}__keypoints.jpg"
    cv2.imwrite(str(out_path), combo)
    return out_path


def _triangulate_points(entry, manifest_lookup):
    stem = entry["stem"]
    pose = manifest_lookup.get(stem)
    if not pose:
        print(f"No manifest pose found for {stem}")
        return None, None

    ref_x = pose.get("x_mm")
    ref_y = pose.get("y_mm")
    if ref_x is None or ref_y is None:
        print(f"Manifest pose for {stem} is missing x_mm or y_mm")
        return None, None

    cm = TriangulationCoarseMover()
    solved = cm.solve_all_from_pose(entry["matched_targets"], ref_x=float(ref_x), ref_y=float(ref_y))
    return solved, (float(ref_x), float(ref_y))


def _draw_capture_box(ax, ref_xy, box_size_mm=50.0):
    if ref_xy is None:
        return

    half = box_size_mm / 2.0
    x0 = ref_xy[0] - half
    x1 = ref_xy[0] + half
    y0 = ref_xy[1] - half
    y1 = ref_xy[1] + half
    z = 0.0

    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    zs = [z, z, z, z, z]

    ax.plot(xs, ys, zs, linewidth=2, label="capture pose box")


def _plot_3d(solved_targets, ref_xy, stem, rank):
    if not solved_targets:
        return None

    xs, ys, zs = [], [], []

    for t in solved_targets:
        target_xy = t.get("target_xy_mm")
        X_rect = t.get("X_rect_m")
        if target_xy is None or X_rect is None:
            continue

        z_mm = float(X_rect[2]) * 1000.0
        if FLIP_Z:
            z_mm = -z_mm

        xs.append(float(target_xy[0]))
        ys.append(float(target_xy[1]))
        zs.append(z_mm)

    if not xs:
        return None

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs, ys, zs, marker="o", label="plants")
    ax.scatter([0.0], [0.0], [0.0], marker="x", s=80, label="base origin")

    _draw_capture_box(ax, ref_xy, box_size_mm=CAPTURE_BOX_SIZE_MM)

    if ref_xy is not None:
        ax.text(ref_xy[0], ref_xy[1], 0.0, "capture", fontsize=10)

    ax.set_xlabel("X base (mm)")
    ax.set_ylabel("Y base (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"rank={rank} | {stem} | qpoint={Path(QPOINT_MODEL_NAME).stem}")
    ax.legend()
    ax.view_init(elev=25, azim=-60)

    out_path = PLOTS_DIR / f"rank_{rank:03d}__{stem}__triangulated.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _save_combined_figure(keypoint_img_path, plot_img_path, stem, rank):
    if keypoint_img_path is None or plot_img_path is None:
        return None

    key_img = cv2.imread(str(keypoint_img_path))
    if key_img is None:
        return None
    key_img = cv2.cvtColor(key_img, cv2.COLOR_BGR2RGB)

    plot_img = mpimg.imread(str(plot_img_path))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(key_img)
    axes[0].set_title("Matched keypoints")
    axes[0].axis("off")

    axes[1].imshow(plot_img)
    axes[1].set_title("Triangulated 3D plot")
    axes[1].axis("off")

    fig.suptitle(
        f"rank={rank} | {stem} | detector={Path(MODEL_NAME).stem} | qpoint={Path(QPOINT_MODEL_NAME).stem}",
        fontsize=14
    )

    out_path = COMBINED_DIR / f"rank_{rank:03d}__{stem}__combined.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _save_stem_csv(solved_targets, ref_xy, entry, rank):
    if not solved_targets:
        return None

    out_path = CSV_PER_STEM_DIR / f"rank_{rank:03d}__{entry['stem']}__triangulated.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "photo_num",
            "stem",
            "detector_model",
            "qpoint_model",
            "target_id",
            "x_base_mm",
            "y_base_mm",
            "z_base_mm",
            "z_plot_mm",
            "left_u_px",
            "left_v_px",
            "right_u_px",
            "right_v_px",
            "capture_x_mm",
            "capture_y_mm",
            "num_matches",
            "avg_score",
        ])

        for i, t in enumerate(solved_targets, start=1):
            target_xy = t.get("target_xy_mm")
            X_rect = t.get("X_rect_m")
            left_px = t.get("left_px")
            right_px = t.get("right_px")

            if target_xy is None or X_rect is None:
                continue

            x_mm = float(target_xy[0])
            y_mm = float(target_xy[1])
            z_mm = float(X_rect[2]) * 1000.0
            z_plot_mm = -z_mm if FLIP_Z else z_mm

            lu = left_px[0] if left_px is not None else ""
            lv = left_px[1] if left_px is not None else ""
            ru = right_px[0] if right_px is not None else ""
            rv = right_px[1] if right_px is not None else ""

            cx = ref_xy[0] if ref_xy is not None else ""
            cy = ref_xy[1] if ref_xy is not None else ""

            writer.writerow([
                rank,
                entry["photo_num"],
                entry["stem"],
                MODEL_NAME,
                QPOINT_MODEL_NAME,
                i,
                x_mm,
                y_mm,
                z_mm,
                z_plot_mm,
                lu,
                lv,
                ru,
                rv,
                cx,
                cy,
                entry["num_matches"],
                entry["avg_score"],
            ])

    return out_path


def _save_all_top_points_csv(all_rows):
    out_path = CSV_SUMMARY_DIR / f"top_{TOP_K_TO_TRIANGULATE:03d}__all_points.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "photo_num",
            "stem",
            "detector_model",
            "qpoint_model",
            "target_id",
            "x_base_mm",
            "y_base_mm",
            "z_base_mm",
            "z_plot_mm",
            "left_u_px",
            "left_v_px",
            "right_u_px",
            "right_v_px",
            "capture_x_mm",
            "capture_y_mm",
            "num_matches",
            "avg_score",
        ])
        writer.writerows(all_rows)

    return out_path


def main():
    manifest_lookup = _load_manifest()

    print(f"Run output folder -> {RUN_DIR}")
    info_path = _save_run_info()
    print(f"Saved run info -> {info_path}")

    if RUN_MODE == "single":
        stem, left_path, right_path = _select_single_pair(manifest_lookup)
        pairs = [(left_path, right_path)]
        print(f"Running single pair: {stem}")

    elif RUN_MODE == "range":
        pairs = _select_range_pairs()
        print(f"Running range: {PHOTO_NUMBER_START} to {PHOTO_NUMBER_END} | pairs={len(pairs)}")

    elif RUN_MODE == "all":
        pairs = _pair_images()
        print(f"Running all pairs: {len(pairs)}")

    else:
        raise ValueError("RUN_MODE must be 'single', 'range', or 'all'")

    detections = detect_pairs(pairs)
    ranked = _rank_detections(detections)

    print("\n=== Summary ===")
    print(f"Detector model : {MODEL_NAME}")
    print(f"Qpoint model   : {QPOINT_MODEL_NAME}")
    print(f"Pairs tested   : {len(ranked)}")

    if not ranked:
        print("No detections found.")
        return

    rank_csv = _save_ranked_summary_csv(ranked)
    print(f"Saved ranked summary CSV -> {rank_csv}")

    top_entries = ranked[:TOP_K_TO_TRIANGULATE]
    print(f"Triangulating/exporting top {len(top_entries)} pairs")

    all_point_rows = []

    for rank, entry in enumerate(top_entries, start=1):
        print(
            f"[rank {rank:03d}] stem={entry['stem']} "
            f"photo_num={entry['photo_num']} matches={entry['num_matches']} avg={entry['avg_score']:.3f}"
        )

        out_img = _render_keypoint_image(entry, rank)
        if out_img:
            print(f"  saved keypoint image -> {out_img.name}")

        solved, ref_xy = _triangulate_points(entry, manifest_lookup)

        out_plot = _plot_3d(solved, ref_xy, entry["stem"], rank)
        if out_plot:
            print(f"  saved 3D plot -> {out_plot.name}")

        out_combo = _save_combined_figure(out_img, out_plot, entry["stem"], rank)
        if out_combo:
            print(f"  saved combined image -> {out_combo.name}")

        out_csv = _save_stem_csv(solved, ref_xy, entry, rank)
        if out_csv:
            print(f"  saved stem CSV -> {out_csv.name}")

        if solved:
            for i, t in enumerate(solved, start=1):
                target_xy = t.get("target_xy_mm")
                X_rect = t.get("X_rect_m")
                left_px = t.get("left_px")
                right_px = t.get("right_px")

                if target_xy is None or X_rect is None:
                    continue

                x_mm = float(target_xy[0])
                y_mm = float(target_xy[1])
                z_mm = float(X_rect[2]) * 1000.0
                z_plot_mm = -z_mm if FLIP_Z else z_mm

                lu = left_px[0] if left_px is not None else ""
                lv = left_px[1] if left_px is not None else ""
                ru = right_px[0] if right_px is not None else ""
                rv = right_px[1] if right_px is not None else ""
                cx = ref_xy[0] if ref_xy is not None else ""
                cy = ref_xy[1] if ref_xy is not None else ""

                all_point_rows.append([
                    rank,
                    entry["photo_num"],
                    entry["stem"],
                    MODEL_NAME,
                    QPOINT_MODEL_NAME,
                    i,
                    x_mm,
                    y_mm,
                    z_mm,
                    z_plot_mm,
                    lu,
                    lv,
                    ru,
                    rv,
                    cx,
                    cy,
                    entry["num_matches"],
                    entry["avg_score"],
                ])

    all_points_csv = _save_all_top_points_csv(all_point_rows)
    print(f"\nSaved combined top-points CSV -> {all_points_csv}")


if __name__ == "__main__":
    main()