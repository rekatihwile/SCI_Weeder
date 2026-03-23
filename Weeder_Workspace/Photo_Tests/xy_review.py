from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "Psuedo_Matching_Results"

runs = [d for d in RESULTS_ROOT.iterdir() if d.is_dir()]
if not runs:
    raise FileNotFoundError(f"No run folders found in {RESULTS_ROOT}")

BASE_DIR = max(runs, key=lambda d: d.stat().st_mtime)
print(f"Using run folder: {BASE_DIR}")

KEYPOINTS_DIR = BASE_DIR / "keypoints"
CSV_DIR = BASE_DIR / "csv_per_stem"
OUT_DIR = BASE_DIR / "xy_review_redrawn"
OUT_DIR.mkdir(exist_ok=True)

for d in [KEYPOINTS_DIR, CSV_DIR]:
    if not d.exists():
        raise FileNotFoundError(f"Missing folder: {d}")

csv_files = sorted(CSV_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")

print(f"Found {len(csv_files)} CSV files.")

# -------------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------------
MIRROR_X = True
MIRROR_Y = True
FLIP_Z_FOR_REVIEW = True

# Mirror around capture pose if available; otherwise use point-cloud center
MIRROR_AROUND_CAPTURE = True

CAPTURE_BOX_HALF_MM = 25.0


def transform_points(df):
    x = df["x_base_mm"].to_numpy(dtype=float).copy()
    y = df["y_base_mm"].to_numpy(dtype=float).copy()

    # Prefer raw z if available, otherwise use z_plot_mm
    if "z_base_mm" in df.columns:
        z = df["z_base_mm"].to_numpy(dtype=float).copy()
    else:
        z = df["z_plot_mm"].to_numpy(dtype=float).copy()

    if "capture_x_mm" in df.columns and "capture_y_mm" in df.columns:
        cx_raw = df["capture_x_mm"].iloc[0]
        cy_raw = df["capture_y_mm"].iloc[0]
    else:
        cx_raw = np.nan
        cy_raw = np.nan

    if MIRROR_AROUND_CAPTURE and pd.notna(cx_raw) and pd.notna(cy_raw):
        cx = float(cx_raw)
        cy = float(cy_raw)
    else:
        cx = 0.5 * (np.nanmin(x) + np.nanmax(x))
        cy = 0.5 * (np.nanmin(y) + np.nanmax(y))

    if MIRROR_X:
        x = 2.0 * cx - x
    if MIRROR_Y:
        y = 2.0 * cy - y
    if FLIP_Z_FOR_REVIEW:
        z = -z

    return x, y, z, cx, cy


for csv_path in csv_files:
    stem_base = csv_path.name.replace("__triangulated.csv", "")

    keypoint_path = KEYPOINTS_DIR / f"{stem_base}__keypoints.jpg"
    out_path = OUT_DIR / f"{stem_base}__review_redrawn.png"

    if not keypoint_path.exists():
        print(f"[WARN] Missing keypoint image: {keypoint_path.name}")
        continue

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARN] Empty CSV: {csv_path.name}")
        continue

    img_keypoints = mpimg.imread(keypoint_path)
    x, y, z, cx, cy = transform_points(df)

    stem = df["stem"].iloc[0] if "stem" in df.columns else stem_base
    rank = df["rank"].iloc[0] if "rank" in df.columns else "?"
    qpoint = df["qpoint_model"].iloc[0] if "qpoint_model" in df.columns else "unknown"

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 1.4],   # bottom row taller
        width_ratios=[1.6, 1.0],    # give more width to 3D plot
        hspace=0.08,
        wspace=0.08
    )

    # ---------------------------------------------------------------------
    # Top row: stereo keypoint image
    # ---------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(img_keypoints)
    ax0.set_title("Stereo keypoints")
    ax0.axis("off")

    # ---------------------------------------------------------------------
    # Bottom left: redrawn 3D plot from CSV
    # ---------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1, 0], projection="3d")
    ax1.scatter(x, y, z, s=28, depthshade=True, label="plants")
    ax1.scatter([0.0], [0.0], [0.0], marker="x", s=80, label="base origin")

    if pd.notna(cx) and pd.notna(cy):
        bx = [cx - CAPTURE_BOX_HALF_MM, cx + CAPTURE_BOX_HALF_MM, cx + CAPTURE_BOX_HALF_MM,
              cx - CAPTURE_BOX_HALF_MM, cx - CAPTURE_BOX_HALF_MM]
        by = [cy - CAPTURE_BOX_HALF_MM, cy - CAPTURE_BOX_HALF_MM, cy + CAPTURE_BOX_HALF_MM,
              cy + CAPTURE_BOX_HALF_MM, cy - CAPTURE_BOX_HALF_MM]
        bz = [0, 0, 0, 0, 0]
        ax1.plot(bx, by, bz, linewidth=2, label="capture pose box")
        ax1.text(cx, cy, 0, "capture", fontsize=8)

    if "target_id" in df.columns:
        for xi, yi, zi, tid in zip(x, y, z, df["target_id"]):
            ax1.text(xi, yi, zi, str(tid), fontsize=7)

    ax1.set_title("Redrawn 3D plot")
    ax1.set_xlabel("X base (mm)")
    ax1.set_ylabel("Y base (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.view_init(elev=25, azim=-60)
    ax1.legend(fontsize=8)

    # ---------------------------------------------------------------------
    # Bottom right: redrawn XY projection from CSV
    # ---------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.scatter(x, y, s=70)
    ax2.set_title("Redrawn XY projection")
    ax2.set_xlabel("X base (mm)")
    ax2.set_ylabel("Y base (mm)")
    ax2.grid(True)
    ax2.set_aspect("equal", adjustable="box")

    if pd.notna(cx) and pd.notna(cy):
        xs = [cx - CAPTURE_BOX_HALF_MM, cx + CAPTURE_BOX_HALF_MM, cx + CAPTURE_BOX_HALF_MM,
              cx - CAPTURE_BOX_HALF_MM, cx - CAPTURE_BOX_HALF_MM]
        ys = [cy - CAPTURE_BOX_HALF_MM, cy - CAPTURE_BOX_HALF_MM, cy + CAPTURE_BOX_HALF_MM,
              cy + CAPTURE_BOX_HALF_MM, cy - CAPTURE_BOX_HALF_MM]
        ax2.plot(xs, ys, linewidth=2)
        ax2.scatter([cx], [cy], marker="x", s=90)
        ax2.text(cx, cy, " capture", fontsize=9)

    if "target_id" in df.columns:
        for xi, yi, tid in zip(x, y, df["target_id"]):
            ax2.text(xi, yi, str(tid), fontsize=8)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    dx = max(20.0, 0.08 * (xmax - xmin + 1e-9))
    dy = max(20.0, 0.08 * (ymax - ymin + 1e-9))
    ax2.set_xlim(xmin - dx, xmax + dx)
    ax2.set_ylim(ymin - dy, ymax + dy)

    fig.suptitle(
        f"rank={rank} | stem={stem} | qpoint={qpoint} | "
        f"mirror_x={MIRROR_X} | mirror_y={MIRROR_Y} | flip_z={FLIP_Z_FOR_REVIEW}",
        fontsize=14
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out_path.name}")

print(f"\nDone. Review images saved in: {OUT_DIR}")