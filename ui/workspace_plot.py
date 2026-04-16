from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import WORKSPACE_X_MIN, WORKSPACE_X_MAX, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX, HAS_DISPLAY


def _extract_xyz_mm(solved_targets):
    xyz = []
    xy = []
    for t in solved_targets:
        tx, ty = map(float, t["target_xy_mm"])
        xy.append((tx, ty))

        x_laser_m = np.asarray(t.get("X_laser_m", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        xyz.append((tx, ty, float(x_laser_m[2] * 1000.0)))
    return xy, xyz


def show_workspace_triangulation_map(
    solved_targets,
    survey_xy=None,
    save_path=None,
    window_name="Triangulation Overview",
    width=1000,
    height=800,
    show_window=True,
):
    if not solved_targets:
        return None

    target_points, xyz_points = _extract_xyz_mm(solved_targets)
    xs = [p[0] for p in target_points]
    ys = [p[1] for p in target_points]
    zs = [p[2] for p in xyz_points]

    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Workspace XY plan view")
    ax1.set_xlim(WORKSPACE_X_MIN, WORKSPACE_X_MAX)
    ax1.set_ylim(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.35)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")

    ax1.plot(
        [WORKSPACE_X_MIN, WORKSPACE_X_MAX, WORKSPACE_X_MAX, WORKSPACE_X_MIN, WORKSPACE_X_MIN],
        [WORKSPACE_Y_MIN, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX, WORKSPACE_Y_MAX, WORKSPACE_Y_MIN],
        linewidth=1.5,
    )
    ax1.scatter(xs, ys, s=55, marker="o")

    for i, (x, y) in enumerate(target_points, start=1):
        ax1.text(x + 4.0, y + 4.0, f"{i}", fontsize=9)

    if survey_xy is not None:
        sx, sy = map(float, survey_xy)
        ax1.scatter([sx], [sy], s=90, marker="x")
        ax1.text(sx + 4.0, sy + 4.0, "Survey", fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_title("Triangulated XYZ view")
    ax2.set_xlim(WORKSPACE_X_MIN, WORKSPACE_X_MAX)
    ax2.set_ylim(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
    zmin = min(zs) if zs else -10.0
    zmax = max(zs) if zs else 10.0
    if abs(zmax - zmin) < 1e-6:
        zmin -= 5.0
        zmax += 5.0
    ax2.set_zlim(zmin, zmax)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_zlabel("Z (mm)")
    ax2.scatter(xs, ys, zs, s=55, marker="o")

    for i, (x, y, z) in enumerate(xyz_points, start=1):
        ax2.text(x, y, z, f"{i}", fontsize=8)

    if survey_xy is not None:
        sx, sy = map(float, survey_xy)
        ax2.scatter([sx], [sy], [0.0], s=90, marker="x")

    fig.suptitle(
        f"Triangulated targets: {len(target_points)} | X range {min(xs):.1f}..{max(xs):.1f} mm | "
        f"Y range {min(ys):.1f}..{max(ys):.1f} mm | Z range {min(zs):.2f}..{max(zs):.2f} mm"
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"Saved triangulation overview -> {save_path}")

    if show_window and HAS_DISPLAY:
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None:
            try:
                manager.set_window_title(window_name)
            except Exception:
                pass
        plt.show(block=True)

    plt.close(fig)
    return save_path
