from pathlib import Path

import cv2
import numpy as np


def _to_canvas_points(points_xy, width, height, pad=60):
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    sx = (width - 2 * pad) / span_x
    sy = (height - 2 * pad) / span_y
    s = min(sx, sy)

    mapped = []
    for x, y in points_xy:
        cx = int(round(pad + (x - min_x) * s))
        cy = int(round(height - pad - (y - min_y) * s))
        mapped.append((cx, cy))

    bounds = {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
    }
    return mapped, bounds


def _draw_axes(canvas, width, height, pad=60):
    cv2.line(canvas, (pad, height - pad), (width - pad, height - pad), (120, 120, 120), 1)
    cv2.line(canvas, (pad, pad), (pad, height - pad), (120, 120, 120), 1)
    cv2.putText(canvas, "X (mm)", (width - pad - 60, height - pad + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(canvas, "Y (mm)", (15, pad - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


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

    target_points = [tuple(map(float, t["target_xy_mm"])) for t in solved_targets]
    all_points = list(target_points)

    if survey_xy is not None:
        all_points.append((float(survey_xy[0]), float(survey_xy[1])))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    _draw_axes(canvas, width, height)
    canvas_points, bounds = _to_canvas_points(all_points, width, height)

    survey_canvas = None
    if survey_xy is not None:
        survey_canvas = canvas_points[-1]
        target_canvas_points = canvas_points[:-1]
    else:
        target_canvas_points = canvas_points

    if survey_canvas is not None:
        cv2.drawMarker(canvas, survey_canvas, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            canvas,
            f"Survey ({survey_xy[0]:.1f}, {survey_xy[1]:.1f}) mm",
            (survey_canvas[0] + 10, survey_canvas[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    for i, ((x_mm, y_mm), (cx, cy)) in enumerate(zip(target_points, target_canvas_points), start=1):
        cv2.circle(canvas, (cx, cy), 6, (0, 200, 0), -1)
        cv2.circle(canvas, (cx, cy), 12, (0, 120, 0), 1)
        cv2.putText(
            canvas,
            f"{i}: ({x_mm:.1f}, {y_mm:.1f})",
            (cx + 10, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )

    title = f"Triangulated planar targets: {len(target_points)}"
    cv2.putText(canvas, title, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        f"X range: {bounds['min_x']:.1f} to {bounds['max_x']:.1f} mm | Y range: {bounds['min_y']:.1f} to {bounds['max_y']:.1f} mm",
        (30, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        canvas,
        "Press any key to continue",
        (30, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), canvas)
        print(f"Saved triangulation overview -> {save_path}")

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.moveWindow(window_name, 100, 100)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    return canvas
