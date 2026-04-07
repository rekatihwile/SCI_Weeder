from pathlib import Path

import cv2
import numpy as np


def _pt(p):
    """Accept either a plain (x, y) tuple or a {"point": (x,y), ...} dict."""
    return p["point"] if isinstance(p, dict) else p


def _fit_frame(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame.copy()
    scale = width / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_LINEAR)


def _draw_points(img, points, color, prefix):
    for i, det in enumerate(points, start=1):
        x, y = _pt(det)
        xi = int(round(x))
        yi = int(round(y))
        cv2.circle(img, (xi, yi), 7, color, -1)
        cv2.circle(img, (xi, yi), 14, color, 2)
        cv2.putText(img, f"{prefix}{i}", (xi + 8, yi - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _make_stereo_match_canvas(frameL, frameR, left_points, right_points, matched_targets, solved_targets):
    dispL = frameL.copy()
    dispR = frameR.copy()

    _draw_points(dispL, left_points, (0, 0, 255), "L")
    _draw_points(dispR, right_points, (0, 255, 0), "R")

    stereo = np.hstack([dispL, dispR])
    split_x = frameL.shape[1]

    for i, (match, solved) in enumerate(zip(matched_targets, solved_targets), start=1):
        xl, yl = match["left_px"]
        xr, yr = match["right_px"]
        ptL = (int(round(xl)), int(round(yl)))
        ptR = (int(round(xr + split_x)), int(round(yr)))

        cv2.line(stereo, ptL, ptR, (255, 255, 0), 1)
        cv2.putText(stereo, str(i), ((ptL[0] + ptR[0]) // 2, max(20, ptL[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        xw, yw = solved["target_xy_mm"]
        label = f"{i}: ({xw:.1f}, {yw:.1f}) mm"
        cv2.putText(stereo, label, (ptL[0] + 8, min(stereo.shape[0] - 10, ptL[1] + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(stereo, "LEFT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(stereo, "RIGHT", (split_x + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(stereo, (split_x, 0), (split_x, stereo.shape[0]), (255, 255, 255), 2)

    return stereo


def _make_text_panel(matched_targets, solved_targets, width=1440, row_h=28, top_pad=60, bottom_pad=30):
    rows = max(len(matched_targets), 1)
    height = top_pad + rows * row_h + bottom_pad
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)

    cv2.putText(panel, "Triangulation debug summary", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(panel, "IDX | LEFT px | RIGHT px | SCORE | X mm | Y mm | dX mm | dY mm", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    for i, (match, solved) in enumerate(zip(matched_targets, solved_targets), start=1):
        xw, yw = solved["target_xy_mm"]
        dx, dy = solved.get("delta_xy_mm", (0.0, 0.0))
        line = (
            f"{i:02d} | {tuple(match['left_px'])} | {tuple(match['right_px'])} | "
            f"{match.get('score', 0.0):.3f} | {xw:8.2f} | {yw:8.2f} | {dx:8.2f} | {dy:8.2f}"
        )
        y = top_pad + i * row_h
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    return panel


def show_match_debug_view(
    frameL,
    frameR,
    left_points,
    right_points,
    matched_targets,
    solved_targets,
    save_path=None,
    window_name="Triangulation Debug",
    show_window=True,
):
    if frameL is None or frameR is None:
        return None

    stereo = _make_stereo_match_canvas(frameL, frameR, left_points, right_points, matched_targets, solved_targets)
    stereo = _fit_frame(stereo, 1440)
    panel = _make_text_panel(matched_targets, solved_targets, width=stereo.shape[1])

    canvas = np.vstack([stereo, panel])
    cv2.putText(canvas, "Yellow lines are stereo matches. White labels are solved planar XY targets.", (20, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), canvas)
        print(f"Saved triangulation debug view -> {save_path}")

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, canvas.shape[1], canvas.shape[0])
        cv2.moveWindow(window_name, 60, 60)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    return canvas