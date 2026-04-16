from pathlib import Path

import cv2
import numpy as np

from config import STEREO_MATCH_MAX_Y_DIFF_PX


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

        y_diff = float(match.get("y_diff_px", abs(yr - yl)))
        line_color = (0, 255, 255) if y_diff <= STEREO_MATCH_MAX_Y_DIFF_PX else (0, 0, 255)
        cv2.line(stereo, ptL, ptR, line_color, 2, cv2.LINE_AA)
        mid = ((ptL[0] + ptR[0]) // 2, max(20, min(ptL[1], ptR[1]) - 10))
        cv2.putText(stereo, str(i), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(stereo, str(i), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2)

        xw, yw = solved["target_xy_mm"]
        label = f"{i}: yD={y_diff:.0f}px ({xw:.1f}, {yw:.1f}) mm"
        lpos  = (ptL[0] + 10, min(stereo.shape[0] - 10, ptL[1] + 24))
        cv2.putText(stereo, label, lpos, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
        cv2.putText(stereo, label, lpos, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    cv2.putText(stereo, "LEFT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(stereo, "RIGHT", (split_x + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(stereo, (split_x, 0), (split_x, stereo.shape[0]), (255, 255, 255), 2)

    return stereo


def _make_text_panel(matched_targets, solved_targets, width=1440, row_h=30, top_pad=65, bottom_pad=20):
    # Each column: (header, x_pos).  x positions are fixed so headers and
    # values always line up regardless of value width.
    COLS = [
        ("#",        20),
        ("LEFT px",  75),
        ("RIGHT px", 240),
        ("yD px",    410),
        ("disp",     485),
        ("score",    565),
        ("X mm",     660),
        ("Y mm",     790),
        ("dX mm",    920),
        ("dY mm",    1050),
    ]
    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    HDR_SCALE = 0.55
    VAL_SCALE = 0.52
    HDR_CLR   = (140, 200, 255)
    VAL_CLR   = (220, 220, 220)

    rows   = max(len(matched_targets), 1)
    height = top_pad + rows * row_h + bottom_pad
    panel  = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)

    cv2.putText(panel, "Triangulation debug summary",
                (20, 30), FONT, 0.85, (255, 255, 255), 2)

    # Horizontal rule under title
    cv2.line(panel, (20, 40), (width - 20, 40), (60, 60, 60), 1)

    # Column headers
    for hdr, x in COLS:
        cv2.putText(panel, hdr, (x, top_pad - 8), FONT, HDR_SCALE, HDR_CLR, 1)

    cv2.line(panel, (20, top_pad - 2), (width - 20, top_pad - 2), (60, 60, 60), 1)

    for i, (match, solved) in enumerate(zip(matched_targets, solved_targets), start=1):
        xw, yw = solved["target_xy_mm"]
        dx, dy = solved.get("delta_xy_mm", (0.0, 0.0))
        lx, ly = match["left_px"]
        rx, ry = match["right_px"]

        values = [
            f"{i:02d}",
            f"({int(lx)}, {int(ly)})",
            f"({int(rx)}, {int(ry)})",
            f"{match.get('y_diff_px', abs(ry - ly)):.0f}",
            f"{match.get('disp_px', abs(rx - lx)):.0f}",
            f"{match.get('score', 0.0):.3f}",
            f"{xw:.2f}",
            f"{yw:.2f}",
            f"{dx:+.2f}",
            f"{dy:+.2f}",
        ]

        y = top_pad + i * row_h
        # Alternating row tint for readability
        if i % 2 == 0:
            cv2.rectangle(panel, (15, y - row_h + 6), (width - 15, y + 4), (28, 28, 28), -1)

        for val, (_, x) in zip(values, COLS):
            cv2.putText(panel, val, (x, y), FONT, VAL_SCALE, VAL_CLR, 1)

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
