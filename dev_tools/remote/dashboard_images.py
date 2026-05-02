"""Pure image drawing and encoding helpers — no hardware, no Flask."""

import base64
import cv2
import numpy as np  # noqa: F401  (imported so callers can rely on this module for numpy)

# =============================================================================
# Encoding
# =============================================================================

def jpg_bytes(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def b64_img(img):
    return base64.b64encode(jpg_bytes(img)).decode("ascii")


# =============================================================================
# Drawing overlays
# =============================================================================

def draw_crop(frame, crop, label_prefix="crop"):
    x0, y0, x1, y1 = crop["x0"], crop["y0"], crop["x1"], crop["y1"]
    out = frame.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 2)
    cv2.putText(
        out,
        f"{label_prefix} x={x0}:{x1} y={y0}:{y1}",
        (x0 + 8, max(22, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 200, 255),
        2,
    )
    return out


def draw_detections(frame, detections, color=(0, 0, 255)):
    out = frame.copy()
    for d in detections:
        px, py = d["point"]
        x1, y1, x2, y2 = d["box"]
        cls_id = d.get("cls", "?")
        conf = d.get("conf", 0.0)
        views = d.get("views", 1)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(out, (int(px), int(py)), 7, color, -1)
        cv2.putText(
            out,
            f"cls={cls_id} {conf:.2f} {views}v",
            (int(x1), max(18, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    return out


def draw_matches(left_img, right_img, matches):
    left_out = left_img.copy()
    right_out = right_img.copy()
    for i, m in enumerate(matches):
        lp = m.get("left_px")
        rp = m.get("right_px")
        if lp is None or rp is None:
            continue
        lx, ly = int(lp[0]), int(lp[1])
        rx, ry = int(rp[0]), int(rp[1])
        cv2.circle(left_out, (lx, ly), 9, (255, 0, 255), 2)
        cv2.circle(right_out, (rx, ry), 9, (255, 0, 255), 2)
        cv2.putText(left_out, f"M{i}", (lx + 8, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(right_out, f"M{i}", (rx + 8, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.line(left_out, (0, ly), (left_out.shape[1] - 1, ly), (255, 0, 255), 1)
        cv2.line(right_out, (0, ry), (right_out.shape[1] - 1, ry), (255, 0, 255), 1)
    return left_out, right_out


def draw_horizontal_lines(frame, spacing=40):
    out = frame.copy()
    h, w = out.shape[:2]
    for y in range(0, h, spacing):
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 255), 1)
        cv2.putText(
            out,
            str(y),
            (8, max(16, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
    return out
