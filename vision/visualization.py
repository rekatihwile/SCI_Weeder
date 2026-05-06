"""
vision/visualization.py
-----------------------
Single source of truth for all detection drawing and image annotation.

Both the live runtime and the dashboard import from here, so changing a colour
or label style here updates *everything* at once.

KNOBS — edit these to restyle detections across the whole system:
  BOX_COLOR_CONFIRMED   — burst-stable locked target box colour
  BOX_COLOR_RAW         — single-frame raw YOLO detection box colour
  QPOINT_COLOR          — meristem / laser aim-point dot colour
  QPOINT_OUTLINE_COLOR  — ring around aim-point dot
  CROP_COLOR            — survey crop-window rectangle colour
  MATCH_COLOR           — stereo matched-pair highlight colour
  LABEL_FONT_SCALE      — text size on every detection label
  SHOW_CONFIDENCE       — True → prints "cls 0.85", False → prints "cls"
  SHOW_VIEW_COUNT       — True → appends "[3v]" on burst-stable detections
"""

import base64

import cv2

# =============================================================================
# KNOBS — change here to restyle detections everywhere
# =============================================================================

BOX_COLOR_CONFIRMED  = (0, 200, 255)   # orange  — burst-stable confirmed target
BOX_COLOR_RAW        = (0, 220, 0)     # green   — single-frame raw YOLO box
QPOINT_COLOR         = (0, 0, 255)     # red     — meristem / laser aim-point
QPOINT_OUTLINE_COLOR = (255, 255, 255) # white   — thin ring around aim-point
CROP_COLOR           = (0, 200, 255)   # orange  — survey crop window border
MATCH_COLOR          = (255, 0, 255)   # magenta — stereo matched pair / epipolar

LABEL_FONT       = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_THICKNESS  = 1

QPOINT_RADIUS         = 7
QPOINT_OUTLINE_RADIUS = 10

SHOW_CONFIDENCE = True   # False hides confidence score in labels
SHOW_VIEW_COUNT = True   # False hides "[3v]" burst count on stable detections


# =============================================================================
# Encoding helpers (previously in dev_tools/remote/dashboard_images.py)
# =============================================================================

def jpg_bytes(img, quality=85):
    """JPEG-encode a BGR numpy image and return raw bytes."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def b64_img(img, quality=85):
    """JPEG-encode then base64-encode an image for embedding in HTML/JSON."""
    return base64.b64encode(jpg_bytes(img, quality)).decode("ascii")


# =============================================================================
# Internal helpers
# =============================================================================

def _label_text(cls_name, conf, views=None):
    parts = [str(cls_name)]
    if SHOW_CONFIDENCE and conf is not None:
        parts.append(f"{conf:.2f}")
    if SHOW_VIEW_COUNT and views is not None:
        parts.append(f"[{views}v]")
    return " ".join(parts)


def _draw_box_with_label(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img, label,
        (x1, max(18, y1 - 6)),
        LABEL_FONT, LABEL_FONT_SCALE, color, LABEL_THICKNESS,
    )


def _draw_qpoint(img, px, py):
    cv2.circle(img, (px, py), QPOINT_RADIUS, QPOINT_COLOR, -1)
    cv2.circle(img, (px, py), QPOINT_OUTLINE_RADIUS, QPOINT_OUTLINE_COLOR, 1)


# =============================================================================
# Public drawing API
# =============================================================================

def draw_detections(frame, detections, color=None):
    """Draw dict-format detections onto *frame* and return an annotated copy.

    Each element of *detections* must be a dict with keys:
        "point"  — (x, y) centre pixel
        "box"    — (x1, y1, x2, y2) bounding box
        "cls"    — class id
        "conf"   — confidence score (0-1)
        "views"  — (optional) number of burst frames that agreed

    *color* defaults to BOX_COLOR_RAW.  Pass a BGR tuple to override per-call.
    """
    color = color or BOX_COLOR_RAW
    out = frame.copy()
    for d in detections:
        px, py = int(d["point"][0]), int(d["point"][1])
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        label = _label_text(d.get("cls", "?"), d.get("conf"), d.get("views"))
        _draw_box_with_label(out, x1, y1, x2, y2, label, color)
        _draw_qpoint(out, px, py)
    return out


def draw_stable_detections(frame, stable_points, cls_names=None):
    """Draw burst-stable cluster results onto *frame* and return an annotated copy.

    Each element of *stable_points* must be a dict with keys:
        "point"  — (x, y) burst-averaged centre pixel
        "box"    — (x1, y1, x2, y2) bounding box
        "cls"    — class id
        "conf"   — confidence score
        "views"  — number of burst frames that agreed

    *cls_names*: optional {class_id: name_str} mapping from YOLO model.names.
    Uses BOX_COLOR_CONFIRMED for all boxes.
    """
    out = frame.copy()
    for s in stable_points:
        x1, y1, x2, y2 = (int(v) for v in s["box"])
        cls_id   = s.get("cls", 0)
        cls_name = (cls_names or {}).get(cls_id, str(cls_id))
        label    = _label_text(cls_name, s.get("conf"), s.get("views"))
        _draw_box_with_label(out, x1, y1, x2, y2, label, BOX_COLOR_CONFIRMED)
        px, py = int(s["point"][0]), int(s["point"][1])
        _draw_qpoint(out, px, py)
    return out


def draw_crop(frame, crop, label_prefix="crop"):
    """Draw a survey crop-window rectangle on *frame* and return an annotated copy.

    *crop* must be a dict with keys: "x0", "y0", "x1", "y1".
    """
    x0, y0, x1, y1 = crop["x0"], crop["y0"], crop["x1"], crop["y1"]
    out = frame.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), CROP_COLOR, 2)
    cv2.putText(
        out,
        f"{label_prefix} x={x0}:{x1} y={y0}:{y1}",
        (x0 + 8, max(22, y0 - 8)),
        LABEL_FONT, 0.55, CROP_COLOR, 2,
    )
    return out


def draw_matches(left_img, right_img, matches):
    """Draw stereo matched pairs with epipolar lines on both images.

    Returns (left_annotated, right_annotated).
    Each element of *matches* must have keys "left_px" and "right_px".
    """
    left_out  = left_img.copy()
    right_out = right_img.copy()
    for i, m in enumerate(matches):
        lp = m.get("left_px")
        rp = m.get("right_px")
        if lp is None or rp is None:
            continue
        lx, ly = int(lp[0]), int(lp[1])
        rx, ry = int(rp[0]), int(rp[1])
        cv2.circle(left_out,  (lx, ly), 9, MATCH_COLOR, 2)
        cv2.circle(right_out, (rx, ry), 9, MATCH_COLOR, 2)
        cv2.putText(left_out,  f"M{i}", (lx + 8, ly - 8), LABEL_FONT, 0.6, MATCH_COLOR, 2)
        cv2.putText(right_out, f"M{i}", (rx + 8, ry - 8), LABEL_FONT, 0.6, MATCH_COLOR, 2)
        cv2.line(left_out,  (0, ly), (left_out.shape[1]  - 1, ly), MATCH_COLOR, 1)
        cv2.line(right_out, (0, ry), (right_out.shape[1] - 1, ry), MATCH_COLOR, 1)
    return left_out, right_out
