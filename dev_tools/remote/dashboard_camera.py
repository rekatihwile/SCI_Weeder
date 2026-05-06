"""Camera lifecycle helpers: open, close, recover, preview frame, MJPEG stream.

Also owns parse_bool and crop_bounds_for_side because both are needed here and
placing them here avoids a circular import with dashboard_yolo.
"""

from dashboard_state import state, lock, FRAME_WIDTH, FRAME_HEIGHT
from dashboard_images import b64_img, jpg_bytes
from dashboard_rectify import maybe_rectify_pair
from hardware.cameras import StereoCameras

# =============================================================================
# Parameter parsers (also imported by dashboard_yolo)
# =============================================================================

def parse_bool(value):
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "on")


def crop_bounds_for_side(params, side):
    w, h = FRAME_WIDTH, FRAME_HEIGHT

    mode = params.get("mode", "center")
    crop_w = int(params.get("crop_w", 704))
    crop_h = int(params.get("crop_h", 704))

    if side == "left":
        offset_x = int(params.get("left_offset_x", 0))
        offset_y = int(params.get("left_offset_y", 0))
    else:
        offset_x = int(params.get("right_offset_x", 0))
        offset_y = int(params.get("right_offset_y", 0))

    crop_w = max(32, min(w, crop_w))
    crop_h = max(32, min(h, crop_h))

    if mode == "full":
        return {"x0": 0, "y0": 0, "x1": w, "y1": h}

    cx = w // 2 + offset_x
    cy = h // 2 + offset_y

    if mode == "center_facing":
        if side == "left":
            cx = 3 * w // 4 + offset_x
        else:
            cx = w // 4 + offset_x
    elif mode == "left":
        cx = w // 4 + offset_x
    elif mode == "right":
        cx = 3 * w // 4 + offset_x
    elif mode == "top":
        cy = h // 4 + offset_y
    elif mode == "bottom":
        cy = 3 * h // 4 + offset_y

    x0 = int(max(0, min(w - crop_w, cx - crop_w // 2)))
    y0 = int(max(0, min(h - crop_h, cy - crop_h // 2)))

    return {"x0": x0, "y0": y0, "x1": x0 + crop_w, "y1": y0 + crop_h}


# =============================================================================
# Camera lifecycle
# =============================================================================

def ensure_cameras():
    if state.cameras is None:
        state.cameras = StereoCameras()
    if state.cameras.left is None or state.cameras.right is None:
        try:
            state.cameras.open(start_recorder=False)
        except RuntimeError:
            state.cameras.recover()
    return state.cameras


def close_all():
    if state.cameras is not None:
        state.cameras.close()
        state.cameras = None


def reset_cameras_sequence():
    """Recover the shared stereo camera interface and return preview images."""
    if state.cameras is None:
        state.cameras = StereoCameras()

    state.cameras.recover()

    fL, fR = state.cameras.read_pair()
    if fL is None or fR is None:
        raise RuntimeError("Camera reset reopened cameras, but stereo read_pair returned None.")

    return {
        "recovery": "StereoCameras.recover",
        "left_image": b64_img(fL),
        "right_image": b64_img(fR),
        "frame": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
    }


# =============================================================================
# Preview frame
# =============================================================================

def get_preview_frame(params):
    cam = ensure_cameras()
    use_rectification = parse_bool(params.get("rectified", False))
    left_crop = crop_bounds_for_side(params, "left")
    right_crop = crop_bounds_for_side(params, "right")

    fL, fR = cam.read_pair()
    fL, fR, frame_mode = maybe_rectify_pair(fL, fR, use_rectification)

    return {
        "ok": True,
        "left_image": b64_img(fL),
        "right_image": b64_img(fR),
        "left_crop": left_crop,
        "right_crop": right_crop,
        "frame": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
        "frame_mode": frame_mode,
    }


# =============================================================================
# MJPEG stream generator
# =============================================================================

def mjpeg_generator(side, rectified=False):
    cam = ensure_cameras()
    while True:
        with lock:
            fL, fR = cam.read_pair()
            fL, fR, _ = maybe_rectify_pair(fL, fR, rectified)
        frame = fL if side == "left" else fR
        data = jpg_bytes(frame, quality=75)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        )
