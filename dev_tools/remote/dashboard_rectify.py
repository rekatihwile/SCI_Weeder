"""Stereo rectification helpers — load maps, apply remap, maybe-rectify a frame pair."""

import cv2
import numpy as np

from dashboard_state import state, RECT_NPZ_PATH

# =============================================================================
# Internal key search
# =============================================================================

def _find_npz_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    return None


# =============================================================================
# Map loading (result cached in state.rectify_cache)
# =============================================================================

def load_rectify_maps():
    if state.rectify_cache is not None:
        return state.rectify_cache

    data = np.load(RECT_NPZ_PATH)

    left_x_key = _find_npz_key(data, [
        "map1L", "left_map_x", "map1_left", "left_map1", "mapLx", "mapxL", "lmapx",
        "map1x", "mapx1",
    ])
    left_y_key = _find_npz_key(data, [
        "map2L", "left_map_y", "map2_left", "left_map2", "mapLy", "mapyL", "lmapy",
        "map1y", "mapy1",
    ])
    right_x_key = _find_npz_key(data, [
        "map1R", "right_map_x", "map1_right", "right_map1", "mapRx", "mapxR", "rmapx",
        "map2x", "mapx2",
    ])
    right_y_key = _find_npz_key(data, [
        "map2R", "right_map_y", "map2_right", "right_map2", "mapRy", "mapyR", "rmapy",
        "map2y", "mapy2",
    ])

    if None in (left_x_key, left_y_key, right_x_key, right_y_key):
        raise RuntimeError(
            "Could not find rectification map keys in "
            f"{RECT_NPZ_PATH}. Available keys: {list(data.keys())}"
        )

    state.rectify_cache = {
        "left_map_x": data[left_x_key],
        "left_map_y": data[left_y_key],
        "right_map_x": data[right_x_key],
        "right_map_y": data[right_y_key],
        "keys": {
            "left_x": left_x_key,
            "left_y": left_y_key,
            "right_x": right_x_key,
            "right_y": right_y_key,
        },
    }

    return state.rectify_cache


# =============================================================================
# Remap application
# =============================================================================

def rectify_pair(left_frame, right_frame):
    maps = load_rectify_maps()
    left_rect = cv2.remap(
        left_frame,
        maps["left_map_x"],
        maps["left_map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rect = cv2.remap(
        right_frame,
        maps["right_map_x"],
        maps["right_map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left_rect, right_rect, maps["keys"]


def maybe_rectify_pair(left_frame, right_frame, use_rectification):
    if not use_rectification:
        return left_frame, right_frame, "raw"
    left_rect, right_rect, _ = rectify_pair(left_frame, right_frame)
    return left_rect, right_rect, "rectified"
