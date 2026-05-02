"""Survey detection step helpers."""

from config import (
    DETECTOR_MODE,
    OVERRIDE_BURST_COUNT,
    OVERRIDE_BURST_NUMBER,
    OVERRIDE_POINT_MODE,
    OVERRIDE_POINT_MODE_VALUE,
    SURVEY_BURST_COUNT,
    SURVEY_CLUSTER_RADIUS_PX,
    SURVEY_MIN_HITS,
    SURVEY_POINT_MODE,
    SURVEY_TARGET_CLASSES,
)


def _resolve_burst_count(default_count):
    if OVERRIDE_BURST_NUMBER:
        return max(1, int(OVERRIDE_BURST_COUNT))
    return max(1, int(default_count))


def _resolve_point_mode(default_mode):
    mode = OVERRIDE_POINT_MODE_VALUE if OVERRIDE_POINT_MODE else default_mode
    mode = str(mode or "box_center").strip().lower()
    if mode == "heatmap":
        mode = "qpoint"
    if mode not in ("box_center", "qpoint"):
        raise ValueError(f"Unknown point mode: {mode}")
    return mode


def flush_camera_buffer(cameras, n=8):
    for _ in range(n):
        cameras.read_pair()


def run_survey_detection(cameras, detector, coarse_mover):
    flush_camera_buffer(cameras, n=8)

    survey_burst_count = _resolve_burst_count(SURVEY_BURST_COUNT)
    survey_point_mode = _resolve_point_mode(SURVEY_POINT_MODE)
    print(f"[CV CONFIG] SURVEY burst_count={survey_burst_count} point_mode={survey_point_mode}")

    return coarse_mover.detect_stable_points(
        cameras=cameras,
        detector=detector,
        detector_mode=DETECTOR_MODE,
        burst_count=survey_burst_count,
        min_hits=SURVEY_MIN_HITS,
        cluster_radius_px=SURVEY_CLUSTER_RADIUS_PX,
        survey_classes=SURVEY_TARGET_CLASSES,
        point_mode=survey_point_mode,
    )
