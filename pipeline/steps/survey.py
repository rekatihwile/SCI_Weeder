"""Survey detection step helpers."""

from config import (
    DETECTOR_MODE,
    SURVEY_BURST_COUNT,
    SURVEY_CLUSTER_RADIUS_PX,
    SURVEY_MIN_HITS,
    SURVEY_POINT_MODE,
    TARGET_CLASSES,
    AVOID_CLASSES,
)
from config.survey_params import resolve_burst_count, resolve_point_mode


def flush_camera_buffer(cameras, n=8):
    for _ in range(n):
        cameras.read_pair()


def run_survey_detection(cameras, detector, coarse_mover):
    flush_camera_buffer(cameras, n=8)

    survey_burst_count = resolve_burst_count(SURVEY_BURST_COUNT)
    survey_point_mode = resolve_point_mode(SURVEY_POINT_MODE)
    print(
        f"[CV CONFIG] SURVEY burst_count={survey_burst_count} "
        f"point_mode={survey_point_mode} "
        f"target_classes={TARGET_CLASSES} "
        f"avoid_classes={AVOID_CLASSES}"
    )

    return coarse_mover.detect_stable_points(
        cameras=cameras,
        detector=detector,
        detector_mode=DETECTOR_MODE,
        burst_count=survey_burst_count,
        min_hits=SURVEY_MIN_HITS,
        cluster_radius_px=SURVEY_CLUSTER_RADIUS_PX,
        survey_classes=TARGET_CLASSES,
        point_mode=survey_point_mode,
    )
