"""Camera setup step helpers."""


def open_cameras(start_recorder=False):
    from hardware.cameras import StereoCameras

    cameras = StereoCameras()
    try:
        cameras.open(start_recorder=start_recorder)
    except RuntimeError:
        cameras.recover()
        if start_recorder:
            cameras.start_recording()
    return cameras


def close_cameras(cameras):
    if cameras is not None:
        cameras.close()
