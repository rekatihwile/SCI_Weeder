"""Camera setup step helpers."""


def open_cameras(start_recorder=False):
    from hardware.cameras import StereoCameras

    cameras = StereoCameras()
    cameras.open(start_recorder=start_recorder)
    return cameras


def close_cameras(cameras):
    if cameras is not None:
        cameras.close()
