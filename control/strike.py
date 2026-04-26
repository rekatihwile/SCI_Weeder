from concurrent.futures import ThreadPoolExecutor
import time

from config import (
    STRIKE_PATTERN,
    LASER_FIRE_POWER,
    LASER_FIRE_DURATION_SEC,
    LASER_ARM_DELAY_SEC,
    FIRE,
    RECORD_VIDEO_FPS,
)


def _pull_recording_frames(cameras, duration_s, until_future=None):
    """Keep the raw trial recorder fed while the strike window is active."""
    duration_s = max(0.0, float(duration_s))
    interval = 1.0 / max(1.0, float(RECORD_VIDEO_FPS or 15.0))
    deadline = time.perf_counter() + duration_s

    if cameras is None:
        while time.perf_counter() < deadline or (until_future is not None and not until_future.done()):
            time.sleep(0.02)
        return 0

    frames = 0
    warned = False
    while True:
        now = time.perf_counter()
        future_done = until_future is None or until_future.done()
        if now >= deadline and future_done:
            break

        t0 = time.perf_counter()
        try:
            cameras.read_pair(retries=1)
            frames += 1
        except Exception as exc:
            if not warned:
                print(f"[STRIKE] Warning: could not pull recording frame during fire window: {exc}")
                warned = True

        elapsed = time.perf_counter() - t0
        sleep_s = max(0.0, interval - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)

    return frames


def fire_target(gantry, target, cameras=None):
    print("\n=== STRIKE ===")
    print("Fine align complete.")
    print("Firing now...")
    print(
        f"Target (mm): X={target['target_xy_mm'][0]:.2f}, "
        f"Y={target['target_xy_mm'][1]:.2f}"
    )
    print(
        f"Strike settings: S={LASER_FIRE_POWER}, "
        f"duration={LASER_FIRE_DURATION_SEC:.3f}s, "
        f"delay={LASER_ARM_DELAY_SEC:.3f}s"
    )

    if STRIKE_PATTERN != "pulse":
        raise ValueError(f"Unsupported STRIKE_PATTERN: {STRIKE_PATTERN}")

    if FIRE:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="laser-fire") as pool:
            future = pool.submit(
                gantry.fire_pulse,
                power=LASER_FIRE_POWER,
                duration_s=LASER_FIRE_DURATION_SEC,
                arm_delay_s=LASER_ARM_DELAY_SEC,
            )
            frames = _pull_recording_frames(cameras, LASER_FIRE_DURATION_SEC, until_future=future)
            future.result()
    else:
        frames = _pull_recording_frames(cameras, LASER_FIRE_DURATION_SEC)
        print("[STRIKE] FIRE=False; simulated strike window only.")

    print(f"Strike complete. Recorded {frames} frame pair(s) during fire window.")
