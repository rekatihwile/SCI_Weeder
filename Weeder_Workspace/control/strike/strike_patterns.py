from config import (
    STRIKE_PATTERN,
    LASER_FIRE_POWER,
    LASER_FIRE_DURATION_SEC,
    LASER_ARM_DELAY_SEC,
)
import time
def fire_target(gantry, target):
    print("\n=== STRIKE ===")
    print("Fine align complete.")
    time.sleep(0.5)  # Brief pause before firing
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

    gantry.fire_pulse(
        power=LASER_FIRE_POWER,
        duration_s=LASER_FIRE_DURATION_SEC,
        arm_delay_s=LASER_ARM_DELAY_SEC,
    )

    print("Strike complete.")