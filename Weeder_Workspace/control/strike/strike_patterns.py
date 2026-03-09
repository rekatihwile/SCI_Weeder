
from time import time


def fire_target(gantry, target):
    print("\n=== FIRE ===")
    print(f"Firing target at (mm): X={target['target_xy_mm'][0]:.2f}, Y={target['target_xy_mm'][1]:.2f}")
    # print("Strike pattern not implemented yet.")
    return