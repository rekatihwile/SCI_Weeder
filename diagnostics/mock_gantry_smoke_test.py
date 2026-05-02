"""
diagnostics/mock_gantry_smoke_test.py
--------------------------------------
Instantiate MockGantry and verify that all key methods behave correctly
without opening serial or moving hardware.
"""

import sys
import os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hardware.mock_gantry import MockGantry

print()
print("=" * 56)
print("  MockGantry Smoke Test")
print("=" * 56)

passed = 0
failed = 0


def check(label, actual, expected):
    global passed, failed
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        passed += 1
    else:
        print(f"  [FAIL] {label}: expected {expected!r}, got {actual!r}")
        failed += 1


def approx_eq(a, b, tol=1e-9):
    return abs(a - b) < tol


def check_approx(label, actual, expected, tol=1e-9):
    global passed, failed
    if approx_eq(actual, expected, tol):
        print(f"  [OK]   {label}: {actual!r}")
        passed += 1
    else:
        print(f"  [FAIL] {label}: expected ~{expected!r}, got {actual!r}")
        failed += 1


# --- Instantiate at default origin ---
g = MockGantry()
check("initial est_x", g.est_x, 0.0)
check("initial est_y", g.est_y, 0.0)

# --- get_estimated_xy / get_estimated_position / get_position ---
xy = g.get_estimated_xy()
check("get_estimated_xy type", type(xy), tuple)
check_approx("get_estimated_xy x", xy[0], 0.0)
check_approx("get_estimated_xy y", xy[1], 0.0)

pos = g.get_estimated_position()
check("get_estimated_position keys", sorted(pos.keys()), ["x", "y"])

gpos = g.get_position()
check("get_position keys", sorted(gpos.keys()), ["x", "y"])

# --- home() resets to (0, 0) ---
g.sync_position_estimate(99.0, 88.0)          # move first so reset is visible
g.home()
check_approx("after home est_x", g.est_x, 0.0)
check_approx("after home est_y", g.est_y, 0.0)

# --- move_absolute updates position ---
g.move_absolute(150.0, 200.0, feed=8000)
check_approx("after move_absolute est_x", g.est_x, 150.0)
check_approx("after move_absolute est_y", g.est_y, 200.0)
xy2 = g.get_estimated_xy()
check_approx("get_estimated_xy after move x", xy2[0], 150.0)
check_approx("get_estimated_xy after move y", xy2[1], 200.0)

# --- jog updates relatively ---
g.jog(10.0, -5.0)
check_approx("after jog est_x", g.est_x, 160.0)
check_approx("after jog est_y", g.est_y, 195.0)

# --- sync_estimate_to_machine returns current estimate ---
result = g.sync_estimate_to_machine()
check_approx("sync_estimate_to_machine x", result["x"], 160.0)
check_approx("sync_estimate_to_machine y", result["y"], 195.0)

# --- wait_for_idle returns True ---
check("wait_for_idle", g.wait_for_idle(), True)

# --- fire_pulse (just prints, no exception) ---
print("  [LOG]  fire_pulse output:")
g.fire_pulse(power=500, duration_s=0.05)
passed += 1
print(f"  [OK]   fire_pulse completed without exception")

# --- stop / soft_reset / close (all no-ops, no exception) ---
g.stop()
passed += 1
print("  [OK]   stop() completed without exception")
g.soft_reset()
passed += 1
print("  [OK]   soft_reset() completed without exception")
g.close()
passed += 1
print("  [OK]   close() completed without exception")

# --- Custom start position ---
g2 = MockGantry(start_x=100.0, start_y=50.0)
check_approx("custom start_x", g2.est_x, 100.0)
check_approx("custom start_y", g2.est_y, 50.0)

# --- Summary ---
print()
print("=" * 56)
print(f"  Results: {passed} passed, {failed} failed")
print("=" * 56)

if failed:
    sys.exit(1)
