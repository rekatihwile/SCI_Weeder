"""
hardware/mock_gantry.py
-----------------------
A no-hardware stand-in for Gantry.

Drop-in replacement for hardware.gantry.Gantry.  All methods that would
open serial, send GRBL commands, or physically move hardware are replaced
with silent no-ops or simple print statements.  Internal position state
(est_x, est_y) is updated identically to real Gantry so downstream code
that calls get_estimated_xy() / get_position() sees consistent values.

Usage (controlled by config.MOCK_GANTRY):
    from hardware.mock_gantry import MockGantry
    gantry = MockGantry()           # starts at (0, 0) by default
    gantry = MockGantry(200, 200)   # starts at a custom position
"""


class MockGantry:
    def __init__(self, start_x=0.0, start_y=0.0):
        self.est_x = float(start_x)
        self.est_y = float(start_y)
        # Mirrors the real Gantry attribute so callers that inspect it don't crash.
        self.hold_steppers = False
        print(f"[MOCK GANTRY] Initialised at ({self.est_x:.3f}, {self.est_y:.3f}) — no serial opened")

    # ── position / status ─────────────────────────────────────────────────────

    def get_estimated_position(self):
        return {"x": self.est_x, "y": self.est_y}

    def get_estimated_xy(self):
        return (self.est_x, self.est_y)

    def get_position(self):
        """Return the same dict format as real Gantry.get_position()."""
        return {"x": self.est_x, "y": self.est_y}

    def sync_position_estimate(self, x, y):
        self.est_x = float(x)
        self.est_y = float(y)

    def sync_estimate_to_machine(self):
        """No machine to query; returns the current estimate unchanged."""
        return {"x": self.est_x, "y": self.est_y}

    def wait_for_idle(self, timeout=60.0):
        """Immediately returns True — mock gantry is always idle."""
        return True

    # ── motion ────────────────────────────────────────────────────────────────

    def home(self):
        print("[MOCK GANTRY] home() called — resetting position to (0, 0)")
        self.est_x = 0.0
        self.est_y = 0.0

    def move_absolute(self, x, y, feed=12000):
        self.est_x = float(x)
        self.est_y = float(y)
        print(f"[MOCK GANTRY] move_absolute X={x:.3f} Y={y:.3f} F={int(feed)}")

    def jog(self, dx, dy, feed=5000):
        self.est_x += float(dx)
        self.est_y += float(dy)

    # ── laser ─────────────────────────────────────────────────────────────────

    def fire_pulse(self, power, duration_s, arm_delay_s=0.10):
        print(f"[MOCK GANTRY] fire_pulse power={power} duration_s={duration_s:.3f} "
              f"arm_delay_s={arm_delay_s:.3f} — no laser fired")

    # ── control / shutdown ────────────────────────────────────────────────────

    def stop(self):
        pass  # No motion to cancel.

    def soft_reset(self):
        pass  # No GRBL to reset.

    def close(self):
        print("[MOCK GANTRY] close() called — nothing to close")
