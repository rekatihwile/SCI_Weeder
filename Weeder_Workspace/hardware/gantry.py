import re
import serial
import time


class Gantry:
    def __init__(self, port, baudrate=115200, timeout=0.2, hold_steppers=True):
        self.serial = serial.Serial(port, baudrate, timeout=timeout, write_timeout=timeout)
        time.sleep(2.0)
        self.serial.reset_input_buffer()

        self.est_x = 0.0
        self.est_y = 0.0
        self.hold_steppers = hold_steppers
        self.default_idle_delay_ms = 25
        self._stepper_hold_active = False   # cached — avoids re-sending $1=255 every move

        if self.hold_steppers:
            self.enable_stepper_hold()

    # ── serial primitives ─────────────────────────────────────────────────────

    def send_raw(self, cmd):
        self.serial.write(f"{cmd.strip()}\n".encode())
        return self.serial.readline().decode(errors="ignore").strip()

    def _send_and_wait(self, cmd):
        self.serial.write((cmd.strip() + "\r\n").encode())
        self.serial.flush()
        while True:
            resp = self.serial.readline().decode(errors="ignore").strip()
            if not resp:
                time.sleep(0.001)
                continue
            if "ok" in resp.lower():
                return resp
            if "error" in resp.lower() or "alarm" in resp.lower():
                raise RuntimeError(f"GRBL returned: {resp!r} for command: {cmd!r}")

    def stop(self):
        self.serial.write(b"\x85")

    def soft_reset(self):
        self.serial.write(b"\x18")
        time.sleep(0.5)
        self.serial.reset_input_buffer()

    # ── status / position ─────────────────────────────────────────────────────

    def get_status_line(self):
        self.serial.reset_input_buffer()
        self.serial.write(b"?")
        for _ in range(10):
            line = self.serial.readline().decode(errors="ignore").strip()
            if line.startswith("<") and line.endswith(">"):
                return line
        return None

    def get_position(self):
        line = self.get_status_line()
        if line is None or "MPos:" not in line:
            return None
        try:
            content = line[1:-1]
            parts   = content.split("|")
            mpos    = [p for p in parts if p.startswith("MPos:")][0]
            coords  = mpos.replace("MPos:", "").split(",")
            return {"x": float(coords[0]), "y": float(coords[1])}
        except Exception:
            return None

    def get_estimated_position(self):
        return {"x": self.est_x, "y": self.est_y}

    def get_estimated_xy(self):
        return (self.est_x, self.est_y)

    def sync_position_estimate(self, x, y):
        self.est_x = float(x)
        self.est_y = float(y)

    def sync_estimate_to_machine(self):
        pos = self.get_position()
        if pos is not None:
            self.est_x = pos["x"]
            self.est_y = pos["y"]
            return pos
        return None

    def wait_for_idle(self, timeout=60.0):
        start = time.time()
        while time.time() - start < timeout:
            line = self.get_status_line()
            if line is not None and ("<Idle" in line or line.startswith("<Idle")):
                return True
            time.sleep(0.05)
        raise TimeoutError("Gantry did not return to Idle in time.")

    def read_all_available_lines(self, duration=0.25):
        lines    = []
        end_time = time.time() + duration
        while time.time() < end_time:
            line = self.serial.readline().decode(errors="ignore").strip()
            if line:
                lines.append(line)
        return lines

    def get_setting(self, setting_num):
        self.serial.reset_input_buffer()
        self.serial.write(b"$$\n")
        lines   = self.read_all_available_lines(duration=0.5)
        pattern = re.compile(rf"^\${setting_num}=([^\s]+)")
        for line in lines:
            m = pattern.match(line)
            if m:
                return m.group(1)
        return None

    # ── laser ────────────────────────────────────────────────────────────────

    def fire_pulse(self, power, duration_s, arm_delay_s=0.10):
        power      = max(0, min(1000, int(power)))
        duration_s = max(0.001, float(duration_s))
        print(f"Laser pulse: S{power}, {duration_s:.3f}s")
        if arm_delay_s > 0:
            time.sleep(arm_delay_s)
        self.serial.reset_input_buffer()
        self._send_and_wait("$32=0")
        self._send_and_wait(f"M3 S{power}")
        self._send_and_wait(f"G4 P{duration_s:.3f}")
        self._send_and_wait("M5")
        self._send_and_wait("$32=1")

    # ── stepper hold ──────────────────────────────────────────────────────────

    def enable_stepper_hold(self):
        """
        Set $1=255 so steppers stay energised indefinitely.
        Cached: if already active in this session, the GRBL command is
        still sent once (cheap) but the 500ms get_setting() verification
        is skipped – saving ~650ms per call.
        """
        self.send_raw("$1=255")
        if not self._stepper_hold_active:
            # Verify only on first call (startup / after home)
            time.sleep(0.15)
            val = self.get_setting(1)
            print(f"[Gantry] Stepper hold enabled ($1={val})")
            self._stepper_hold_active = True

    def disable_stepper_hold(self, idle_delay_ms=25):
        print(f"[Gantry] Releasing stepper hold ($1={int(idle_delay_ms)})...")
        self.send_raw(f"$1={int(idle_delay_ms)}")
        self._stepper_hold_active = False
        time.sleep(0.20)
        try:
            self.wait_for_idle(timeout=2.0)
        except Exception:
            pass
        self.get_status_line()
        time.sleep(max(0.35, idle_delay_ms / 1000.0 + 0.10))
        val = self.get_setting(1)
        print(f"[Gantry] Verified $1={val}")

    # ── motion ────────────────────────────────────────────────────────────────

    def home(self):
        print("[Gantry] Homing...")
        self.send_raw("$H")
        self.wait_for_idle()
        self.est_x = 0.0
        self.est_y = 0.0
        self._stepper_hold_active = False   # homing resets GRBL settings
        if self.hold_steppers:
            self.enable_stepper_hold()
        print("[Gantry] Homing complete.")

    def move_absolute(self, x, y, feed=12000):
        """
        Blocking absolute move.

        Steps:
          1. Jog-cancel + buffer flush  — fine-align queues many small jogs; if
             they're still in GRBL's buffer when G1 arrives, the move either
             executes from the wrong position or is dropped entirely.  Sending
             \x85 (real-time jog cancel) and flushing before the G1 fixes this.
          2. G90 + G1 + wait_for_idle  — standard move.
          3. enable_stepper_hold()     — fast (~10 ms) after first call; locks
             carriage against gravity on a slanted frame.
          4. 100 ms mechanical settle  — replaces the accidental settle that the
             old slow stepper-hold verification provided (~650 ms).  Prevents
             motion-blur during the first re-ID camera read.
          5. get_position() sync       — verifies arrival and warns on real
             positioning errors (> 5 mm after settle).
        """
        # 1. Cancel any pending jog motion and drain the serial response buffer.
        self.serial.write(b"\x85")   # real-time jog cancel — no 'ok' response
        time.sleep(0.02)
        self.serial.reset_input_buffer()

        # 2. Move
        self.send_raw("G90")
        self.send_raw(f"G1 X{x:.3f} Y{y:.3f} F{int(feed)}")
        self.wait_for_idle()

        # 3. Lock steppers (fast after first call — cached flag)
        if self.hold_steppers:
            self.enable_stepper_hold()

        # 4. Mechanical settle so cameras see a still image
        time.sleep(0.10)

        # 5. Sync and sanity-check position
        pos = self.get_position()
        if pos is not None:
            drift = ((pos["x"] - x) ** 2 + (pos["y"] - y) ** 2) ** 0.5
            if drift > 5.0:
                # > 5 mm after settle = real positioning problem (not fine-align offset)
                print(f"[Gantry] ⚠ Positioning error {drift:.1f} mm "
                      f"(commanded {x:.1f},{y:.1f}  actual {pos['x']:.1f},{pos['y']:.1f})")
            self.est_x = pos["x"]
            self.est_y = pos["y"]
        else:
            self.est_x = float(x)
            self.est_y = float(y)

    def jog(self, dx, dy, feed=5000):
        cmd = f"$J=G91 X{dx:.3f} Y{dy:.3f} F{int(feed)}"
        self.serial.write(f"{cmd}\n".encode())
        self.est_x += float(dx)
        self.est_y += float(dy)

    # ── shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        for cmd in (lambda: self.send_raw("M5"),
                    lambda: self.stop()):
            try:
                cmd()
            except Exception:
                pass
        if self.hold_steppers:
            try:
                self.disable_stepper_hold(self.default_idle_delay_ms)
                time.sleep(0.50)
                print("[Gantry] Soft-resetting GRBL...")
                self.soft_reset()
            except Exception:
                pass

    def close(self):
        try:
            self.shutdown()
        except Exception:
            pass
        try:
            self.serial.close()
        except Exception:
            pass