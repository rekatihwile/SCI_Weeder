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

        if self.hold_steppers:
            self.enable_stepper_hold()

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
                raise RuntimeError(f"GRBL returned: {resp} for command: {cmd}")

    def stop(self):
        self.serial.write(b"\x85")

    def soft_reset(self):
        self.serial.write(b"\x18")
        time.sleep(0.5)
        self.serial.reset_input_buffer()

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
            parts = content.split("|")
            mpos_part = [p for p in parts if p.startswith("MPos:")][0]
            coords = mpos_part.replace("MPos:", "").split(",")
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
        lines = []
        end_time = time.time() + duration
        while time.time() < end_time:
            line = self.serial.readline().decode(errors="ignore").strip()
            if line:
                lines.append(line)
        return lines

    def get_setting(self, setting_num):
        self.serial.reset_input_buffer()
        self.serial.write(b"$$\n")
        lines = self.read_all_available_lines(duration=0.5)
        pattern = re.compile(rf"^\${setting_num}=([^\s]+)")
        for line in lines:
            m = pattern.match(line)
            if m:
                return m.group(1)
        return None

    def fire_pulse(self, power, duration_s, arm_delay_s=0.10):
        power = max(0, min(1000, int(power)))
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

    def enable_stepper_hold(self):
        print("Enabling stepper hold ($1=255)...")
        self.send_raw("$1=255")
        time.sleep(0.15)
        val = self.get_setting(1)
        print(f"Verified $1={val}")

    def disable_stepper_hold(self, idle_delay_ms=25):
        print(f"Disabling continuous stepper hold ($1={int(idle_delay_ms)})...")
        self.send_raw(f"$1={int(idle_delay_ms)}")
        time.sleep(0.20)
        try:
            self.wait_for_idle(timeout=2.0)
        except Exception:
            pass
        self.get_status_line()
        time.sleep(max(0.35, idle_delay_ms / 1000.0 + 0.10))
        val = self.get_setting(1)
        print(f"Verified $1={val}")

    def shutdown(self):
        try:
            self.send_raw("M5")
        except Exception:
            pass

        try:
            self.stop()
        except Exception:
            pass

        try:
            if self.hold_steppers:
                self.disable_stepper_hold(self.default_idle_delay_ms)
                time.sleep(0.50)
                print("Soft-resetting GRBL so the release takes effect immediately...")
                self.soft_reset()
        except Exception:
            pass

    def home(self):
        print("\n=== HOMING GANTRY ===")
        self.send_raw("$H")
        self.wait_for_idle()
        self.est_x = 0.0
        self.est_y = 0.0
        if self.hold_steppers:
            self.enable_stepper_hold()
        print("Homing complete.")

    def move_absolute(self, x, y, feed=12000):
        print(f"\n=== MOVING TO ABSOLUTE POSITION X:{x} Y:{y} ===")
        self.send_raw("G90")
        self.send_raw(f"G1 X{x:.3f} Y{y:.3f} F{int(feed)}")
        self.wait_for_idle()
        self.est_x = float(x)
        self.est_y = float(y)
        if self.hold_steppers:
            self.enable_stepper_hold()
        print("Move complete.")

    def jog(self, dx, dy, feed=5000):
        cmd = f"$J=G91 X{dx:.3f} Y{dy:.3f} F{int(feed)}"
        self.serial.write(f"{cmd}\n".encode())
        self.est_x += float(dx)
        self.est_y += float(dy)

    def close(self):
        try:
            self.shutdown()
        except Exception:
            pass
        try:
            self.serial.close()
        except Exception:
            pass