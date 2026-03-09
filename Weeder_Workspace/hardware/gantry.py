import serial
import time


class Gantry:
    def __init__(self, port, baudrate=115200, timeout=0.1):
        self.serial = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2.0)
        self.serial.reset_input_buffer()

    def send_raw(self, cmd):
        self.serial.write(f"{cmd.strip()}\n".encode())
        return self.serial.readline().decode().strip()

    def stop(self):
        self.serial.write(b"\x85")

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

            return {
                "x": float(coords[0]),
                "y": float(coords[1]),
            }
        except:
            return None

    def wait_for_idle(self, timeout=60.0):
        start = time.time()

        while time.time() - start < timeout:
            line = self.get_status_line()
            if line is not None and ("<Idle" in line or line.startswith("<Idle")):
                return True
            time.sleep(0.05)

        raise TimeoutError("Gantry did not return to Idle in time.")

    def home(self):
        print("\n=== HOMING GANTRY ===")
        self.send_raw("$H")
        self.wait_for_idle()
        print("Homing complete.")

    def move_absolute(self, x, y, feed=12000):
        print(f"\n=== MOVING TO ABSOLUTE POSITION X:{x} Y:{y} ===")
        self.send_raw("G90")
        self.send_raw(f"G1 X{x:.3f} Y{y:.3f} F{int(feed)}")
        self.wait_for_idle()
        print("Move complete.")

    def jog(self, dx, dy, feed=5000):
        cmd = f"$J=G91 X{dx:.3f} Y{dy:.3f} F{int(feed)}"
        self.serial.write(f"{cmd}\n".encode())

    def close(self):
        try:
            self.send_raw("M5")
            self.stop()
        except:
            pass
        self.serial.close()