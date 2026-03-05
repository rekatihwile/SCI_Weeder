import time, serial

# === CONFIG ===
PORT = "COM6"
BAUD = 115200
TRAVEL_F = 8000
MAX_POWER = 1000  # corresponds to $30=1000


def send(ser, line):
    """Send a G-code line and wait for GRBL to acknowledge 'ok' or 'error'."""
    ser.write((line + "\r\n").encode())
    ser.flush()
    while True:
        resp = ser.readline()
        if not resp:
            time.sleep(0.001)
            continue
        if b"ok" in resp or b"error" in resp:
            break

def wait_for_idle(ser, poll_interval=0.1):
    """Poll GRBL until it reports Idle (movement finished)."""
    while True:
        ser.write(b"?\n")
        line = ser.readline().decode(errors="ignore").strip()
        if "Idle" in line:
            break
        time.sleep(poll_interval)

# === CONNECTION SETUP ===
def connect():
    """Open a serial connection and configure GRBL settings."""
    ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.2, write_timeout=0.2)
    time.sleep(1)
    ser.reset_input_buffer()
    send(ser, "$X")       # unlock
    send(ser, "G21")      # mm
    send(ser, "G90")      # absolute
    send(ser, "$30=1000") # power scaling
    send(ser, "S0")
    send(ser, "M5")       # laser off
    return ser

# === MOVE FUNCTION ===
def move_to(ser, x, y, feedrate=8000):
    """Move the laser head and wait until motion completes."""
    send(ser, f"G0 X{x:.3f} Y{y:.3f} F{feedrate}")
    wait_for_idle(ser)
    print(f"Move complete: X{x:.3f}, Y{y:.3f}")


# === LASER BURN FUNCTION ===
def burn(ser, power=1000, duration=0.05):
    """Fire laser for a given duration, then wait until burn completes."""
    send(ser, "$32=0")  # disable laser mode for dwell
    send(ser, f"M3 S{power}")
    send(ser, f"G4 P{duration:.3f}")
    send(ser, "M5")
    send(ser, "$32=1")  # restore laser mode
    print(f"Burn complete ({duration:.3f}s)")

# === DISCONNECT ===
def close(ser):
    """Safely turn off laser and close serial connection."""
    send(ser, "M5")
    ser.close()
    print("Serial connection closed safely.")
