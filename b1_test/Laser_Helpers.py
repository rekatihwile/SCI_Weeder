import time, serial

# === CONFIG ===
PORT = "/dev/ttyUSB0"
BAUD = 115200
TRAVEL_F = 8000
MAX_POWER = 1000 

def send(ser, line):
    """Send a line and return the full response (ok, error, or ALARM)."""
    ser.write((line + "\r\n").encode())
    ser.flush()
    full_response = []
    while True:
        resp = ser.readline().decode(errors="ignore").strip()
        if not resp:
            continue
        full_response.append(resp)
        # B1 often sends ALARM or error messages before or instead of 'ok'
        if "ok" in resp or "error" in resp or "ALARM" in resp:
            break
    return " | ".join(full_response)

def wait_for_idle(ser, poll_interval=0.1):
    """Poll GRBL and return the status line if an Alarm occurs."""
    while True:
        ser.write(b"?\n")
        line = ser.readline().decode(errors="ignore").strip()
        if "Idle" in line:
            break
        if "Alarm" in line or "ALARM" in line:
            return line # Return the alarm status immediately
        time.sleep(poll_interval)
    return "Idle"

def connect():
    ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.5, write_timeout=0.5)
    time.sleep(2) # Increased for B1 controller boot-up
    ser.reset_input_buffer()
    # Capture initial boot string
    print(f"Boot Message: {ser.read_all().decode(errors='ignore')}")
    send(ser, "$X")       
    send(ser, "G21")      
    send(ser, "G90")      
    return ser

def move_to(ser, x, y, feedrate=8000):
    """Moves and returns the response to catch mid-move Alarms."""
    cmd_resp = send(ser, f"G0 X{x:.3f} Y{y:.3f} F{feedrate}")
    idle_resp = wait_for_idle(ser)
    return f"{cmd_resp} | {idle_resp}"

def burn(ser, power=1000, duration=0.05):
    send(ser, "$32=0")  
    send(ser, f"M3 S{power}")
    send(ser, f"G4 P{duration:.3f}")
    send(ser, "M5")
    send(ser, "$32=1")  
    print(f"Burn complete ({duration:.3f}s)")

def close(ser):
    send(ser, "M5")
    ser.close()
    print("Serial connection closed safely.")