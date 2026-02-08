import time
import serial

# === CONFIGURATION ===
PORT = "/dev/ttyUSB0"
BAUD = 115200
TRAVEL_F = 8000
MAX_POWER = 1000 
ITERATIONS = 10  # Number of squares to run

# Bed Dimensions for Longer B1
BED_X = 450
BED_Y = 440

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

def connect():
    """Open a serial connection and configure GRBL settings."""
    print(f"Connecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.2, write_timeout=0.2)
        time.sleep(2) # Wait for GRBL to initialize
        ser.reset_input_buffer()
        
        print("Unlocking and initializing...")
        send(ser, "$X")       # Unlock
        send(ser, "G21")      # Metric
        send(ser, "G90")      # Absolute positioning
        send(ser, "$30=1000") # Ensure power scaling matches
        send(ser, "S0")       # Laser power 0
        send(ser, "M5")       # Laser off
        return ser
    except Exception as e:
        print(f"Connection failed: {e}")
        exit()

def run_test():
    ser = connect()
    
    try:
        # 1. Homing
        print("Homing machine... please wait.")
        send(ser, "$H")
        wait_for_idle(ser)
        
        # 2. Calculate Center (100x100mm square)
        # Center of 450x440 is X225, Y220
        x_min, x_max = 175, 275
        y_min, y_max = 170, 270

        print(f"Moving to start position: X{x_min} Y{y_min}")
        send(ser, f"G0 X{x_min} Y{y_min} F{TRAVEL_F}")
        wait_for_idle(ser)

        print(f"Starting power test ({ITERATIONS} loops)...")
        print("Watch your power meter now!")
        time.sleep(1)

        for i in range(ITERATIONS):
            print(f"  Loop {i+1}/{ITERATIONS}")
            # We use G1 to maintain a consistent speed throughout the move
            send(ser, f"G1 X{x_max} Y{y_min} F{TRAVEL_F}")
            send(ser, f"G1 X{x_max} Y{y_max} F{TRAVEL_F}")
            send(ser, f"G1 X{x_min} Y{y_max} F{TRAVEL_F}")
            send(ser, f"G1 X{x_min} Y{y_min} F{TRAVEL_F}")

        wait_for_idle(ser)
        print("Test sequence complete.")

    except KeyboardInterrupt:
        print("\nStop requested by user.")
    finally:
        # 3. Cleanup
        print("Shutting down laser and returning home...")
        send(ser, "M5")
        send(ser, "G0 X0 Y0 F8000")
        wait_for_idle(ser)
        ser.close()
        print("Serial connection closed.")

if __name__ == "__main__":
    run_test()