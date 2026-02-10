import time
import serial

# === CONFIGURATION ===
PORT = "/dev/ttyUSB0"
BAUD = 115200
ITERATIONS = 10 
TRAVEL_F = 8000


# Bed Dimensions for Longer B1 (450x440)
# Center point is approx X225, Y220
# For a 300x300 square: 225 +/- 150 and 220 +/- 150
X_MIN, X_MAX = 75, 375
Y_MIN, Y_MAX = 70, 370

def send(ser, line):
    """Send a G-code line and wait for acknowledgement."""
    ser.write((line + "\r\n").encode())
    ser.flush()
    while True:
        resp = ser.readline()
        if not resp:
            time.sleep(0.001)
            continue
        if b"ok" in resp or b"error" in resp:
            break

def wait_for_idle(ser):
    """Poll GRBL until it reports Idle."""
    while True:
        ser.write(b"?\n")
        line = ser.readline().decode(errors="ignore").strip()
        if "Idle" in line:
            break
        time.sleep(0.1)

def connect():
    """Initialize connection and unlock machine."""
    print(f"Connecting to Longer B1...")
    try:
        ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.2)
        time.sleep(2) 
        ser.reset_input_buffer()
        send(ser, "$X") # Unlock
        send(ser, "G21") # Metric
        send(ser, "G90") # Absolute
        return ser
    except Exception as e:
        print(f"Connection failed: {e}")
        exit()

def run_300mm_test():
    ser = connect()
    
    try:
        # 1. Home
        print("Homing...")
        send(ser, "$H")
        wait_for_idle(ser)
        
        # 2. Move to start corner
        print(f"Moving to start position (X{X_MIN}, Y{Y_MIN})...")
        send(ser, f"G0 X{X_MIN} Y{Y_MIN} F{TRAVEL_F}")
        wait_for_idle(ser)

        print(f"--- STARTING 300mm RAPID TEST ({ITERATIONS} LOOPS) ---")
        print("Maximum Velocity and Acceleration Active.")
        
        for i in range(ITERATIONS):
            # Rapid moves (G0) use internal machine max speed
            send(ser, f"G0 X{X_MAX} Y{Y_MIN} F{TRAVEL_F}")
            send(ser, f"G0 X{X_MAX} Y{Y_MAX} F{TRAVEL_F}")
            send(ser, f"G0 X{X_MIN} Y{Y_MAX} F{TRAVEL_F}")
            send(ser, f"G0 X{X_MIN} Y{Y_MIN} F{TRAVEL_F}")

            # Brief status update every few loops
            if (i + 1) % 2 == 0:
                print(f" Loop {i+1} complete...")

        wait_for_idle(ser)
        print("Test complete.")

    except KeyboardInterrupt:
        print("\nStopping...")
        send(ser, "M5")
    finally:
        print("Returning home...")
        send(ser, "G0 X0 Y0")
        wait_for_idle(ser)
        ser.close()
        print("Done.")

if __name__ == "__main__":
    run_300mm_test()