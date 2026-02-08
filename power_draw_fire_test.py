import time
import serial

# === CONFIG ===
PORT = "/dev/ttyUSB0"
BAUD = 115200
X_CENTER, Y_CENTER = 225, 220
BURSTS = 30
# We use 1000ms instead of 1.0s to avoid firmware unit confusion
BURN_TIME = 5
PAUSE_TIME = 2.0 
MAX_POWER = 1000 

def send(ser, line):
    """Send command and wait for 'ok' with a clean buffer."""
    ser.reset_input_buffer() # Clear any noise/latency before sending
    ser.write((line + "\r\n").encode())
    while True:
        resp = ser.readline()
        if b"ok" in resp:
            break
        if b"error" in resp:
            print(f"!!! GRBL Error: {resp.decode().strip()}")
            break

def wait_for_idle(ser):
    """Wait for the 'Idle' state confirmation."""
    while True:
        ser.write(b"?")
        status = ser.readline().decode(errors="ignore")
        if "Idle" in status:
            break
        time.sleep(0.1)

def run_test():
    try:
        # Increased timeout to 1s to handle power-spike lag
        ser = serial.Serial(PORT, baudrate=BAUD, timeout=1.0)
        time.sleep(2)
        
        # 1. Initialization
        print("Initializing & Homing...")
        send(ser, "$X")
        send(ser, "G21")
        send(ser, "G90")
        send(ser, "$H")
        wait_for_idle(ser)
        
        # 2. Move to position (Standard Laser Mode ON)
        print(f"Moving to center...")
        send(ser, "$32=1") 
        send(ser, f"G0 X{X_CENTER} Y{Y_CENTER}")
        wait_for_idle(ser)

        # 3. Burst Loop
        print("Disabling Laser Mode for stationary firing...")
        send(ser, "$32=0") 

        for i in range(BURSTS):
            print(f"[{i+1}/{BURSTS}] Firing 1000ms...")
            
            # Fire
            send(ser, f"M3 S{MAX_POWER}")
            # Use G4 P1 (standardizes to ms across most firmwares)
            send(ser, f"G4 P{BURN_TIME}")
            # Turn off
            send(ser, "M5")
            
            # Add a tiny delay here so GRBL can process the M5 before the Python sleep
            time.sleep(0.1) 
            time.sleep(PAUSE_TIME)

    except KeyboardInterrupt:
        print("\nEmergency Stop.")
    finally:
        send(ser, "M5")
        send(ser, "$32=1")
        print("Returning Home...")
        send(ser, "G0 X0 Y0")
        wait_for_idle(ser)
        ser.close()

if __name__ == "__main__":
    run_test()