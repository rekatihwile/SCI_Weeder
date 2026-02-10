import serial
import time
import threading

class B1LaserController:
    def __init__(self, port, baudrate=115200, timeout=0.1):
        self.serial = serial.Serial(port, baudrate, timeout=timeout)
        self.lock = threading.Lock()
        
    def send_raw(self, cmd):
        with self.lock:
            self.serial.write(f"{cmd.strip()}\n".encode())
            return self.serial.readline().decode().strip()

    # --- NEW FIRE FUNCTIONS ---

    def fire_low(self):
        """Fires laser with a hard state-clear to prevent Jog-mode lock."""
        print("🔥 Firing Low (1%)...")
        with self.lock:
            # 1. Clear any pending jog state/buffer
            self.serial.write(b"\x85") # Jog Cancel
            time.sleep(0.05)
            self.serial.reset_input_buffer()
            
            # 2. Force G-Code Mode and Fire
            cmds = [
                "G1 F10",        # Switch to Motion State
                "M3 S10",         # Laser On 1%
                "G1 G91 X0 Y0",   # Trigger PWM
                "G4 P1.0",        # Dwell
                "M5",             # Laser Off
                "G90"             # Back to Absolute
            ]
            for c in cmds:
                self.serial.write(f"{c}\n".encode())
                # Wait for 'ok' for each critical fire command
                for _ in range(10):
                    if b"ok" in self.serial.readline(): break

    def fire_high(self):
        """Fires laser at 100% power using a Zero-Distance G1 move."""
        print("☢️ FIRING HIGH (100%)...")
        self.send_raw("M3 S10")
        self.send_raw("G1 G91 X0 Y0 F100")
        self.send_raw("G4 P1.0")
        self.send_raw("M5")
        self.send_raw("G90")

    def spiral_burn(self, sx, sy, radius=7.0, steps=20, speed=2500):
        """
        Total Time: ~1.5s
        - 0.5s: Stationary Burn @ F10
        - 1.0s: Fast Spiral Move @ F2500
        """
        import numpy as np
        print(f"🔥 Rapid Spiral Burn: 0.5s Static + 1.0s Motion")
        
        with self.lock:
            # 1. Prep Buffer
            self.serial.write(b"\x85") 
            time.sleep(0.02)
            self.serial.reset_input_buffer()

            # 2. 0.5s STATIONARY (The "Core Hit")
            setup_cmds = [
                "G1 F10",           # Movement state entry
                "M3 S10",          # High Power (85%)
                "G1 G91 X0 Y0",     # Trigger PWM
                "G4 P0.2",          # Hard Dwell 0.5s
                "G90"               # Absolute mode for spiral
            ]
            for c in setup_cmds:
                self.serial.write(f"{c}\n".encode())
                for _ in range(5):
                    if b"ok" in self.serial.readline(): break
            
            # 3. 1.0s SPIRAL (The "Leaf Destruction")
            # Reduced steps to 20 to prevent serial saturation at high speed
            for i in range(1, steps + 1):
                angle = 0.8 * i
                r = (radius / steps) * i
                tx = sx + (r * np.cos(angle))
                ty = sy + (r * np.sin(angle))
                # Sending F2500 (~41mm/s) to cover the spiral path in ~1s
                self.serial.write(f"G1 X{tx:.2f} Y{ty:.2f} F{speed}\n".encode())

            # 4. KILL & RETURN
            self.serial.write(b"M5\n") 
            self.serial.write(f"G1 X{sx} Y{sy} F12000\n".encode())
            
            # Short wait for gantry to physically finish the return
            time.sleep(0.5)

    # --- EXISTING HELPER METHODS ---

    def set_acceleration(self, accel_value=2500):
        print(f"Tuning firmware acceleration to {accel_value}...")
        self.send_raw(f"$120={accel_value}")
        self.send_raw(f"$121={accel_value}")
        self.send_raw("$11=0.010") 

    def home(self):
        print("Homing...")
        self.send_raw("$H")
        
    def jog(self, dx, dy, feed):
        cmd = f"$J=G91 X{dx:.3f} Y{dy:.3f} F{int(feed)}"
        with self.lock:
            self.serial.write(f"{cmd}\n".encode())

    def stop(self):
        with self.lock:
            self.serial.write(b'\x85') 

    def jog_clear(self, dx, dy, feed):
        with self.lock:
            self.serial.write(b'\x85')
            cmd = f"$J=G91 X{dx:.3f} Y{dy:.3f} F{int(feed)}\n"
            self.serial.write(cmd.encode())

    def update_status(self):
        with self.lock:
            self.serial.reset_input_buffer()
            self.serial.write(b'?')
            for _ in range(5): 
                line = self.serial.readline().decode().strip()
                if line.startswith('<') and 'MPos:' in line:
                    try:
                        content = line[1:line.find('>')]
                        parts = content.split('|')
                        mpos_part = [p for p in parts if p.startswith('MPos:')][0]
                        coords = mpos_part.replace('MPos:', '').split(',')
                        return {'x': float(coords[0]), 'y': float(coords[1])}
                    except (IndexError, ValueError):
                        continue
        return None

    def close(self):
        self.send_raw("M5") # Safety: Ensure laser is off
        self.stop()
        self.serial.close()