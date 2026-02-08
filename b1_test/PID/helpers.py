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

    def set_acceleration(self, accel_value=2500):
        """
        Sets the Grbl acceleration for X ($120) and Y ($121).
        Higher = Snappier but more vibration.
        Lower = Smoother but more 'lag'.
        """
        print(f"Tuning firmware acceleration to {accel_value}...")
        self.send_raw(f"$120={accel_value}")
        self.send_raw(f"$121={accel_value}")
        self.send_raw("$11=0.010") # Junction Deviation for smoother motion

    def home(self):
        print("Homing...")
        self.send_raw("$H")
        
    def jog(self, dx, dy, feed):
        """
        Standard Jog. No cancel (\x85) used here to maintain 
        the planner's smooth acceleration ramps.
        """
        cmd = f"$J=G91 X{dx:.3f} Y{dy:.3f} F{int(feed)}"
        with self.lock:
            self.serial.write(f"{cmd}\n".encode())

    def stop(self):
        """Only use this when we actually want to HALT (e.g. Deadzone)."""
        with self.lock:
            self.serial.write(b'\x85') 
    def jog_clear(self, dx, dy, feed):
        """
        Clears any residual motion and instantly starts a new vector.
        """
        with self.lock:
            # 1. Real-time Jog Cancel: Tells Grbl to stop the current jog move NOW.
            self.serial.write(b'\x85')
            
            # 2. Immediately send the new command. 
            # Because we used \x85, the buffer is ready for fresh input.
            cmd = f"$J=G91 X{dx:.3f} Y{dy:.3f} F{int(feed)}\n"
            self.serial.write(cmd.encode())

    def close(self):
        self.stop()
        self.serial.close()