import time
from pynput import keyboard
from SCI_Weeder.b1_test.PID.helpers import B1LaserController

# Configuration
PORT = '/dev/ttyUSB1'
JOG_SPEED = 4000       # Velocity
HEARTBEAT_STEP = 1    # Short 10mm "segments"
UPDATE_INTERVAL = 0.05 # 50ms (20Hz) - Sending 10mm every 50ms is plenty fast

class HeartbeatController:
    def __init__(self):
        self.laser = B1LaserController(PORT)
        self.active_keys = set()
        self.running = True

    def start(self):
        self.laser.home()
        time.sleep(1)
        # Move to initial position
        self.laser.send_raw("G90")
        self.laser.send_raw("G1 X225 Y220 F4000")
        
        print("Heartbeat Control: WASD to move (held), Release to stop.")
        
        # We start the listener in the background
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        try:
            while self.running:
                self.update_motion()
                time.sleep(UPDATE_INTERVAL)
        finally:
            self.laser.stop()
            listener.stop()

    def on_press(self, key):
        try:
            k = key.char.lower()
            if k in ['w', 'a', 's', 'd']:
                self.active_keys.add(k)
        except AttributeError:
            if key == keyboard.Key.esc:
                self.running = False

    def on_release(self, key):
        try:
            k = key.char.lower()
            if k in self.active_keys:
                self.active_keys.remove(k)
            
            # When keys are released, we kill the movement immediately 
            # so it doesn't finish the last 10mm segment.
            if not self.active_keys:
                self.laser.stop()
        except:
            pass

    def update_motion(self):
        if not self.active_keys:
            return

        dx, dy = 0, 0
        if 'w' in self.active_keys: dy = HEARTBEAT_STEP
        if 's' in self.active_keys: dy = -HEARTBEAT_STEP
        if 'a' in self.active_keys: dx = -HEARTBEAT_STEP
        if 'd' in self.active_keys: dx = HEARTBEAT_STEP

        if dx != 0 or dy != 0:
            # We keep sending the 10mm move. Grbl's $J= will overwrite
            # the previous 10mm move with a fresh 10mm from 'current position'.
            self.laser.jog(dx, dy, JOG_SPEED)

if __name__ == "__main__":
    ctrl = HeartbeatController()
    ctrl.start()