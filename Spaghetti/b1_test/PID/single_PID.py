import cv2
import numpy as np
import time
from helpers import B1LaserController

# --- PD CONFIGURATION ---
PORT = '/dev/ttyUSB0'
TARGET_X, TARGET_Y = 320, 240
UPDATE_INTERVAL = 0.033 # 30Hz: Balanced for serial reliability
HEARTBEAT_MM = 1.0     

# Tuning Gains (B1 Laser specific)
Kp = 18.0*2   # Proportional: The 'Pull' strength
Kd = 35.0   # Derivative: The 'Brake' strength (High for heavy B1 head)
DEADZONE = 2

class B1ZeroLagServo:
    def __init__(self):
        self.laser = B1LaserController(PORT)
        self.cap = cv2.VideoCapture(0)
        self.target_point = None
        self.old_gray = None
        self.last_mag = 0.0
        self.last_vec = np.array([0.0, 0.0])

    def start(self):
        # Tune firmware for snappiness
        self.laser.set_acceleration(2000) 
        self.laser.home()
        time.sleep(1)
        self.laser.send_raw("G90")
        self.laser.send_raw("G1 X225 Y220 F4000")

        cv2.namedWindow("B1 PD Zero-Lag")
        cv2.setMouseCallback("B1 PD Zero-Lag", self.on_mouse)

        while True:
            t_start = time.time()
            ret, frame = self.cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.target_point is not None and self.old_gray is not None:
                pts, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.target_point, None)
                
                if status[0] == 1:
                    self.target_point = pts
                    curr_x, curr_y = pts.ravel()
                    
                    err_x = curr_x - TARGET_X
                    err_y = curr_y - TARGET_Y
                    mag = np.sqrt(err_x**2 + err_y**2)

                    if mag > DEADZONE:
                        # 1. Normalized Vector
                        unit_vec = np.array([err_x / mag, -err_y / mag])
                        
                        # 2. Derivative (Brake)
                        mag_diff = mag - self.last_mag
                        
                        # 3. PD Output (No Floor)
                        feed = (mag * Kp) + (mag_diff * Kd)
                        
                        # 4. Turn Damping (Dot product logic)
                        dot = np.dot(unit_vec, self.last_vec) if self.last_vec.any() else 1.0
                        turn_factor = np.clip((dot + 1) / 2, 0.2, 1.0)
                        
                        final_feed = np.clip(feed * turn_factor, 0, 10000)

                        if final_feed > 50: # Only jog if there's meaningful speed
                            # USE THE NEW BUFFER-CLEARING JOG
                            self.laser.jog_clear(unit_vec[0]*HEARTBEAT_MM, unit_vec[1]*HEARTBEAT_MM, final_feed)
                        
                        self.last_mag = mag
                        self.last_vec = unit_vec
                    else:
                        self.laser.stop()
                        self.last_mag = 0
                        self.last_vec = np.array([0.0, 0.0])

                    cv2.circle(frame, (int(curr_x), int(curr_y)), 5, (0, 255, 0), -1)
                else:
                    self.target_point = None
                    self.laser.stop()

            cv2.drawMarker(frame, (TARGET_X, TARGET_Y), (0,0,255), cv2.MARKER_CROSS, 15, 1)
            cv2.imshow("B1 PD Zero-Lag", frame)
            
            self.old_gray = gray.copy()
            
            # Maintenance of Loop Frequency
            elapsed = time.time() - t_start
            wait = max(1, int((UPDATE_INTERVAL - elapsed) * 1000))
            if cv2.waitKey(wait) & 0xFF == 27: break

        self.laser.close()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.target_point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            self.last_mag = 0.0

if __name__ == "__main__":
    B1ZeroLagServo().start()