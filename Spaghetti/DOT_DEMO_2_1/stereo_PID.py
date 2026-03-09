import cv2
import numpy as np
import time
from helpers import B1LaserController

# --- STEREO PD CONFIGURATION ---
PORT = '/dev/ttyUSB0'
W, H = 640, 480
TARGET_Y = 240
UPDATE_INTERVAL = 0.033 

# Tuning Gains
Kp = 36.0  
Kd = 45.0  
K_ratio = 1.0  
DEADZONE = 3
MAX_SPEED = 10000

# Cluster Tracking Settings
TRACK_WINDOW = 12
MIN_POINTS = 8

class B1StereoUltimateServo:
    def __init__(self):
        self.laser = B1LaserController(PORT)
        # ID 0 = Right, ID 2 = Left
        self.cap_R = cv2.VideoCapture(0)
        self.cap_L = cv2.VideoCapture(2)
        
        self.target_pts_L = None
        self.target_pts_R = None
        self.old_gray_L = None
        self.old_gray_R = None
        
        self.last_mag = 0.0
        self.last_vec = np.array([0.0, 0.0])

    def get_cluster(self, x, y):
        pts = []
        for i in range(-TRACK_WINDOW, TRACK_WINDOW, 4):
            for j in range(-TRACK_WINDOW, TRACK_WINDOW, 4):
                pts.append([float(x + i), float(y + j)])
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def start(self):
        self.laser.set_acceleration(2000) 
        self.laser.home()
        time.sleep(1)
        self.laser.send_raw("G90\nG1 X225 Y220 F4000")

        cv2.namedWindow("Left Camera (ID 2)")
        cv2.namedWindow("Right Camera (ID 0)")
        cv2.setMouseCallback("Left Camera (ID 2)", self.on_mouse_L)
        cv2.setMouseCallback("Right Camera (ID 0)", self.on_mouse_R)

        print("\n--- STEREO READY ---")
        print("1. Click target on Left Screen")
        print("2. Click target on Right Screen")
        print("Press 'c' to clear and re-select | Press 'ESC' to quit")

        while True:
            t_start = time.time()
            retL, frameL = self.cap_L.read()
            retR, frameR = self.cap_R.read()
            if not (retL and retR): break
            
            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

            # --- TRACKING AND CONTROL LOGIC ---
            if self.target_pts_L is not None and self.target_pts_R is not None and self.old_gray_L is not None:
                # Track Clusters
                new_L, stL, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_L, grayL, self.target_pts_L, None)
                new_R, stR, _ = cv2.calcOpticalFlowPyrLK(self.old_gray_R, grayR, self.target_pts_R, None)
                
                # Filter and Reshape to (N, 2) to avoid IndexError
                vptsL = new_L[stL.flatten() == 1].reshape(-1, 2)
                vptsR = new_R[stR.flatten() == 1].reshape(-1, 2)

                if len(vptsL) >= 4 and len(vptsR) >= 4:
                    # Robust Median Calculation
                    xl, yl = np.median(vptsL[:, 0]), np.median(vptsL[:, 1])
                    xr, yr = np.median(vptsR[:, 0]), np.median(vptsR[:, 1])
                    
                    # Auto-Refill logic
                    if len(vptsL) < MIN_POINTS:
                        self.target_pts_L = self.get_cluster(xl, yl)
                    else:
                        self.target_pts_L = vptsL.reshape(-1, 1, 2)
                        
                    if len(vptsR) < MIN_POINTS:
                        self.target_pts_R = self.get_cluster(xr, yr)
                    else:
                        self.target_pts_R = vptsR.reshape(-1, 1, 2)

                    # --- PD Logic ---
                    err_x = xl - (W - xr)
                    err_y = ((yl + yr) / 2) - TARGET_Y
                    err_x_scaled = err_x * K_ratio
                    mag = np.sqrt(err_x_scaled**2 + err_y**2)

                    if mag > DEADZONE:
                        unit_vec = np.array([err_x_scaled / mag, -err_y / mag])
                        mag_diff = mag - self.last_mag
                        feed = (mag * Kp) + (mag_diff * Kd)
                        
                        dot = np.dot(unit_vec, self.last_vec) if self.last_vec.any() else 1.0
                        turn_factor = np.clip((dot + 1) / 2, 0.2, 1.0)
                        final_feed = np.clip(feed * turn_factor, 0, MAX_SPEED)

                        if final_feed > 50:
                            step = 1.0 * (1 + (final_feed / 2500))
                            self.laser.jog_clear(unit_vec[0]*step, unit_vec[1]*step, final_feed)
                        
                        self.last_mag = mag
                        self.last_vec = unit_vec
                    else:
                        self.laser.stop()
                        self.last_mag = 0
                else:
                    self.laser.stop()

            # --- VISUALS ---
            cv2.drawMarker(frameL, (W//2, H//2), (0,0,255), cv2.MARKER_CROSS, 15, 1)
            cv2.drawMarker(frameR, (W//2, H//2), (0,0,255), cv2.MARKER_CROSS, 15, 1)
            
            # Show tracked center
            if self.target_pts_L is not None:
                # Use a try-except to prevent crash during transient re-sampling
                try:
                    cur_vptsL = self.target_pts_L.reshape(-1, 2)
                    xl_c, yl_c = np.median(cur_vptsL[:,0]), np.median(cur_vptsL[:,1])
                    cv2.circle(frameL, (int(xl_c), int(yl_c)), 5, (0, 255, 0), -1)
                except: pass
                
            if self.target_pts_R is not None:
                try:
                    cur_vptsR = self.target_pts_R.reshape(-1, 2)
                    xr_c, yr_c = np.median(cur_vptsR[:,0]), np.median(cur_vptsR[:,1])
                    cv2.circle(frameR, (int(xr_c), int(yr_c)), 5, (0, 255, 0), -1)
                except: pass

            cv2.imshow("Left Camera (ID 2)", frameL)
            cv2.imshow("Right Camera (ID 0)", frameR)
            self.old_gray_L, self.old_gray_R = grayL.copy(), grayR.copy()
            
            # Key Handling
            key = cv2.waitKey(max(1, int((UPDATE_INTERVAL - (time.time() - t_start)) * 1000))) & 0xFF
            if key == 27: break
            elif key == ord('c'):
                print("Clearing targets.")
                self.laser.stop()
                self.target_pts_L = None
                self.target_pts_R = None

        self.laser.close()

    def on_mouse_L(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.target_pts_L = self.get_cluster(x, y)
            print(f"Left Set: {x,y}")

    def on_mouse_R(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.target_pts_R = self.get_cluster(x, y)
            print(f"Right Set: {x,y}")

if __name__ == "__main__":
    B1StereoUltimateServo().start()