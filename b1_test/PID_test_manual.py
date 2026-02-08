import cv2
import numpy as np
import time
from SCI_Weeder.b1_test.PID.helpers import LaserHelper

# ===========================
# Configuration
# ===========================
W, H = 640, 480
CAM_ID = 0
PORT = "/dev/ttyUSB0"
BAUD = 115200

CENTER_X_MM = 225.0
CENTER_Y_MM = 220.0

# PID Gains - Tuned for snappier braking
KP_X, KD_X = 0.20, 0.12  # Lowered P, Increased D to act as a stronger brake
KP_Y, KD_Y = 0.18, 0.10
DEADZONE = 4             # Slightly wider deadzone for stability 

class SingleCamDirectionalWeeder:
    def __init__(self):
        print("🔗 Connecting to Longer B1...")
        self.laser = LaserHelper(PORT, BAUD)
        self.prepare_hardware()
        
        self.cap = cv2.VideoCapture(CAM_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.target_pt = None
        self.old_gray = None
        self.prev_err_x, self.prev_err_y = 0, 0
        self.on_target_since = None
        self.is_moving = False # State tracker to prevent redundant stops

    def prepare_hardware(self):
        print("🏠 Homing...")
        self.laser.ser.write(b"$X\n") 
        time.sleep(0.5)
        self.laser.send_command("$H") 
        print(f"🚀 Centering to ({CENTER_X_MM}, {CENTER_Y_MM})...")
        self.laser.move_to(CENTER_X_MM, CENTER_Y_MM, speed=8000)
        print("✅ Ready.")

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.target_pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            self.prev_err_x, self.prev_err_y = 0, 0
            self.on_target_since = None

    def process_frames(self):
        ret, frame = self.cap.read()
        if not ret: return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis_frame = frame.copy()

        if self.target_pt is not None and self.old_gray is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.target_pt, None, **self.lk_params)
            
            if status[0] == 1:
                self.target_pt = new_pts
                tx, ty = new_pts.ravel()
                
                err_x = tx - (W / 2)
                err_y = ty - (H / 2)

                # --- Active Braking Logic ---
                if abs(err_x) <= DEADZONE and abs(err_y) <= DEADZONE:
                    if self.is_moving:
                        self.laser.stop_motion() # Send 0x85 to freeze motors
                        self.is_moving = False
                        print("🛑 BRAKE: Within Deadzone")
                    
                    # Check if stable enough to fire
                    if self.on_target_since is None: self.on_target_since = time.time()
                    if time.time() - self.on_target_since > 0.4:
                        print("🎯 LOCKED. ZAPPING.")
                        self.laser.burn(power=1000, duration=0.2)
                        self.target_pt = None 
                else:
                    # --- PD Calculation ---
                    out_x = (KP_X * err_x) + (KD_X * (err_x - self.prev_err_x))
                    out_y = (KP_Y * err_y) + (KD_Y * (err_y - self.prev_err_y))
                    
                    # Update movement
                    self.laser.send_jog(out_x, -out_y, 2000)
                    self.is_moving = True
                    self.on_target_since = None # Reset fire timer if we moved

                self.prev_err_x, self.prev_err_y = err_x, err_y
                cv2.circle(vis_frame, (int(tx), int(ty)), 8, (0, 255, 0), -1)
            else:
                if self.is_moving:
                    self.laser.stop_motion()
                    self.is_moving = False
                self.target_pt = None

        # UI Overlays
        cv2.line(vis_frame, (W//2, 0), (W//2, H), (255, 0, 0), 1)
        cv2.line(vis_frame, (0, H//2), (W, H//2), (255, 0, 0), 1)
        self.old_gray = gray
        cv2.imshow("Laser Control (Cam 0)", vis_frame)

    def run(self):
        cv2.namedWindow("Laser Control (Cam 0)")
        cv2.setMouseCallback("Laser Control (Cam 0)", self.handle_click)
        while True:
            self.process_frames()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('c'): 
                self.laser.stop_motion()
                self.target_pt = None
            if key == ord('r'): self.prepare_hardware()
        self.laser.close()
        self.cap.release()

if __name__ == "__main__":
    app = SingleCamDirectionalWeeder()
    app.run()