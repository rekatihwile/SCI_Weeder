import cv2
import numpy as np
import time

# --- CONFIGURATION ---
CAM_ID = 2
W, H = 640, 480 

# LK Parameters
# winSize: Search window size (larger = handles faster motion but more CPU)
# maxLevel: Pyramid levels (3 levels means it looks at 1/8th scale for big moves)
LK_PARAMS = dict(winSize=(21, 21), 
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class LKTracker:
    def __init__(self):
        self.target_pt = None   # Current tracked point (x, y)
        self.prev_gray = None   # Grayscale frame from the previous loop
        self.tracking = False
        print("✅ Lucas-Kanade Optical Flow Tracker Initialized")

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # LK requires points in float32 and shape (N, 1, 2)
            self.target_pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            self.tracking = True
            print(f"🎯 LK Locked Point: {x}, {y}")

    def update(self, frame):
        vis_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.tracking and self.target_pt is not None:
            # We need a previous frame to calculate flow
            if self.prev_gray is not None:
                # Calculate the movement from prev_gray to current gray
                new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.target_pt, None, **LK_PARAMS
                )

                # Check if the tracker successfully found the point (status == 1)
                if status[0][0] == 1:
                    self.target_pt = new_pts
                    x, y = new_pts.ravel()
                    
                    # DRAWING
                    cv2.circle(vis_frame, (int(x), int(y)), 10, (0, 255, 255), -1) # Yellow Dot
                    cv2.putText(vis_frame, "LOCKED", (int(x)+15, int(y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    self.tracking = False
                    print("❌ Track Lost: Feature moved too fast or obscured")
            
        # Update previous frame for the next iteration
        self.prev_gray = gray
        return vis_frame

def main():
    # Setup Camera (matches your Orin/B1 config)
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      
    cap.set(cv2.CAP_PROP_EXPOSURE, 550)
    cap.set(cv2.CAP_PROP_GAIN, 44)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
    cap.set(cv2.CAP_PROP_CONTRAST, 40)
    cap.set(cv2.CAP_PROP_SATURATION, 63)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)            
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4079)

    tracker = LKTracker()
    cv2.namedWindow("LK Laser Tracker")
    cv2.setMouseCallback("LK Laser Tracker", tracker.handle_click)

    prev_time = time.time()
    print("🚀 LK Running. Click a plant to track. 'c' = clear, 'q' = quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        processed_frame = tracker.update(frame)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 40), 1, 2, (0, 255, 0), 2)
        
        cv2.imshow("LK Laser Tracker", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            tracker.tracking = False
            tracker.target_pt = None
            print("🧹 Tracker Cleared")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()