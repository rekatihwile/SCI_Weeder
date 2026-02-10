#!/usr/bin/env python3
import cv2
import numpy as np
import time

# --- CONFIGURATION ---
CAM_ID = 0
W, H = 1280, 720 # Adjusted to a standard HD res, change back to 640/480 if needed

# LK Parameters
LK_PARAMS = dict(winSize=(21, 21), 
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class MultiLKTracker:
    def __init__(self):
        self.target_pts = None   # Shape will be (N, 1, 2)
        self.prev_gray = None   
        self.tracking = False
        print("✅ Multi-Point Lucas-Kanade Tracker Initialized")

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            new_pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            
            if self.target_pts is None:
                self.target_pts = new_pt
            else:
                # Append the new point to the existing array of tracked points
                self.target_pts = np.vstack([self.target_pts, new_pt])
            
            self.tracking = True
            print(f"🎯 Point Added: {x}, {y} | Total Points: {len(self.target_pts)}")

    def update(self, frame):
        vis_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.tracking and self.target_pts is not None:
            if self.prev_gray is not None:
                # Calculate flow for ALL points at once
                new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.target_pts, None, **LK_PARAMS
                )

                # Filter points: only keep those where status == 1 (successfully tracked)
                # status.ravel() creates a boolean mask
                good_new = new_pts[status == 1].reshape(-1, 1, 2)
                
                if len(good_new) > 0:
                    self.target_pts = good_new
                    # DRAWING loop for all active points
                    for i, pt in enumerate(self.target_pts):
                        x, y = pt.ravel()
                        cv2.circle(vis_frame, (int(x), int(y)), 8, (255, 0, 0), -1)
                        cv2.putText(vis_frame, f"ID:{i}", (int(x)+10, int(y)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    self.tracking = False
                    self.target_pts = None
                    print("❌ All tracks lost")
            
        self.prev_gray = gray
        return vis_frame

def main():
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    
    # Standard Manual Settings
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      
    cap.set(cv2.CAP_PROP_EXPOSURE, 550)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    tracker = MultiLKTracker()
    cv2.namedWindow("Multi-Point Laser Tracker")
    cv2.setMouseCallback("Multi-Point Laser Tracker", tracker.handle_click)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        processed_frame = tracker.update(frame)

        # FPS UI
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 40), 1, 2, (0, 255, 0), 2)
        cv2.putText(processed_frame, "Click to add points | 'c' to clear | 'q' to quit", 
                    (20, H-20), 1, 1.2, (255, 255, 255), 1)
        
        cv2.imshow("Multi-Point Laser Tracker", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            tracker.tracking = False
            tracker.target_pts = None
            print("🧹 All Points Cleared")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()