import cv2
import numpy as np
import time

# --- CONFIGURATION ---
CAM_ID = 1
W, H = 640, 480 
TEMPLATE_SIZE = 40  # Size of the "patch" we are tracking
SEARCH_MARGIN = int(TEMPLATE_SIZE * 2)  # Area around the last point to look in (prevents jumping)

class TemplateTracker:
    def __init__(self):
        self.target_pt = None  # [x, y] center of the template
        self.tracking = False
        self.template = None
        print("✅ Hybrid Template Tracker Initialized")

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Center the click and ensure it stays in bounds
            self.target_pt = [
                max(TEMPLATE_SIZE, min(W - TEMPLATE_SIZE, x)),
                max(TEMPLATE_SIZE, min(H - TEMPLATE_SIZE, y))
            ]
            self.tracking = "INIT"
            print(f"🎯 Locked Target at: {self.target_pt}")

    def update(self, frame):
        vis_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. INITIALIZE: Capture the template patch from the clicked point
        if self.tracking == "INIT":
            tx, ty = int(self.target_pt[0]), int(self.target_pt[1])
            half = TEMPLATE_SIZE // 2
            self.template = gray[ty-half:ty+half, tx-half:tx+half].copy()
            self.tracking = True
            print("📸 Template Captured")

        # 2. TRACK: Find the template in the current frame
        if self.tracking is True and self.template is not None:
            # To stay fast, we only search in a region around the last known point
            tx, ty = int(self.target_pt[0]), int(self.target_pt[1])
            
            # Define Search ROI (Region of Interest)
            x1 = max(0, tx - (TEMPLATE_SIZE // 2) - SEARCH_MARGIN)
            y1 = max(0, ty - (TEMPLATE_SIZE // 2) - SEARCH_MARGIN)
            x2 = min(W, tx + (TEMPLATE_SIZE // 2) + SEARCH_MARGIN)
            y2 = min(H, ty + (TEMPLATE_SIZE // 2) + SEARCH_MARGIN)
            
            search_area = gray[y1:y2, x1:x2]
            
            # Perform Template Matching
            res = cv2.matchTemplate(search_area, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # If the match quality is decent, update position
            if max_val > 0.6:
                # max_loc is relative to search_area; convert back to global frame
                new_x = max_loc[0] + x1 + (TEMPLATE_SIZE // 2)
                new_y = max_loc[1] + y1 + (TEMPLATE_SIZE // 2)
                
                self.target_pt = [new_x, new_y]
                
                # DRAWING: Draw the tracked point and a box
                cv2.circle(vis_frame, (int(new_x), int(new_y)), 7, (0, 0, 255), -1)
                cv2.rectangle(vis_frame, 
                              (int(new_x - TEMPLATE_SIZE//2), int(new_y - TEMPLATE_SIZE//2)),
                              (int(new_x + TEMPLATE_SIZE//2), int(new_y + TEMPLATE_SIZE//2)), 
                              (0, 255, 0), 2)
            else:
                cv2.putText(vis_frame, "LOST TRACK", (W//2-50, H//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return vis_frame

def main():
    # Attempt to force 60 FPS
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 60)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Auto ON
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 1000) 
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual OFF
    cap.set(cv2.CAP_PROP_EXPOSURE, 200)      
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
    cap.set(cv2.CAP_PROP_SATURATION, 70)

    tracker = TemplateTracker()
    cv2.namedWindow("Laser Tracker")
    cv2.setMouseCallback("Laser Tracker", tracker.handle_click)

    prev_time = time.time()
    print("🚀 Tracker Running. Click to lock a point. 'c' to clear, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        vis_frame = tracker.update(frame)

        # FPS Stats
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(vis_frame, f"FPS: {int(fps)}", (20, 40), 1, 2, (0, 255, 0), 2)
        
        cv2.imshow("Laser Tracker", vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            tracker.tracking = False
            tracker.template = None
            print("🧹 Tracker Cleared")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()