import cv2
import numpy as np
import json
import os
import time

# === CONFIG ===
CAM_ID_RIGHT = 0
CAM_ID_LEFT = 2
SETTINGS_FILE = "camera_settings.json"

class StereoTracker:
    def __init__(self):
        self.settings = self.load_settings()
        self.zapped_memory = [] # Global MPos coordinates of zapped weeds
        
    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        return None

    def apply_hardware_settings(self, caps):
        if not self.settings: return
        s = self.settings
        for cap in caps:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, s['Exposure'])
            cap.set(cv2.CAP_PROP_GAIN, s['Gain'])
            cap.set(cv2.CAP_PROP_BRIGHTNESS, s['Brightness'])
            cap.set(cv2.CAP_PROP_CONTRAST, s['Contrast'])
            cap.set(cv2.CAP_PROP_SATURATION, s['Saturation'])
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, s['WB_Temp'])

    def get_dots(self, frame):
        s = self.settings
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Cleanup
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if s['Min_Area'] < area < s['Max_Area']:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append({'center': (cx, cy), 'area': area})
        return dots

    def match_stereo_pairs(self, dots_l, dots_r):
        """
        Matches dots between eyes. 
        Rule 1: They must be on roughly the same Y-axis (Epipolar constraint).
        Rule 2: The Left dot must be to the RIGHT of the Right dot (Disparity).
        """
        matches = []
        used_r = set()
        
        for i, dot_l in enumerate(dots_l):
            lx, ly = dot_l['center']
            best_match = None
            min_y_diff = 15 # Pixels of vertical tolerance
            
            for j, dot_r in enumerate(dots_r):
                if j in used_r: continue
                rx, ry = dot_r['center']
                
                y_diff = abs(ly - ry)
                # In stereo, dots move horizontally. Large Y diff means different objects.
                if y_diff < min_y_diff:
                    # Calculate disparity (shift)
                    # For standard parallel cams, rx is usually < lx
                    disparity = lx - rx
                    if 0 < disparity < 200: # Typical range for 60mm height
                        matches.append({
                            'left': dot_l['center'],
                            'right': dot_r['center'],
                            'disparity': disparity,
                            'id': i
                        })
                        used_r.add(j)
                        break
        return matches

def main():
    tracker = StereoTracker()
    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)
    
    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker.apply_hardware_settings([cap_l, cap_r])

    while True:
        ret_l, fl = cap_l.read()
        ret_r, fr = cap_r.read()
        if not ret_l or not ret_r: break

        # 1. Perception
        dots_l = tracker.get_dots(fl)
        dots_r = tracker.get_dots(fr)

        # 2. Matching (The Brain)
        matches = tracker.match_stereo_pairs(dots_l, dots_r)

        # 3. Labeling
        for m in matches:
            lx, ly = m['left']
            rx, ry = m['right']
            # Draw green circle on left, yellow on right
            cv2.circle(fl, (lx, ly), 15, (0, 255, 0), 2)
            cv2.circle(fr, (rx, ry), 15, (0, 255, 255), 2)
            # Label with ID
            cv2.putText(fl, f"ID:{m['id']}", (lx+15, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(fr, f"ID:{m['id']}", (rx+15, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Side-by-Side View
        combined = np.hstack((fl, fr))
        cv2.imshow("Stereo Constellation Matching", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_l.release(); cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()