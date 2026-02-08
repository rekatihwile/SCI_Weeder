import cv2
import numpy as np
import json
import os
import time
import math
import Laser_Helpers as lh 

# === CONFIGURATION ===
WS_W, WS_H = 450, 440 
CAM_ID_L, CAM_ID_R = 1, 4 
W, H = 640, 480 
TOL_X, TOL_Y = 10, 10 
WIN_NAME = "Stereo Autonomous Weeder"

# Alignment lines
mL, bL = -0.0061461658179720905, 368.03593073199704
mR, bR = -0.008112495386117758, 359.70747786328616

# Gains and Step Scale
K_Px, K_Dx = 5, 1
K_Py, K_Dy = 10, 1
STEP_MM = 0.001

LENS_DEAD_ZONE_L = (0, 440, 40, 480) 
LENS_DEAD_ZONE_R = (0, 440, 40, 480)

def load_cv_settings():
    if os.path.exists("cv_settings.json"):
        with open("cv_settings.json", "r") as f:
            return json.load(f)
    return {"Filter_d": 9, "Filter_sigma": 75, "BlockSize": 21, "C_Value": 4, "MinArea": 50, "MinCirc": 65}

def line_endpoints(m, b, w, h):
    y0, y1 = int(b), int(m * (w - 1) + b)
    return (0, y0), (w - 1, y1)

def process_frame(frame, params, dead_zone=None):
    if frame is None: return [], None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if dead_zone:
        x1, y1, x2, y2 = dead_zone
        cv2.rectangle(gray, (x1, y1), (x2, y2), (255), -1)

    filtered = cv2.bilateralFilter(gray, params['Filter_d'], params['Filter_sigma'], params['Filter_sigma'])
    bs = params['BlockSize'] if params['BlockSize'] % 2 != 0 else params['BlockSize'] + 1
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, bs, params['C_Value'])
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots, vis = [], frame.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > params['MinArea']:
            peri = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            if circ > (params['MinCirc'] / 100.0):
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    dots.append((cx, cy))
                    cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 2)
    return dots, vis

class StereoPDWeeder:
    def __init__(self):
        self.ser = lh.connect()
        print("🏠 Homing Gantry...")
        lh.send(self.ser, "$H")
        lh.wait_for_idle(self.ser)
        
        self.cap_l = cv2.VideoCapture(CAM_ID_L)
        self.cap_r = cv2.VideoCapture(CAM_ID_R)
        
        for c in [self.cap_l, self.cap_r]:
            c.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        
        self.params = load_cv_settings()
        self.prev_eX, self.prev_eY = 0.0, 0.0
        # MEMORY: Store absolute machine coordinates of zapped dots
        self.zapped_memory = []
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

    def get_combined_display(self, text, color=(255, 255, 255), targeting_dots=None):
        retL, fl = self.cap_l.read()
        retR, fr = self.cap_r.read()
        if not retL or not retR: return None

        dotsL, visL = process_frame(fl, self.params, LENS_DEAD_ZONE_L)
        dotsR, visR = process_frame(fr, self.params, LENS_DEAD_ZONE_R)

        p0, p1 = line_endpoints(mL, bL, W, H)
        q0, q1 = line_endpoints(mR, bR, W, H)
        cv2.line(visL, p0, p1, (0, 0, 255), 2)
        cv2.line(visR, q0, q1, (0, 0, 255), 2)

        if targeting_dots:
            cv2.circle(visL, targeting_dots[0], 12, (0, 0, 255), 2)
            cv2.circle(visR, targeting_dots[1], 12, (0, 0, 255), 2)

        combined = np.hstack((visL, visR))
        cv2.putText(combined, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return combined, dotsL, dotsR

    def get_machine_pos(self):
        """Polls GRBL for current real-time machine coordinates."""
        self.ser.write(b"?\n")
        resp = self.ser.readline().decode(errors="ignore")
        if "MPos:" in resp:
            try:
                pos_str = resp.split("MPos:")[1].split("|")[0]
                return [float(x) for x in pos_str.split(",")]
            except: return [0,0,0]
        return [0,0,0]

    def pd_target_stereo(self, lastL, lastR):
        print("🎯 Fine-Tuning...")
        stable_count = 0
        self.prev_eX, self.prev_eY = 0.0, 0.0

        while True:
            display_data = self.get_combined_display("TARGETING...", (0, 255, 255), (lastL, lastR))
            if display_data is None: break
            combined, dotsL, dotsR = display_data
            
            if not dotsL or not dotsR: return False 
            
            xL, yL = min(dotsL, key=lambda p: math.dist(p, lastL))
            xR, yR = min(dotsR, key=lambda p: math.dist(p, lastR))
            lastL, lastR = (xL, yL), (xR, yR)

            eX = xR - (W - xL) 
            deX = eX - self.prev_eX
            dx = round((eX * K_Px + deX * K_Dx) * STEP_MM, 3)

            eYL = (mL * xL + bL) - yL
            eYR = (mR * xR + bR) - yR
            eY = 0.5 * (eYL + eYR)
            deY = eY - self.prev_eY
            dy = round((eY * K_Py + deY * K_Dy) * STEP_MM, 3)

            if abs(dx) > 0.0 or abs(dy) > 0.0:
                lh.send(self.ser, f"$J=G91 X{dx:.3f} Y{dy:.3f} F3000")
                time.sleep(0.005)

            if abs(eX) <= TOL_X and abs(eY) <= TOL_Y:
                stable_count += 1
                if stable_count > 15:
                    print("✅ TARGETED")
                    # ABORT JOG: Clears the buffer immediately to prevent the 10s freeze
                    self.ser.write(b"\x85") 
                    time.sleep(0.1)
                    
                    # Store current position in memory
                    mpos = self.get_machine_pos()
                    self.zapped_memory.append((mpos[0], mpos[1]))
                    
                    while True:
                        disp = self.get_combined_display("TARGETED | SPACE to Return", (0, 255, 0), (lastL, lastR))
                        if disp: cv2.imshow(WIN_NAME, disp[0])
                        if cv2.waitKey(1) & 0xFF == ord(' '): return True
            else:
                stable_count = 0
            
            cv2.imshow(WIN_NAME, combined)
            if cv2.waitKey(1) & 0xFF == ord('q'): return False
            self.prev_eX, self.prev_eY = eX, eY

    def run(self):
        try:
            while True:
                disp = self.get_combined_display("VERIFICATION | SPACE TO BEGIN")
                if disp: cv2.imshow(WIN_NAME, disp[0])
                if cv2.waitKey(1) & 0xFF == ord(' '): break

            viewpoints = [[60, WS_H-50], [160, WS_H-50], [260, WS_H-50], [360, WS_H-50]]

            for pt in viewpoints:
                lh.move_to(self.ser, pt[0], pt[1])
                lh.wait_for_idle(self.ser)
                
                while True:
                    disp = self.get_combined_display(f"VIEWPOINT: {pt} | SPACE TO TARGET")
                    if disp: 
                        cv2.imshow(WIN_NAME, disp[0])
                        dotsL, dotsR = disp[1], disp[2]
                    
                    if cv2.waitKey(1) & 0xFF == ord(' '): break
                
                if dotsL and dotsR:
                    for dL in dotsL:
                        # 1. Coordinate Conversion (approximate) to check memory
                        # If the dot is within 5mm of a previous zap, skip it
                        approx_abs_x = pt[0] + (dL[0] - W/2) * 0.25 # Assume 0.25mm/px
                        approx_abs_y = pt[1] - (dL[1] - H/2) * 0.25
                        
                        if any(math.dist((approx_abs_x, approx_abs_y), z) < 5 for z in self.zapped_memory):
                            print("⏭️ Skipping already zapped dot.")
                            continue

                        dR = min(dotsR, key=lambda p: abs(p[1] - dL[1]))
                        if self.pd_target_stereo(dL, dR):
                            lh.move_to(self.ser, pt[0], pt[1])
                            lh.wait_for_idle(self.ser)

        finally:
            lh.send(self.ser, "G0 X0 Y0")
            lh.close(self.ser)
            self.cap_l.release(); self.cap_r.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    StereoPDWeeder().run()