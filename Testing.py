#!/usr/bin/env python3
# Clean, modular architecture for:
# Dual cameras + best-fit lines + DOT DETECTION + CLICK SELECTION
from PixelController import PixelPDController
import cv2
import serial
import time
import numpy as np
import math
from Laser_Helpers import send, wait_for_idle, connect, move_to, burn, close
# ---------------- Config ----------------
RIGHT_CAM_ID = 1
LEFT_CAM_ID  = 2
FRAME_W = 1280
FRAME_H = 720

# Partner's best-fit alignment lines
mL = -0.0061461658179720905
bL = 368.03593073199704
mR = -0.008112495386117758
bR = 359.70747786328616

SERIAL_PORT = "COM3"
BAUD = 115200
# ----------------------------------------


# ================================================================
# =============== CAMERA / SERIAL OPEN FUNCTIONS =================
# ================================================================
def open_cam(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    time.sleep(0.02)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(idx)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap

def open_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
        print(f"[serial] Connected to {SERIAL_PORT} at {BAUD}")
        return ser
    except:
        print("[serial] FAILED to open serial port.")
        return None


# ================================================================
# ======================== VISION HELPERS ========================
# ================================================================
def detect_dots(frame):
    """Return list of (x, y) centroids for circular-ish objects."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 10 < area < 2000:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                dots.append((cx, cy))

    return dots


def line_endpoints(m, b, w, h):
    """Compute clipped endpoints for y = m*x + b."""
    pts = []

    y0 = b
    if 0 <= y0 < h: pts.append((0, int(y0)))

    y1 = m*(w-1) + b
    if 0 <= y1 < h: pts.append((w-1, int(y1)))

    if abs(m) > 1e-9:
        x_top = (0 - b)/m
        if 0 <= x_top < w: pts.append((int(x_top), 0))

        x_bot = ((h-1) - b)/m
        if 0 <= x_bot < w: pts.append((int(x_bot), h-1))

    if len(pts) < 2:
        return (0, int(b)), (w-1, int(b))
    return pts[0], pts[1]


def draw_detection_display(frameL, frameR, dotsL, dotsR, selL, selR):
    """Return a display image with dots + selected points + fit lines."""
    display = np.hstack((frameL.copy(), frameR.copy()))

    # draw dots
    for (cx, cy) in dotsL:
        cv2.circle(display, (cx, cy), 5, (0,255,0), -1)
    for (cx, cy) in dotsR:
        cv2.circle(display, (FRAME_W + cx, cy), 5, (0,255,0), -1)

    # draw selected
    if selL is not None:
        cv2.circle(display, selL, 10, (0,0,255), 2)
    if selR is not None:
        cv2.circle(display, (FRAME_W + selR[0], selR[1]), 10, (0,0,255), 2)

    # draw left line
    p0, p1 = line_endpoints(mL, bL, FRAME_W, FRAME_H)
    cv2.line(display, p0, p1, (0,0,255), 2)

    # draw right line
    q0, q1 = line_endpoints(mR, bR, FRAME_W, FRAME_H)
    cv2.line(display, (FRAME_W + q0[0], q0[1]),
             (FRAME_W + q1[0], q1[1]), (0,0,255), 2)

    return display


# ================================================================
# ==================== POINT SELECTION LOGIC =====================
# ================================================================
def select_points_interactively(capL, capR):
    """
    Show live feed, detect dots, allow clicking to choose nearest dot
    Return (left_point, right_point)
    """

    selected_left = None
    selected_right = None

    win = "Select Dots (click to choose)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # mouse callback inner function that closes over variables
    def on_mouse(event, x, y, flags, userdata):
        nonlocal selected_left, selected_right
        frameL, frameR, dotsL, dotsR = userdata["frameL"], userdata["frameR"], userdata["dotsL"], userdata["dotsR"]

        if event == cv2.EVENT_LBUTTONDOWN:
            if x < FRAME_W:     # left image
                click = (x, y)
                if dotsL:
                    selected_left = min(dotsL,
                                        key=lambda p: math.dist(p, click))
                    print("Selected LEFT:", selected_left)
            else:               # right image
                click = (x - FRAME_W, y)
                if dotsR:
                    selected_right = min(dotsR,
                                         key=lambda p: math.dist(p, click))
                    print("Selected RIGHT:", selected_right)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()

        if not okL: frameL = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        if not okR: frameR = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)

        frameL = cv2.resize(frameL, (FRAME_W, FRAME_H))
        frameR = cv2.resize(frameR, (FRAME_W, FRAME_H))

        dotsL = detect_dots(frameL)
        dotsR = detect_dots(frameR)

        display = draw_detection_display(frameL, frameR, dotsL, dotsR, selected_left, selected_right)

        cv2.imshow(win, display)

        # update callback data for this frame
        cv2.setMouseCallback(win, on_mouse,
                             {"frameL": frameL, "frameR": frameR,
                              "dotsL": dotsL, "dotsR": dotsR})

        # exit logic
        if selected_left is not None and selected_right is not None:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(win)
    return selected_left, selected_right

# Simple tracking function

def nearest_neighbor(dots, last_pt):
    """Return the dot closest to last_pt. If no dots detected, return last_pt."""
    if not dots:
        return last_pt
    return min(dots, key=lambda p: math.dist(p, last_pt))

# Vertical error function
def vertical_error(x, y, m, b):
    return  (m * x + b) - y

# Horizontal symmetry error function
def horizontal_error(xL, xR, frame_width):
    dL = frame_width - xL   # dist from right edge (left cam)
    dR = xR                 # dist from left edge  (right cam)
    return dR - dL



# Relative movements
def send_relative_move(ser, dx_mm=0.0, dy_mm=0.0, feedrate=3000):
    if ser is None:
        print("SERIAL NOT CONNECTED — skipping move")
        return
    cmd = f"G91\nG0 X{dx_mm:.3f} Y{dy_mm:.3f} F{feedrate}\nG90\n"
    ser.write(cmd.encode())
    ser.flush()



def run_control_loop(capL, capR, ser, left_pt, right_pt):
    print("\n=== CONTROL LOOP ===")

    # ---- instantiate controller (matches old gains exactly) ----
    controller = PixelPDController(
        frame_width=FRAME_W,
        mL=mL, bL=bL,
        mR=mR, bR=bR
    )

    lastL = left_pt
    lastR = right_pt

    win = "Control Loop View"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    ShouldBurn = True

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            print("Camera read failed.")
            break

        frameL = cv2.resize(frameL, (FRAME_W, FRAME_H))
        frameR = cv2.resize(frameR, (FRAME_W, FRAME_H))

        # ---------- ORIGINAL CV (UNCHANGED) ----------
        dotsL = detect_dots(frameL)
        dotsR = detect_dots(frameR)

        xL, yL = nearest_neighbor(dotsL, lastL)
        xR, yR = nearest_neighbor(dotsR, lastR)

        lastL = (xL, yL)
        lastR = (xR, yR)

        # ---------- NEW: CONTROLLER REPLACES PD MATH ----------
        dx, dy, aligned = controller.step(xL, yL, xR, yR)

        # ---------- ORIGINAL ACTUATION ----------
        if abs(dx) > 0.0 or abs(dy) > 0.0:
            print(f"[CTRL] dx={dx}, dy={dy}")
            send_relative_move(ser, dx_mm=dx, dy_mm=dy)

        # ---------- ORIGINAL BURN LOGIC ----------
        if aligned and ShouldBurn:
            print("\n=== ALIGNED ===\n")
            if cv2.waitKey(1) & 0xFF == ord(' ') or ShouldBurn:
                burn(ser, power=10, duration=2)
                ShouldBurn = False

        if cv2.waitKey(1) & 0xFF == ord(' '):
            ShouldBurn = True

        # ---------- ORIGINAL DISPLAY ----------
        display = draw_detection_display(
            frameL, frameR,
            dotsL, dotsR,
            (xL, yL), (xR, yR)
        )
        cv2.imshow(win, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped manually.")
            break

    cv2.destroyWindow(win)
    print("=== CONTROL COMPLETE ===")

# ========================== MAIN LOOP ===========================
def main():
    capL = open_cam(LEFT_CAM_ID)
    capR = open_cam(RIGHT_CAM_ID)
    ser  = open_serial()
    # send(ser, "$H")  # home if needed
    move_to(ser, 200, 200)

    print("\n=== SELECTING POINTS ===")
    left_pt, right_pt = select_points_interactively(capL, capR)

    print("\nSelected:")
    print(" Left :", left_pt)
    print(" Right:", right_pt)

    print("\n=== STARTING CONTROL LOOP ===")
    run_control_loop(capL, capR, ser, left_pt, right_pt)

    capL.release()
    capR.release()
    if ser: ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
