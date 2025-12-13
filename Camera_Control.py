#!/usr/bin/env python3
# Clean, modular architecture for:
# Dual cameras + best-fit lines + DOT DETECTION + CLICK SELECTION

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
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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
    if 0 <= y0 < h:
        pts.append((0, int(y0)))

    y1 = m*(w-1) + b
    if 0 <= y1 < h:
        pts.append((w-1, int(y1)))

    if abs(m) > 1e-9:
        x_top = (0 - b)/m
        if 0 <= x_top < w:
            pts.append((int(x_top), 0))

        x_bot = ((h-1) - b)/m
        if 0 <= x_bot < w:
            pts.append((int(x_bot), h-1))

    if len(pts) < 2:
        return (0, int(b)), (w-1, int(b))
    return pts[0], pts[1]


def draw_detection_display(frameL, frameR, dotsL, dotsR, selL, selR):
    """Return a display image with dots + selected points + fit lines."""
    display = np.hstack((frameL.copy(), frameR.copy()))

    for (cx, cy) in dotsL:
        cv2.circle(display, (cx, cy), 5, (0,255,0), -1)
    for (cx, cy) in dotsR:
        cv2.circle(display, (FRAME_W + cx, cy), 5, (0,255,0), -1)

    if selL is not None:
        cv2.circle(display, selL, 10, (0,0,255), 2)
    if selR is not None:
        cv2.circle(display, (FRAME_W + selR[0], selR[1]), 10, (0,0,255), 2)

    p0, p1 = line_endpoints(mL, bL, FRAME_W, FRAME_H)
    cv2.line(display, p0, p1, (0,0,255), 2)

    q0, q1 = line_endpoints(mR, bR, FRAME_W, FRAME_H)
    cv2.line(
        display,
        (FRAME_W + q0[0], q0[1]),
        (FRAME_W + q1[0], q1[1]),
        (0,0,255), 2
    )

    return display


# ================================================================
# ==================== POINT SELECTION LOGIC =====================
# ================================================================
def select_points_interactively(capL, capR):
    selected_left = None
    selected_right = None

    win = "Select Dots (click to choose)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, userdata):
        nonlocal selected_left, selected_right
        frameL, frameR, dotsL, dotsR = (
            userdata["frameL"],
            userdata["frameR"],
            userdata["dotsL"],
            userdata["dotsR"],
        )

        if event == cv2.EVENT_LBUTTONDOWN:
            if x < FRAME_W:
                click = (x, y)
                if dotsL:
                    selected_left = min(
                        dotsL, key=lambda p: math.dist(p, click)
                    )
            else:
                click = (x - FRAME_W, y)
                if dotsR:
                    selected_right = min(
                        dotsR, key=lambda p: math.dist(p, click)
                    )

    cv2.setMouseCallback(win, on_mouse)

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()

        if not okL:
            frameL = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        if not okR:
            frameR = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)

        frameL = cv2.resize(frameL, (FRAME_W, FRAME_H))
        frameR = cv2.resize(frameR, (FRAME_W, FRAME_H))

        dotsL = detect_dots(frameL)
        dotsR = detect_dots(frameR)

        display = draw_detection_display(
            frameL, frameR, dotsL, dotsR, selected_left, selected_right
        )
        cv2.imshow(win, display)

        cv2.setMouseCallback(
            win, on_mouse,
            {"frameL": frameL, "frameR": frameR,
             "dotsL": dotsL, "dotsR": dotsR}
        )

        if selected_left is not None and selected_right is not None:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(win)
    return selected_left, selected_right


def nearest_neighbor(dots, last_pt):
    if not dots:
        return last_pt
    return min(dots, key=lambda p: math.dist(p, last_pt))


def vertical_error(x, y, m, b):
    return (m * x + b) - y


def horizontal_error(xL, xR, frame_width):
    dL = frame_width - xL
    dR = xR
    return dR - dL


def send_relative_move(ser, dx_mm=0.0, dy_mm=0.0, feedrate=3000):
    if ser is None:
        print("SERIAL NOT CONNECTED — skipping move")
        return
    cmd = f"G91\nG0 X{dx_mm:.3f} Y{dy_mm:.3f} F{feedrate}\nG90\n"
    ser.write(cmd.encode())
    ser.flush()


def run_control_loop(capL, capR, ser, left_pt, right_pt):
    print("\n=== CONTROL LOOP ===")

    tol_y = 3
    tol_x = 3
    step_mm = 0.001

    K_Px = 5
    K_Dx = 5
    K_Py = 11
    K_Dy = 3

    lastL = left_pt
    lastR = right_pt
    prev_eX = 0.0
    prev_eY = 0.0

    win = "Control Loop View"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    ShouldBurn = False

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            break

        frameL = cv2.resize(frameL, (FRAME_W, FRAME_H))
        frameR = cv2.resize(frameR, (FRAME_W, FRAME_H))

        dotsL = detect_dots(frameL)
        dotsR = detect_dots(frameR)

        xL, yL = nearest_neighbor(dotsL, lastL)
        xR, yR = nearest_neighbor(dotsR, lastR)
        lastL = (xL, yL)
        lastR = (xR, yR)

        eX = horizontal_error(xL, xR, FRAME_W)
        deX = eX - prev_eX
        prev_eX = eX

        if abs(eX) > tol_x and abs(deX*K_Dx) < abs(eX*K_Px):
            dx = round((eX*K_Px + deX*K_Dx)*step_mm, 2)
        else:
            dx = 0.0

        eYL = vertical_error(xL, yL, mL, bL)
        eYR = vertical_error(xR, yR, mR, bR)
        eY = 0.5*(eYL + eYR)

        deY = eY - prev_eY
        prev_eY = eY

        if abs(eY) > tol_y and abs(deY*K_Dy) < abs(eY*K_Py):
            dy = round((eY*K_Py + deY*K_Dy)*step_mm, 2)
        else:
            dy = 0.0

        if abs(dx) > 0.0 or abs(dy) > 0.0:
            send_relative_move(ser, dx_mm=dx, dy_mm=dy)

        if abs(eX) <= tol_x and abs(eY) <= tol_y and ShouldBurn:
            burn(ser, power=10, duration=2)
            ShouldBurn = False

        if cv2.waitKey(1) & 0xFF == ord(' '):
            ShouldBurn = True

        display = draw_detection_display(
            frameL, frameR, dotsL, dotsR, (xL,yL), (xR,yR)
        )
        cv2.imshow(win, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(win)
    print("=== CONTROL COMPLETE ===")


def main():
    capL = open_cam(LEFT_CAM_ID)
    capR = open_cam(RIGHT_CAM_ID)
    ser  = open_serial()
    send(ser,"$H")
    move_to(ser, 200, 200)

    left_pt, right_pt = select_points_interactively(capL, capR)
    run_control_loop(capL, capR, ser, left_pt, right_pt)

    capL.release()
    capR.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
