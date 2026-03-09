import time
import cv2
import numpy as np

W = 640
H = 480
TARGET_Y_L = 240
TARGET_Y_R = 240

LK_PARAMS = dict(
    winSize=(31, 31),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

Kp_x = 20.0
Kd_x = 2.5
Kp_y = 20.0
Kd_y = 2.5

STEP_MM = 0.001
DEADZONE = 4
MAX_JOG = 10.0
FINE_FEED = 5000


def fine_align_target(gantry, cameras, detector, max_time=6.0, settle_frames=3, show_debug=True):
    left_pt, right_pt = detector.refine_live(cameras)

    if left_pt is None or right_pt is None:
        print("Fine align cancelled or no target found.")
        return False

    track_pt_L = np.array([[left_pt]], dtype=np.float32)
    track_pt_R = np.array([[right_pt]], dtype=np.float32)

    old_left, old_right = cameras.read_pair()
    old_gray_L = cv2.cvtColor(old_left, cv2.COLOR_BGR2GRAY)
    old_gray_R = cv2.cvtColor(old_right, cv2.COLOR_BGR2GRAY)

    prev_ex = 0.0
    prev_ey = 0.0
    inside_count = 0
    t0 = time.time()

    while time.time() - t0 < max_time:
        frameL, frameR = cameras.read_pair()
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        new_pt_L, stL, _ = cv2.calcOpticalFlowPyrLK(old_gray_L, grayL, track_pt_L, None, **LK_PARAMS)
        new_pt_R, stR, _ = cv2.calcOpticalFlowPyrLK(old_gray_R, grayR, track_pt_R, None, **LK_PARAMS)

        if stL is None or stR is None or stL[0][0] == 0 or stR[0][0] == 0:
            gantry.stop()
            print("Fine align lost LK tracking.")
            return False

        track_pt_L = new_pt_L
        track_pt_R = new_pt_R

        xl, yl = float(track_pt_L[0, 0, 0]), float(track_pt_L[0, 0, 1])
        xr, yr = float(track_pt_R[0, 0, 0]), float(track_pt_R[0, 0, 1])

        err_x = (xl - W + xr)
        err_y = -((yl - TARGET_Y_L) + (yr - TARGET_Y_R)) / 2.0

        dex = err_x - prev_ex
        dey = err_y - prev_ey
        prev_ex = err_x
        prev_ey = err_y

        dx = 0.0
        dy = 0.0

        if abs(err_x) > DEADZONE:
            dx = round((err_x * Kp_x + dex * Kd_x) * STEP_MM, 3)

        if abs(err_y) > DEADZONE:
            dy = round((err_y * Kp_y + dey * Kd_y) * STEP_MM, 3)

        dx = float(np.clip(dx, -MAX_JOG, MAX_JOG))
        dy = float(np.clip(dy, -MAX_JOG, MAX_JOG))

        if abs(err_x) <= DEADZONE and abs(err_y) <= DEADZONE:
            inside_count += 1
            gantry.stop()

            if inside_count >= settle_frames:
                print(f"Fine align locked: ex={err_x:.2f}px ey={err_y:.2f}px")
                return True
        else:
            inside_count = 0
            if dx != 0.0 or dy != 0.0:
                gantry.jog(dx, dy, FINE_FEED)

        if show_debug:
            dispL = frameL.copy()
            dispR = frameR.copy()

            cv2.circle(dispL, (int(xl), int(yl)), 5, (0, 0, 255), -1)
            cv2.circle(dispR, (int(xr), int(yr)), 5, (0, 0, 255), -1)

            cv2.line(dispL, (W // 2, 0), (W // 2, H), (255, 0, 0), 1)
            cv2.line(dispL, (0, TARGET_Y_L), (W, TARGET_Y_L), (255, 0, 0), 1)
            cv2.line(dispR, (W // 2, 0), (W // 2, H), (255, 0, 0), 1)
            cv2.line(dispR, (0, TARGET_Y_R), (W, TARGET_Y_R), (255, 0, 0), 1)

            status = f"FINE ALIGN | ex={err_x:.1f}px ey={err_y:.1f}px | dx={dx:.3f} dy={dy:.3f}"
            cv2.putText(dispL, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Fine Align - Left", dispL)
            cv2.imshow("Fine Align - Right", dispR)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("r"):
                gantry.stop()
                print("Fine align cancelled.")
                return False

        old_gray_L = grayL
        old_gray_R = grayR

    gantry.stop()
    print("Fine align timeout.")
    return False