import cv2
import numpy as np
import glob
import json
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
PATTERN_SIZE = (6, 7)      # inner corners
SQUARE_SIZE_MM = 25.0
MAX_VIEWS = 30             # keep dataset stable
MIN_VIEWS = 8
SAVE_DEBUG = False

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "calib_pairs"
DEBUG_DIR = BASE_DIR / "checkerboard_debug"

CALIB_FILE = BASE_DIR / "stereo_checkerboard_fisheye_calib.npz"
RECTIFY_FILE = BASE_DIR / "stereo_checkerboard_fisheye_rectify_maps.npz"
JSON_FILE = BASE_DIR / "stereo_checkerboard_fisheye_calib.json"

SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-6,
)

CALIB_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    200,
    1e-7,
)


# ============================================================
# IO
# ============================================================
def load_image_pairs():
    left_images = sorted(glob.glob(str(IMAGE_DIR / "left_*.png")))
    right_images = sorted(glob.glob(str(IMAGE_DIR / "right_*.png")))

    print(f"Looking for images in: {IMAGE_DIR}")
    print(f"Found left images: {len(left_images)}")
    print(f"Found right images: {len(right_images)}")

    if not left_images or not right_images:
        raise FileNotFoundError(f"No stereo images found in {IMAGE_DIR}")

    n = min(len(left_images), len(right_images))
    pairs = list(zip(left_images[:n], right_images[:n]))

    if not pairs:
        raise RuntimeError("No stereo pairs available.")

    return pairs


def get_image_size(pairs):
    for l_path, r_path in pairs:
        imgL = cv2.imread(l_path)
        imgR = cv2.imread(r_path)
        if imgL is not None and imgR is not None:
            h, w = imgL.shape[:2]
            return (w, h)

    raise RuntimeError("Could not read any valid image pair.")


# ============================================================
# CHECKERBOARD
# ============================================================
def make_object_points(pattern_size, square_size_mm):
    cols, rows = pattern_size
    n = cols * rows

    objp = np.zeros((n, 1, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, 0, :2] = grid * float(square_size_mm)
    return objp


def detect_checkerboard(gray, pattern_size):
    corners = None
    found = False

    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            flags_sb = (
                cv2.CALIB_CB_EXHAUSTIVE
                + cv2.CALIB_CB_ACCURACY
                + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags_sb)
        except Exception:
            found, corners = False, None

    if not found or corners is None:
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if found and corners is not None:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (5, 5),
                (-1, -1),
                SUBPIX_CRITERIA,
            )

    if not found or corners is None:
        return False, None

    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    expected = pattern_size[0] * pattern_size[1]

    if corners.shape != (expected, 1, 2):
        return False, None

    return True, corners


def draw_debug(img, pattern_size, corners, found, label=""):
    vis = img.copy()

    if found and corners is not None:
        try:
            corners_draw = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
            expected = pattern_size[0] * pattern_size[1]
            if corners_draw.shape[0] == expected:
                cv2.drawChessboardCorners(vis, pattern_size, corners_draw, True)
        except Exception:
            pass

    if label:
        cv2.putText(
            vis,
            label,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return vis


# ============================================================
# DATASET BUILD
# ============================================================
def gather_points(pairs, pattern_size, save_debug=False):
    obj_template = make_object_points(pattern_size, SQUARE_SIZE_MM)

    objpoints = []
    imgpointsL = []
    imgpointsR = []
    per_pair = []

    if save_debug:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    for i, (l_path, r_path) in enumerate(pairs):
        imgL = cv2.imread(l_path)
        imgR = cv2.imread(r_path)

        if imgL is None or imgR is None:
            per_pair.append({
                "index": i,
                "left_path": Path(l_path).name,
                "right_path": Path(r_path).name,
                "found_left": False,
                "found_right": False,
                "used": False,
            })
            continue

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = detect_checkerboard(grayL, pattern_size)
        foundR, cornersR = detect_checkerboard(grayR, pattern_size)

        used = False

        if foundL and foundR:
            expected = pattern_size[0] * pattern_size[1]

            cornersL = np.asarray(cornersL, dtype=np.float64).reshape(-1, 1, 2)
            cornersR = np.asarray(cornersR, dtype=np.float64).reshape(-1, 1, 2)

            if cornersL.shape == (expected, 1, 2) and cornersR.shape == (expected, 1, 2):
                objpoints.append(obj_template.copy())
                imgpointsL.append(cornersL.copy())
                imgpointsR.append(cornersR.copy())
                used = True

        per_pair.append({
            "index": i,
            "left_path": Path(l_path).name,
            "right_path": Path(r_path).name,
            "found_left": bool(foundL),
            "found_right": bool(foundR),
            "used": bool(used),
        })

        if save_debug:
            cv2.imwrite(
                str(DEBUG_DIR / f"left_{i:03d}.png"),
                draw_debug(imgL, pattern_size, cornersL if foundL else None, foundL, f"L {'OK' if foundL else 'MISS'}")
            )
            cv2.imwrite(
                str(DEBUG_DIR / f"right_{i:03d}.png"),
                draw_debug(imgR, pattern_size, cornersR if foundR else None, foundR, f"R {'OK' if foundR else 'MISS'}")
            )

    return objpoints, imgpointsL, imgpointsR, per_pair


def thin_dataset(objpoints, imgpointsL, imgpointsR, per_pair, max_views):
    if len(objpoints) <= max_views:
        return objpoints, imgpointsL, imgpointsR, per_pair

    idx = np.linspace(0, len(objpoints) - 1, max_views).astype(int)
    idx = sorted(set(idx.tolist()))

    objpoints = [objpoints[i] for i in idx]
    imgpointsL = [imgpointsL[i] for i in idx]
    imgpointsR = [imgpointsR[i] for i in idx]
    per_pair_used = [p for p in per_pair if p["used"]]
    per_pair_used = [per_pair_used[i] for i in idx]

    print(f"Reduced to {len(objpoints)} views for stability")
    return objpoints, imgpointsL, imgpointsR, per_pair_used


# ============================================================
# CALIBRATION
# ============================================================
def calibrate_single_fisheye(objpoints, imgpoints, img_size):
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        image_size=img_size,
        K=K,
        D=D,
        rvecs=None,
        tvecs=None,
        flags=flags,
        criteria=CALIB_CRITERIA,
    )
    return rms, K, D, rvecs, tvecs


def sanitize_stereo_inputs(objpoints, imgpointsL, imgpointsR, pattern_size):
    expected = pattern_size[0] * pattern_size[1]

    obj_clean = []
    left_clean = []
    right_clean = []

    for obj, lpt, rpt in zip(objpoints, imgpointsL, imgpointsR):
        obj = np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3)
        lpt = np.asarray(lpt, dtype=np.float64).reshape(-1, 1, 2)
        rpt = np.asarray(rpt, dtype=np.float64).reshape(-1, 1, 2)

        if obj.shape != (expected, 1, 3):
            continue
        if lpt.shape != (expected, 1, 2):
            continue
        if rpt.shape != (expected, 1, 2):
            continue

        obj_clean.append(obj.copy())
        left_clean.append(lpt.copy())
        right_clean.append(rpt.copy())

    return obj_clean, left_clean, right_clean


def calibrate_stereo_fisheye(objpoints, imgpointsL, imgpointsR, K1, D1, K2, D2, img_size, pattern_size):
    objpoints, imgpointsL, imgpointsR = sanitize_stereo_inputs(
        objpoints, imgpointsL, imgpointsR, pattern_size
    )

    print("\n=== STEREO ARRAY CHECK ===")
    for i, (obj, lpt, rpt) in enumerate(zip(objpoints, imgpointsL, imgpointsR)):
        print(
            f"{i:02d}: obj {obj.shape} {obj.dtype}, "
            f"L {lpt.shape} {lpt.dtype}, "
            f"R {rpt.shape} {rpt.dtype}"
        )

    if len(objpoints) < MIN_VIEWS:
        raise RuntimeError(f"Only {len(objpoints)} clean stereo pairs remain.")

    flags = cv2.fisheye.CALIB_FIX_INTRINSIC
    R_init = np.eye(3, dtype=np.float64)
    T_init = np.zeros((3, 1), dtype=np.float64)

    out = cv2.fisheye.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpointsL,
        imagePoints2=imgpointsR,
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        imageSize=img_size,
        R=R_init,
        T=T_init,
        flags=flags,
        criteria=CALIB_CRITERIA,
    )

    rms, K1o, D1o, K2o, D2o, R, T = out[:7]
    return rms, K1o, D1o, K2o, D2o, R, T


def np_to_list(x):
    return np.asarray(x).tolist()


# ============================================================
# MAIN
# ============================================================
def main():
    pairs = load_image_pairs()
    img_size = get_image_size(pairs)

    print(f"Total pairs found: {len(pairs)}")

    objpoints, imgpointsL, imgpointsR, per_pair = gather_points(
        pairs,
        PATTERN_SIZE,
        save_debug=SAVE_DEBUG,
    )

    print(f"Usable stereo pairs: {len(objpoints)}")

    if len(objpoints) < MIN_VIEWS:
        raise RuntimeError(
            f"Not enough usable stereo pairs. Need at least {MIN_VIEWS}, got {len(objpoints)}."
        )

    objpoints, imgpointsL, imgpointsR, per_pair_used = thin_dataset(
        objpoints, imgpointsL, imgpointsR, per_pair, MAX_VIEWS
    )

    rmsL, K1, D1, _, _ = calibrate_single_fisheye(objpoints, imgpointsL, img_size)
    rmsR, K2, D2, _, _ = calibrate_single_fisheye(objpoints, imgpointsR, img_size)

    rmsStereo, K1o, D1o, K2o, D2o, R, T = calibrate_stereo_fisheye(
        objpoints,
        imgpointsL,
        imgpointsR,
        K1,
        D1,
        K2,
        D2,
        img_size,
        PATTERN_SIZE,
    )

    baseline_mm = float(np.linalg.norm(T))

    print("\n=== CALIBRATION RESULTS ===")
    print(f"Image size: {img_size[0]} x {img_size[1]}")
    print(f"Pattern used (inner corners): {PATTERN_SIZE[0]} x {PATTERN_SIZE[1]}")
    print(f"Square size: {SQUARE_SIZE_MM:.3f} mm")
    print(f"Usable stereo pairs: {len(objpoints)}")
    print(f"Left fisheye RMS:   {rmsL:.6f}")
    print(f"Right fisheye RMS:  {rmsR:.6f}")
    print(f"Stereo fisheye RMS: {rmsStereo:.6f}")
    print(f"Estimated baseline: {baseline_mm:.3f} mm")
    print("K1 =\n", K1o)
    print("D1 =\n", D1o.ravel())
    print("K2 =\n", K2o)
    print("D2 =\n", D2o.ravel())
    print("R =\n", R)
    print("T =\n", T.ravel())

    np.savez(
        CALIB_FILE,
        model="fisheye_checkerboard",
        imageSize=np.array(img_size, dtype=np.int32),
        patternSize=np.array(PATTERN_SIZE, dtype=np.int32),
        squareSizeMM=float(SQUARE_SIZE_MM),
        K1=K1o,
        D1=D1o,
        K2=K2o,
        D2=D2o,
        R=R,
        T=T,
        rmsLeft=float(rmsL),
        rmsRight=float(rmsR),
        rmsStereo=float(rmsStereo),
    )

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1o,
        D1o,
        K2o,
        D2o,
        img_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=img_size,
        balance=0.0,
        fov_scale=1.0,
    )

    map1L, map2L = cv2.fisheye.initUndistortRectifyMap(
        K1o, D1o, R1, P1, img_size, cv2.CV_16SC2
    )
    map1R, map2R = cv2.fisheye.initUndistortRectifyMap(
        K2o, D2o, R2, P2, img_size, cv2.CV_16SC2
    )

    np.savez(
        RECTIFY_FILE,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        map1L=map1L,
        map2L=map2L,
        map1R=map1R,
        map2R=map2R,
    )

    report = {
        "model": "fisheye_checkerboard",
        "image_size": list(img_size),
        "pattern_size_inner_corners": list(PATTERN_SIZE),
        "square_size_mm": SQUARE_SIZE_MM,
        "usable_stereo_pairs": len(objpoints),
        "rms_left": float(rmsL),
        "rms_right": float(rmsR),
        "rms_stereo": float(rmsStereo),
        "baseline_mm": baseline_mm,
        "K1": np_to_list(K1o),
        "D1": np_to_list(D1o),
        "K2": np_to_list(K2o),
        "D2": np_to_list(D2o),
        "R": np_to_list(R),
        "T": np_to_list(T),
        "used_pairs": per_pair_used,
    }

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved calibration: {CALIB_FILE}")
    print(f"Saved rectification maps: {RECTIFY_FILE}")
    print(f"Saved JSON summary: {JSON_FILE}")
    if SAVE_DEBUG:
        print(f"Saved debug detections in: {DEBUG_DIR}")


if __name__ == "__main__":
    main()