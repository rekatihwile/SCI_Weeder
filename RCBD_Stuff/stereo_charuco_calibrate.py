import cv2
import numpy as np
from pathlib import Path
import glob

# =========================
# USER SETTINGS (EDIT THESE)
# =========================
PAIRS_DIR = Path(__file__).resolve().parent / "calib_pairs"

# Number of chessboard squares across and down on the printed ChArUco board
# If your board is 8 squares wide by 8 squares tall, keep these as 8, 8.
SQUARES_X = 8
SQUARES_Y = 8

# Physical dimensions of the printed board
SQUARE_LEN_MM = 30.0   # chess square edge length in mm
MARKER_LEN_MM = 14.0   # ArUco marker edge length in mm

# Dictionary used to generate/print the board
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_1000

# Minimum detected ChArUco corners required per image to keep that image
MIN_CHARUCO_CORNERS_PER_VIEW = 12

# Minimum number of common ChArUco corners between left/right image in a pair
MIN_COMMON_STEREO_CORNERS = 8

# Minimum number of usable stereo pairs
MIN_STEREO_PAIRS = 10

# Calibration output file
OUT_FILE = Path(__file__).resolve().parent / "stereo_charuco_fisheye_calib.npz"
RECTIFY_FILE = Path(__file__).resolve().parent / "stereo_fisheye_rectify_maps.npz"

# Debug visualization output (optional)
SAVE_DEBUG_DETECTIONS = False
DEBUG_DIR = Path(__file__).resolve().parent / "charuco_debug"


def load_pairs(pairs_dir: Path):
    lefts = sorted(glob.glob(str(pairs_dir / "left_*.png")))
    rights = sorted(glob.glob(str(pairs_dir / "right_*.png")))

    if len(lefts) == 0 or len(rights) == 0:
        raise FileNotFoundError(
            f"No image pairs found in {pairs_dir} "
            f"(expected left_*.png and right_*.png)"
        )

    if len(lefts) != len(rights):
        print(f"Warning: left count {len(lefts)} != right count {len(rights)}. Using min count.")

    n = min(len(lefts), len(rights))
    return lefts[:n], rights[:n]


def make_board(dictionary_id: int, square_len_m: float, marker_len_m: float):
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        square_len_m,
        marker_len_m,
        dictionary
    )
    return dictionary, board


def detect_charuco(gray, dictionary, board):
    """
    Returns:
        charuco_corners: shape (N,1,2) or None
        charuco_ids:     shape (N,1)   or None
        score:           integer count of detected charuco corners
        marker_corners, marker_ids for optional debug drawing
    """
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if marker_ids is None or len(marker_ids) < 2:
        return None, None, 0, marker_corners, marker_ids

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board
    )

    if charuco_ids is None or charuco_corners is None or retval is None:
        return None, None, 0, marker_corners, marker_ids

    return charuco_corners, charuco_ids, int(retval), marker_corners, marker_ids


def save_debug_image(img, marker_corners, marker_ids, charuco_corners, charuco_ids, out_path: Path):
    vis = img.copy()
    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if charuco_ids is not None and charuco_corners is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def build_fisheye_points_single(all_charuco_corners, all_charuco_ids, board):
    """
    Build per-image object/image point lists for cv2.fisheye.calibrate.

    Returns:
        objpoints: list of arrays, each shape (N,1,3), float64
        imgpoints: list of arrays, each shape (N,1,2), float64
    """
    board_corners_3d = board.getChessboardCorners()  # shape (num_charuco_corners, 3)

    objpoints = []
    imgpoints = []

    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        if corners is None or ids is None:
            continue

        ids_flat = ids.flatten()
        if len(ids_flat) < MIN_CHARUCO_CORNERS_PER_VIEW:
            continue

        obj = np.array([board_corners_3d[int(cid)] for cid in ids_flat], dtype=np.float64)
        img = np.array([corners[k, 0] for k in range(len(ids_flat))], dtype=np.float64)

        objpoints.append(obj.reshape(-1, 1, 3))
        imgpoints.append(img.reshape(-1, 1, 2))

    return objpoints, imgpoints


def calibrate_single_camera_fisheye(all_charuco_corners, all_charuco_ids, board, img_size):
    objpoints, imgpoints = build_fisheye_points_single(all_charuco_corners, all_charuco_ids, board)

    if len(objpoints) < MIN_STEREO_PAIRS:
        raise RuntimeError(
            f"Not enough valid views for fisheye single-camera calibration. "
            f"Need at least {MIN_STEREO_PAIRS}, got {len(objpoints)}."
        )

    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-7
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
        criteria=criteria
    )

    return rms, K, D, rvecs, tvecs

def build_fisheye_stereo_points(cornersL_list, idsL_list, cornersR_list, idsR_list, board):
    """
    Build stereo fisheye points by selecting a subset of stereo pairs that
    all share the same ChArUco corner IDs.

    This is a workaround for the OpenCV Python fisheye.stereoCalibrate bug
    when the number of points varies across image pairs.
    """
    board_corners_3d = board.getChessboardCorners()

    pair_data = []

    # Gather per-pair common IDs
    for pair_idx, (cL, iL, cR, iR) in enumerate(zip(cornersL_list, idsL_list, cornersR_list, idsR_list)):
        if cL is None or iL is None or cR is None or iR is None:
            continue

        idsL_flat = iL.flatten()
        idsR_flat = iR.flatten()

        common = np.intersect1d(idsL_flat, idsR_flat)
        if len(common) < MIN_COMMON_STEREO_CORNERS:
            continue

        idxL = {int(cid): k for k, cid in enumerate(idsL_flat)}
        idxR = {int(cid): k for k, cid in enumerate(idsR_flat)}

        pair_data.append({
            "pair_idx": pair_idx,
            "common_ids": set(int(x) for x in common.tolist()),
            "idxL": idxL,
            "idxR": idxR,
            "cL": cL,
            "cR": cR
        })

    if len(pair_data) < MIN_STEREO_PAIRS:
        raise RuntimeError(
            f"Not enough stereo pairs after filtering. Need at least {MIN_STEREO_PAIRS}, got {len(pair_data)}."
        )

    # Greedy search:
    # Try each pair's common-ID set as a seed, then keep only pairs that contain all those IDs.
    # Choose the seed that gives the best tradeoff:
    #   maximize (#pairs_kept * #ids_kept)
    best_ids = None
    best_pairs = None
    best_score = -1

    for seed in pair_data:
        seed_ids = seed["common_ids"]

        compatible_pairs = []
        for pd in pair_data:
            if seed_ids.issubset(pd["common_ids"]):
                compatible_pairs.append(pd)

        n_pairs = len(compatible_pairs)
        n_ids = len(seed_ids)
        score = n_pairs * n_ids

        if n_pairs >= MIN_STEREO_PAIRS and n_ids >= MIN_COMMON_STEREO_CORNERS and score > best_score:
            best_score = score
            best_ids = sorted(seed_ids)
            best_pairs = compatible_pairs

    if best_pairs is None:
        # fallback: try pairwise intersections as seeds
        for i in range(len(pair_data)):
            for j in range(i + 1, len(pair_data)):
                seed_ids = pair_data[i]["common_ids"].intersection(pair_data[j]["common_ids"])
                if len(seed_ids) < MIN_COMMON_STEREO_CORNERS:
                    continue

                compatible_pairs = []
                for pd in pair_data:
                    if seed_ids.issubset(pd["common_ids"]):
                        compatible_pairs.append(pd)

                n_pairs = len(compatible_pairs)
                n_ids = len(seed_ids)
                score = n_pairs * n_ids

                if n_pairs >= MIN_STEREO_PAIRS and n_ids >= MIN_COMMON_STEREO_CORNERS and score > best_score:
                    best_score = score
                    best_ids = sorted(seed_ids)
                    best_pairs = compatible_pairs

    if best_pairs is None:
        raise RuntimeError(
            "Could not find a subset of stereo pairs with a consistent shared set of ChArUco IDs.\n"
            "You likely need to recapture stereo images with the board more fully visible in both cameras."
        )

    print(f"Selected {len(best_pairs)} stereo pairs sharing {len(best_ids)} identical ChArUco IDs.")
    print(f"Shared IDs: {best_ids}")

    objpoints = []
    imgpointsL = []
    imgpointsR = []

    for pd in best_pairs:
        obj = []
        ptsL = []
        ptsR = []

        for cid in best_ids:
            obj.append(board_corners_3d[cid])
            ptsL.append(pd["cL"][pd["idxL"][cid], 0])
            ptsR.append(pd["cR"][pd["idxR"][cid], 0])

        obj = np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3)
        ptsL = np.asarray(ptsL, dtype=np.float64).reshape(-1, 1, 2)
        ptsR = np.asarray(ptsR, dtype=np.float64).reshape(-1, 1, 2)

        objpoints.append(obj)
        imgpointsL.append(ptsL)
        imgpointsR.append(ptsR)

    n0 = objpoints[0].shape[0]
    for i, (o, l, r) in enumerate(zip(objpoints, imgpointsL, imgpointsR)):
        if o.shape[0] != n0 or l.shape[0] != n0 or r.shape[0] != n0:
            raise RuntimeError(f"Internal error: pair {i} has inconsistent point counts after subset selection.")

    return objpoints, imgpointsL, imgpointsR



def compute_stereo_fisheye(K1, D1, K2, D2, objpoints, imgpointsL, imgpointsR, img_size):

    if len(objpoints) < MIN_STEREO_PAIRS:
        raise RuntimeError(
            f"Not enough matched stereo views. Need {MIN_STEREO_PAIRS}, got {len(objpoints)}"
        )

    R_init = np.eye(3, dtype=np.float64)
    T_init = np.zeros((3,), dtype=np.float64)

    flags = cv2.fisheye.CALIB_FIX_INTRINSIC

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-7
    )

    rms, K1o, D1o, K2o, D2o, R, T, rvecs, tvecs = cv2.fisheye.stereoCalibrate(
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
        criteria=criteria
    )

    return rms, K1o, D1o, K2o, D2o, R, T


def main():
    square_len_m = SQUARE_LEN_MM / 1000.0
    marker_len_m = MARKER_LEN_MM / 1000.0

    left_paths, right_paths = load_pairs(PAIRS_DIR)

    test = cv2.imread(left_paths[0])
    if test is None:
        raise RuntimeError("Could not read first left image.")
    h, w = test.shape[:2]
    img_size = (w, h)

    dictionary, board = make_board(ARUCO_DICT_ID, square_len_m, marker_len_m)

    cornersL_list, idsL_list = [], []
    cornersR_list, idsR_list = [], []

    used_pairs = 0

    if SAVE_DEBUG_DETECTIONS:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    for idx, (lp, rp) in enumerate(zip(left_paths, right_paths)):
        imgL = cv2.imread(lp)
        imgR = cv2.imread(rp)

        if imgL is None or imgR is None:
            print(f"Skipping unreadable pair: {lp}, {rp}")
            continue

        gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        cL, iL, sL, mCL, mIL = detect_charuco(gL, dictionary, board)
        cR, iR, sR, mCR, mIR = detect_charuco(gR, dictionary, board)

        if SAVE_DEBUG_DETECTIONS:
            save_debug_image(
                imgL, mCL, mIL, cL, iL,
                DEBUG_DIR / f"left_detect_{idx:03d}.png"
            )
            save_debug_image(
                imgR, mCR, mIR, cR, iR,
                DEBUG_DIR / f"right_detect_{idx:03d}.png"
            )

        if sL >= MIN_CHARUCO_CORNERS_PER_VIEW and sR >= MIN_CHARUCO_CORNERS_PER_VIEW:
            cornersL_list.append(cL)
            idsL_list.append(iL)
            cornersR_list.append(cR)
            idsR_list.append(iR)
            used_pairs += 1

    print(f"Total pairs found: {len(left_paths)}")
    print(f"Pairs used for calibration (>= {MIN_CHARUCO_CORNERS_PER_VIEW} corners in both): {used_pairs}")

    if used_pairs < MIN_STEREO_PAIRS:
        raise RuntimeError(
            f"Not enough good pairs. Need at least {MIN_STEREO_PAIRS}, got {used_pairs}. "
            f"Capture more sharp images with board near center, edges, and with tilt."
        )

    # Single-camera fisheye calibration
    retL, K1, D1, rvecsL, tvecsL = calibrate_single_camera_fisheye(
        cornersL_list, idsL_list, board, img_size
    )
    retR, K2, D2, rvecsR, tvecsR = calibrate_single_camera_fisheye(
        cornersR_list, idsR_list, board, img_size
    )

    print(f"Left fisheye RMS reprojection error:  {retL:.6f}")
    print(f"Right fisheye RMS reprojection error: {retR:.6f}")

    print("\nK1 =\n", K1)
    print("D1 =\n", D1.ravel())
    print("\nK2 =\n", K2)
    print("D2 =\n", D2.ravel())

    # Stereo fisheye calibration
    objpoints, imgpointsL, imgpointsR = build_fisheye_stereo_points(
        cornersL_list, idsL_list, cornersR_list, idsR_list, board
    )
    print("Per-view stereo point counts:")
    for i, (o, l, r) in enumerate(zip(objpoints, imgpointsL, imgpointsR)):
        print(i, o.shape, l.shape, r.shape)

    print(f"\nStereo matched views used: {len(objpoints)}")

    retStereo, K1o, D1o, K2o, D2o, R, T = compute_stereo_fisheye(
        K1, D1, K2, D2, objpoints, imgpointsL, imgpointsR, img_size
    )

    baseline_m = float(np.linalg.norm(T))

    print(f"Stereo fisheye RMS reprojection error: {retStereo:.6f}")
    print(f"Estimated baseline |T|: {baseline_m * 1000.0:.2f} mm")
    print("R =\n", R)
    print("T =\n", T.ravel())

    # Save raw calibration
    np.savez(
        OUT_FILE,
        squaresX=SQUARES_X,
        squaresY=SQUARES_Y,
        squareLength_m=square_len_m,
        markerLength_m=marker_len_m,
        arucoDict=ARUCO_DICT_ID,
        imageSize=np.array(img_size, dtype=np.int32),
        model="fisheye",
        K1=K1o,
        D1=D1o,
        K2=K2o,
        D2=D2o,
        R=R,
        T=T
    )
    print(f"\nSaved calibration to: {OUT_FILE}")

    # Stereo fisheye rectification
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1o, D1o,
        K2o, D2o,
        img_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=img_size,
        balance=0.0,
        fov_scale=1.0
    )

    map1L, map2L = cv2.fisheye.initUndistortRectifyMap(
        K1o, D1o, R1, P1, img_size, cv2.CV_16SC2
    )
    map1R, map2R = cv2.fisheye.initUndistortRectifyMap(
        K2o, D2o, R2, P2, img_size, cv2.CV_16SC2
    )

    np.savez(
        RECTIFY_FILE,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1L=map1L, map2L=map2L,
        map1R=map1R, map2R=map2R
    )
    print(f"Saved rectification data to: {RECTIFY_FILE}")


if __name__ == "__main__":
    main()