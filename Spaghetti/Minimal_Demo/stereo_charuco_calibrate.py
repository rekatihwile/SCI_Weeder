import cv2
import numpy as np
from pathlib import Path
import glob

# =========================
# USER SETTINGS (EDIT THESE)
# =========================
PAIRS_DIR = Path(__file__).resolve().parent / "calib_pairs"

# Your board is 7 squares wide by 8 squares tall (from your photo/message)
SQUARES_X = 8
SQUARES_Y = 8

# Measure these with a ruler/caliper (in mm) and put the values here:
SQUARE_LEN_MM = 30.0   # <-- EDIT: chess square edge length (mm)
MARKER_LEN_MM = 14.0   # <-- EDIT: aruco marker edge length (mm)

# If you know the exact dictionary used to print the board, set it here.
# Otherwise, the script will try a few common ones and pick the best.
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_1000



# Calibration output file
OUT_FILE = Path(__file__).resolve().parent / "stereo_charuco_calib.npz"


def load_pairs(pairs_dir: Path):
    lefts = sorted(glob.glob(str(pairs_dir / "left_*.png")))
    rights = sorted(glob.glob(str(pairs_dir / "right_*.png")))
    if len(lefts) == 0 or len(rights) == 0:
        raise FileNotFoundError(f"No image pairs found in {pairs_dir} (expected left_*.png and right_*.png)")
    if len(lefts) != len(rights):
        print(f"Warning: left count {len(lefts)} != right count {len(rights)}. Using min.")
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
    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) < 4:
        return None, None, 0

    # Refine marker detection (optional but helps)
    # cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)

    # Interpolate ChArUco corners
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if charuco_ids is None or charuco_corners is None:
        return None, None, 0

    # Use number of detected Charuco corners as score
    return charuco_corners, charuco_ids, int(retval)



def calibrate_single_camera(all_charuco_corners, all_charuco_ids, board, img_size):
    # Returns K, dist, rvecs, tvecs
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=criteria
    )
    return ret, K, dist, rvecs, tvecs


def main():
    square_len_m = SQUARE_LEN_MM / 1000.0
    marker_len_m = MARKER_LEN_MM / 1000.0

    left_paths, right_paths = load_pairs(PAIRS_DIR)

    # Determine image size
    test = cv2.imread(left_paths[0])
    if test is None:
        raise RuntimeError("Could not read first left image.")
    h, w = test.shape[:2]
    img_size = (w, h)

    # Pick dictionary (or set DICT_CANDIDATES to only one if you know it)
    dict_id = ARUCO_DICT_ID
    dictionary, board = make_board(dict_id, square_len_m, marker_len_m)                         

    # Collect detections
    cornersL_list, idsL_list = [], []
    cornersR_list, idsR_list = [], []
    used_pairs = 0

    for lp, rp in zip(left_paths, right_paths):
        imgL = cv2.imread(lp)
        imgR = cv2.imread(rp)
        if imgL is None or imgR is None:
            continue

        gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        cL, iL, sL = detect_charuco(gL, dictionary, board)
        cR, iR, sR = detect_charuco(gR, dictionary, board)

        # Require decent corner count on both views for this pair
        if sL >= 12 and sR >= 12:
            cornersL_list.append(cL)
            idsL_list.append(iL)
            cornersR_list.append(cR)
            idsR_list.append(iR)
            used_pairs += 1

    print(f"Total pairs found: {len(left_paths)}")
    print(f"Pairs used for calibration (>=12 charuco corners in both): {used_pairs}")

    if used_pairs < 10:
        raise RuntimeError("Not enough good pairs. Capture more images with clearer board views / less blur / more tilt.")

    # Calibrate each camera intrinsics
    retL, K1, D1, rvecsL, tvecsL = calibrate_single_camera(cornersL_list, idsL_list, board, img_size)
    retR, K2, D2, rvecsR, tvecsR = calibrate_single_camera(cornersR_list, idsR_list, board, img_size)

    print(f"Left reprojection error:  {retL:.4f}")
    print(f"Right reprojection error: {retR:.4f}")

    # Stereo calibration
    # We need object points + matched image points for both cameras.
    # For Charuco, we can use board.matchImagePoints on each view.


    objpoints = []
    imgpointsL = []
    imgpointsR = []

    # board corner 3D coordinates (in board frame)
    # OpenCV stores chessboard corner coordinates in board.getChessboardCorners()
    board_corners_3d = board.getChessboardCorners()  # shape: (N, 3)

    for cL, iL, cR, iR in zip(cornersL_list, idsL_list, cornersR_list, idsR_list):
        if iL is None or iR is None:
            continue

        idsL = iL.flatten()
        idsR = iR.flatten()

        # intersection of corner IDs
        common = np.intersect1d(idsL, idsR)
        if len(common) < 6:
            continue

        # map id -> index in each list
        idxL = {int(cid): k for k, cid in enumerate(idsL)}
        idxR = {int(cid): k for k, cid in enumerate(idsR)}

        # build matched lists
        obj = []
        ptsL = []
        ptsR = []
        for cid in common:
            cid = int(cid)
            obj.append(board_corners_3d[cid])          # (3,)
            ptsL.append(cL[idxL[cid]][0])              # (2,)
            ptsR.append(cR[idxR[cid]][0])              # (2,)

        objpoints.append(np.array(obj, dtype=np.float32))
        imgpointsL.append(np.array(ptsL, dtype=np.float32))
        imgpointsR.append(np.array(ptsR, dtype=np.float32))

    if len(objpoints) < 10:
        raise RuntimeError("Not enough matched board points for stereoCalibrate. Try capturing sharper/clearer pairs.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    flags = cv2.CALIB_FIX_INTRINSIC  # keep K1,D1,K2,D2 fixed, solve only R,t

    retStereo, K1o, D1o, K2o, D2o, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpointsL,
        imagePoints2=imgpointsR,
        cameraMatrix1=K1,
        distCoeffs1=D1,
        cameraMatrix2=K2,
        distCoeffs2=D2,
        imageSize=img_size,
        criteria=criteria,
        flags=flags
    )

    baseline_m = float(np.linalg.norm(T))
    print(f"Stereo reprojection error: {retStereo:.4f}")
    print(f"Estimated baseline |T|: {baseline_m*1000.0:.2f} mm")

    # Save results
    np.savez(
        OUT_FILE,
        squaresX=SQUARES_X,
        squaresY=SQUARES_Y,
        squareLength_m=square_len_m,
        markerLength_m=marker_len_m,
        arucoDict=dict_id,
        imageSize=np.array(img_size, dtype=np.int32),
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        R=R, T=T,
        E=E, F=F
    )

    print(f"\nSaved calibration to: {OUT_FILE}")

    # Optional: quick rectification matrices for later use (handy for debugging)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K1, distCoeffs1=D1,
        cameraMatrix2=K2, distCoeffs2=D2,
        imageSize=img_size,
        R=R, T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    np.savez(
        OUT_FILE.with_name("stereo_rectify_maps.npz"),
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q
    )
    print(f"Saved rectification matrices to: {OUT_FILE.with_name('stereo_rectify_maps.npz')}")


if __name__ == "__main__":
    main()