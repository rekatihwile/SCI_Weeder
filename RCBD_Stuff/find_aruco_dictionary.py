import cv2
import glob
from pathlib import Path

# Change this if your images live somewhere else
IMG_GLOB = str(Path(__file__).resolve().parent / "calib_pairs" / "left_*.png")
SHOW_BEST_OVERLAY = True  # set False if you don't want a window

# All common ArUco dictionaries OpenCV supports
DICT_LIST = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
    ("DICT_5X5_1000", cv2.aruco.DICT_5X5_1000),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_6X6_1000", cv2.aruco.DICT_6X6_1000),
    ("DICT_7X7_50", cv2.aruco.DICT_7X7_50),
    ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
    ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
    ("DICT_7X7_1000", cv2.aruco.DICT_7X7_1000),
    ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
]

def detect_markers(gray, dictionary):
    # Works on newer OpenCV (ArucoDetector) and older (detectMarkers)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    n = 0 if ids is None else len(ids)
    return corners, ids, n

def main():
    paths = sorted(glob.glob(IMG_GLOB))
    if not paths:
        print(f"No images found. Check IMG_GLOB = {IMG_GLOB}")
        return

    # Use a subset (fast) but enough to be confident
    sample = paths[0:min(len(paths), 10)]
    print(f"Testing {len(sample)} images...")

    scores = []
    for name, did in DICT_LIST:
        dictionary = cv2.aruco.getPredefinedDictionary(did)
        total = 0
        best_one = 0
        for p in sample:
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, _, n = detect_markers(gray, dictionary)
            total += n
            best_one = max(best_one, n)

        scores.append((total, best_one, name, did))
        print(f"{name:20s}  total={total:3d}  best_in_one_image={best_one:2d}")

    scores.sort(reverse=True, key=lambda x: x[0])

    best_total, best_one, best_name, best_id = scores[0]
    print("\n=== BEST DICTIONARY ===")
    print(f"{best_name}  (total markers across sample={best_total}, best single image={best_one})")
    print(f"Use this in your calibration script: cv2.aruco.{best_name}")

    if SHOW_BEST_OVERLAY and best_total > 0:
        # Show overlay on the first image where it detects something
        dictionary = cv2.aruco.getPredefinedDictionary(best_id)
        for p in sample:
            img = cv2.imread(p)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, n = detect_markers(gray, dictionary)
            if n > 0:
                vis = img.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.imshow(f"Detected with {best_name} (press any key)", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main()