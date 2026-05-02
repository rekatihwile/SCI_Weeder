"""
bringup/00_env_check.py
-----------------------
Verify runtime environment only. No hardware.

Run with:
    ./run_with_eli_venv.sh bringup/00_env_check.py | tee bringup/logs/00_env_check.log
"""

import sys
import subprocess
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = True
errors = []


def check(label, fn):
    global PASS
    try:
        result = fn()
        print(f"  {label}: {result}")
        return result
    except Exception as e:
        print(f"  {label}: FAILED — {e}")
        PASS = False
        errors.append(f"{label}: {e}")
        return None


def main():
    global PASS

    print("=" * 60)
    print("BRINGUP 00 — Environment Check")
    print("=" * 60)

    print("\n--- Python ---")
    print(f"  sys.executable : {sys.executable}")
    print(f"  Python version : {sys.version}")

    print("\n--- Core Libraries ---")

    check("cv2.__version__", lambda: __import__("cv2").__version__)
    check("numpy.__version__", lambda: __import__("numpy").__version__)

    pil_ver = check("PIL.__version__", lambda: __import__("PIL").__version__)

    torch_ver = check("torch.__version__", lambda: __import__("torch").__version__)

    cuda_ok = check(
        "torch.cuda.is_available()",
        lambda: __import__("torch").cuda.is_available()
    )
    if not cuda_ok:
        PASS = False
        errors.append("torch.cuda.is_available() returned False")

    check("ultralytics.__version__", lambda: __import__("ultralytics").__version__)

    def _serial_version():
        import serial
        return getattr(serial, "__version__", getattr(serial, "VERSION", "unknown"))
    check("pyserial version", _serial_version)

    print("\n--- Git HEAD ---")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent)
        )
        git_head = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        git_head = "unavailable"
    print(f"  git HEAD: {git_head}")

    print("\n--- Config Values ---")
    try:
        from config import (
            LEFT_CAMERA_INDEX,
            RIGHT_CAMERA_INDEX,
            GRBL_PORT,
            HOMING,
            MOCK_GANTRY,
            RECORD_TRIAL,
            YOLO_WARMUP,
            DETECTOR_MODE,
        )
        print(f"  LEFT_CAMERA_INDEX  : {LEFT_CAMERA_INDEX}")
        print(f"  RIGHT_CAMERA_INDEX : {RIGHT_CAMERA_INDEX}")
        print(f"  GRBL_PORT          : {GRBL_PORT}")
        print(f"  HOMING             : {HOMING}")
        print(f"  MOCK_GANTRY        : {MOCK_GANTRY}")
        print(f"  RECORD_TRIAL       : {RECORD_TRIAL}")
        print(f"  YOLO_WARMUP        : {YOLO_WARMUP}")
        print(f"  DETECTOR_MODE      : {DETECTOR_MODE}")
    except Exception as e:
        print(f"  config import FAILED: {e}")
        PASS = False
        errors.append(f"config import: {e}")

    print("\n" + "=" * 60)
    if PASS:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        for err in errors:
            print(f"  ERROR: {err}")
    print("=" * 60)

    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
