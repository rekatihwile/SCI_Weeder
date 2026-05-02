"""
pipeline/preflight.py
---------------------
Print a human-readable preflight summary of the runtime environment and key
config flags.  Does NOT touch any hardware.
"""

import subprocess
import sys


def _git_head(cwd=None):
    """Return the current git HEAD hash + branch (or detached note), or 'unknown'."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Try to get branch name; will fail gracefully if detached.
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            branch = "detached"
        return f"{sha} ({branch})"
    except Exception:
        return "unknown"


def print_preflight():
    """Print preflight info to stdout.  No hardware is opened or initialised."""
    import os
    from pathlib import Path

    # Resolve the repo root relative to this file.
    repo_root = Path(__file__).resolve().parent.parent

    # Import packages — each import is guarded so a missing package is reported
    # rather than raising an uncaught exception.
    def _version(pkg_name, attr="__version__"):
        try:
            import importlib
            mod = importlib.import_module(pkg_name)
            return getattr(mod, attr, "unknown")
        except ImportError:
            return "NOT INSTALLED"

    try:
        import cv2
        cv2_ver = cv2.__version__
    except ImportError:
        cv2_ver = "NOT INSTALLED"

    try:
        import numpy as np
        np_ver = np.__version__
    except ImportError:
        np_ver = "NOT INSTALLED"

    try:
        import PIL
        pil_ver = PIL.__version__
    except ImportError:
        pil_ver = "NOT INSTALLED"

    try:
        import torch
        torch_ver = torch.__version__
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        torch_ver = "NOT INSTALLED"
        cuda_ok = False

    try:
        import ultralytics
        ult_ver = ultralytics.__version__
    except ImportError:
        ult_ver = "NOT INSTALLED"

    try:
        import serial
        serial_ver = serial.VERSION
    except ImportError:
        serial_ver = "NOT INSTALLED"

    # Config — import lazily so preflight can run without hardware JSON present;
    # on failure each value is reported as "config import failed".
    cfg_ok = False
    try:
        import config as cfg
        cfg_ok = True
    except Exception as e:
        cfg_err = str(e)

    def _cfg(attr, default="<config import failed>"):
        if cfg_ok:
            return getattr(cfg, attr, "<missing>")
        return default

    git_head = _git_head(cwd=str(repo_root))

    print("=" * 62)
    print("  PREFLIGHT CHECK")
    print("=" * 62)
    print(f"  Python executable : {sys.executable}")
    print(f"  Python version    : {sys.version.split()[0]}")
    print()
    print("  --- Package Versions ---")
    print(f"  cv2               : {cv2_ver}")
    print(f"  numpy             : {np_ver}")
    print(f"  PIL               : {pil_ver}")
    print(f"  torch             : {torch_ver}  cuda={cuda_ok}")
    print(f"  ultralytics       : {ult_ver}")
    print(f"  pyserial          : {serial_ver}")
    print()
    print("  --- Repository ---")
    print(f"  git HEAD          : {git_head}")
    print()
    if not cfg_ok:
        print(f"  [WARNING] config.py failed to import: {cfg_err}")
        print("=" * 62)
        return
    print("  --- Hardware Config ---")
    print(f"  LEFT_CAMERA_INDEX : {_cfg('LEFT_CAMERA_INDEX')}")
    print(f"  RIGHT_CAMERA_INDEX: {_cfg('RIGHT_CAMERA_INDEX')}")
    print(f"  GRBL_PORT         : {_cfg('GRBL_PORT')}")
    print()
    print("  --- Operator Toggles ---")
    print(f"  RECORD_TRIAL      : {_cfg('RECORD_TRIAL')}")
    print(f"  YOLO_WARMUP       : {_cfg('YOLO_WARMUP')}")
    print(f"  HOMING            : {_cfg('HOMING')}")
    print(f"  TRIANGULATION_ONLY_MODE : {_cfg('TRIANGULATION_ONLY_MODE')}")
    print(f"  FULL_AUTO         : {_cfg('FULL_AUTO')}")
    print("=" * 62)
