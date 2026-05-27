"""Remote dashboard entry point.

Run with:
    ./run_with_eli_venv.sh dev_tools/remote/dashboard.py
"""

import getpass
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask

# Allow sibling imports when this file is run as a script.
REMOTE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REMOTE_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard_routes import register_routes
from dashboard_camera import close_all
from dashboard_gantry import close_gantry
from dashboard_scout import close_scout


app = Flask(__name__)
register_routes(app)


def _listening_pids(port: int):
    pids = set()

    # Prefer lsof for a direct pid list.
    try:
        out = subprocess.check_output(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for raw in out.splitlines():
            raw = raw.strip()
            if raw.isdigit():
                pids.add(int(raw))
        return sorted(pids)
    except Exception:
        pass

    # Fallback if lsof is unavailable.
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if f":{port}" not in line:
                continue
            for match in re.finditer(r"pid=(\d+)", line):
                pids.add(int(match.group(1)))
    except Exception:
        pass

    return sorted(pids)


def _pid_owner(pid: int):
    try:
        out = subprocess.check_output(
            ["ps", "-o", "user=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        owner = out.strip()
        return owner if owner else None
    except Exception:
        return None


def _free_port(port: int):
    this_pid = os.getpid()
    this_user = getpass.getuser()
    candidate_pids = _listening_pids(port)

    if not candidate_pids:
        return

    target_pids = []
    for pid in candidate_pids:
        if pid == this_pid:
            continue
        owner = _pid_owner(pid)
        if owner is not None and owner != this_user:
            continue
        target_pids.append(pid)

    if not target_pids:
        return

    print(f"Port {port} busy. Stopping stale process(es): {target_pids}")
    for pid in target_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[WARN] No permission to stop pid {pid} on port {port}.")

    # Give processes a brief chance to exit cleanly.
    deadline = time.time() + 2.0
    remaining = set(target_pids)
    while time.time() < deadline and remaining:
        live = set(_listening_pids(port))
        remaining = remaining.intersection(live)
        if not remaining:
            break
        time.sleep(0.1)

    if remaining:
        print(f"Port {port} still busy after SIGTERM. Forcing stop: {sorted(remaining)}")
        for pid in sorted(remaining):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                print(f"[WARN] No permission to force-stop pid {pid} on port {port}.")


if __name__ == "__main__":
    try:
        _free_port(5000)
        print("Starting LaserWeeder debug dashboard...")
        print("Home:    http://0.0.0.0:5000/")
        print("Camera:  http://0.0.0.0:5000/camera")
        print("Survey Photos: http://0.0.0.0:5000/survey_photos")
        print("Survey:  http://0.0.0.0:5000/survey")
        print("Fine:    http://0.0.0.0:5000/fine")
        print("Match:   http://0.0.0.0:5000/match")
        print("Rectify: http://0.0.0.0:5000/rectify")
        print("Gantry:      http://0.0.0.0:5000/gantry")
        print("Workspace3D: http://0.0.0.0:5000/workspace3d")
        print("Fine Align:  http://0.0.0.0:5000/fine_align")
        print("Scout:       http://0.0.0.0:5000/scout")

        app.run(host="0.0.0.0", port=5000, threaded=True)

    finally:
        close_all()
        close_gantry()
        close_scout()
