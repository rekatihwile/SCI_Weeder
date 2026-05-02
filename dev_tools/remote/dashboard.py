"""Remote dashboard entry point.

Run with:
    ./run_with_eli_venv.sh dev_tools/remote/dashboard.py
"""

import sys
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


app = Flask(__name__)
register_routes(app)


if __name__ == "__main__":
    try:
        print("Starting LaserWeeder debug dashboard...")
        print("Home:    http://0.0.0.0:5000/")
        print("Camera:  http://0.0.0.0:5000/camera")
        print("Survey:  http://0.0.0.0:5000/survey")
        print("Fine:    http://0.0.0.0:5000/fine")
        print("Match:   http://0.0.0.0:5000/match")
        print("Rectify: http://0.0.0.0:5000/rectify")
        print("Gantry:  http://0.0.0.0:5000/gantry")
        print("Workspace3D: http://0.0.0.0:5000/workspace3d")

        app.run(host="0.0.0.0", port=5000, threaded=True)

    finally:
        close_all()
        close_gantry()