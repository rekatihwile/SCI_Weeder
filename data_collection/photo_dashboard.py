from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, send_from_directory, render_template_string

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent  # data_collection/ → workspace root
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import TRAINING_PHOTOS_DIR

app = Flask(__name__)

ROOT_DIR = Path(TRAINING_PHOTOS_DIR)
COMBINED_DIR = ROOT_DIR / "combined"


HTML = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Training Photo Dashboard</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #111;
            color: #eee;
            margin: 20px;
        }
        h1, h2 {
            margin-bottom: 10px;
        }
        .latest img {
            max-width: 95%;
            border: 2px solid #444;
            margin-bottom: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }
        .card {
            background: #1b1b1b;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #333;
        }
        .card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .name {
            margin-top: 8px;
            font-size: 14px;
            word-break: break-all;
        }
    </style>
</head>
<body>
    <h1>Training Photo Dashboard</h1>

    {% if latest %}
    <h2>Latest Combined Image</h2>
    <div class="latest">
        <img src="/images/{{ latest }}" alt="{{ latest }}">
        <div>{{ latest }}</div>
    </div>
    {% endif %}

    <h2>All Combined Images</h2>
    <div class="grid">
        {% for img in images %}
        <div class="card">
            <img src="/images/{{ img }}" alt="{{ img }}">
            <div class="name">{{ img }}</div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


def get_images():
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(
        [p.name for p in COMBINED_DIR.glob("*.jpg")],
        reverse=True
    )
    return images


@app.route("/")
def index():
    images = get_images()
    latest = images[0] if images else None
    return render_template_string(HTML, images=images, latest=latest)


@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(COMBINED_DIR, filename)


if __name__ == "__main__":
    print("Dashboard folder:", ROOT_DIR.resolve())
    print("Open: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)