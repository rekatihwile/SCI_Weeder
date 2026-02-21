import cv2
import json
import time
import sys
import os
import threading
import numpy as np
from pathlib import Path
from flask import Flask, Response, request, render_template_string

# Local imports
from cv_helpers import WeedCV

# --- CONFIGURATION ---
STREAM_ONLY = False  # Set to True to disable AI and just see the raw feed for tuning

app = Flask(__name__)
output_frame = None
frame_ready = False # Flag to signal a new AI-processed frame is ready
lock = threading.Lock()

# Base defaults (used only if the config file doesn't exist yet or is missing keys)
config = {
    "left":  {"brightness": 15, "contrast": 30, "exposure": 350, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100},
    "right": {"brightness": 15, "contrast": 30, "exposure": 350, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100}
}

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
HW_CFG_PATH = BASE_DIR / "hardware_config.json"
CAM_CFG_PATH = BASE_DIR / "camera_config.json"
WEIGHTS_DIR = BASE_DIR / "weights"
YOLO_PT = str(WEIGHTS_DIR / "pigweed-yolo.pt")
SNIPER_PT = str(WEIGHTS_DIR / "sniper.pt")   

IS_WINDOWS = sys.platform.startswith('win')
HAS_DISPLAY = os.environ.get('DISPLAY') is not None or IS_WINDOWS
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

# --- LOAD SAVED CONFIG EARLY ---
# This ensures the dictionary is populated with your saved JSON values 
# BEFORE the Flask web dashboard ever starts.
if CAM_CFG_PATH.exists():
    try:
        with open(CAM_CFG_PATH, 'r') as f:
            config.update(json.load(f))
    except Exception as e:
        print(f"Could not load {CAM_CFG_PATH}: {e}")

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SCI_Weeder Remote Dashboard</title>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; text-align: center; padding: 10px; }
        .stream-box { width: 100%; max-width: 1000px; border: 2px solid #444; margin: 0 auto 20px; background: #000; }
        .stream-img { width: 100%; display: block; }
        .controls-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; width: 100%; }
        .column { background: #222; padding: 10px; border-radius: 8px; border: 1px solid #333; }
        .slider-group { margin-bottom: 10px; text-align: left; font-size: 0.8em; }
        input[type=range] { width: 100%; }
        .btn-save { margin-top: 20px; padding: 10px 30px; background: #28a745; color: white; border: none; cursor: pointer; border-radius: 4px; }
    </style>
    <script>
        function updateParam(side, param, val) {
            document.getElementById(side + '_' + param + '_val').innerText = val;
            fetch(`/set_param?side=${side}&param=${param}&val=${val}`);
        }
        function saveConfig() { fetch('/save').then(r => r.ok ? alert("Exposure Saved!") : alert("Failed")); }
    </script>
</head>
<body>
    <h3>🌿 SCI_Weeder Exposure Control</h3>
    <div class="stream-box"><img src="/video" class="stream-img"></div>
    <div class="controls-grid">
        {% for side in ['left', 'right'] %}
        <div class="column">
            <strong>{{ side.upper() }} CAMERA</strong>
            <div class="slider-group">
                <label>Exposure: <span id="{{ side }}_exposure_val">{{ config[side]['exposure'] }}</span></label>
                <input type="range" min="0" max="1000" value="{{ config[side]['exposure'] }}" 
                       oninput="updateParam('{{ side }}', 'exposure', this.value)">
            </div>
        </div>
        {% endfor %}
    </div>
    <button class="btn-save" onclick="saveConfig()">SAVE EXPOSURE TO JSON</button>
</body>
</html>
"""

@app.route('/')
def index(): 
    return render_template_string(HTML_TEMPLATE, config=config)

@app.route('/set_param')
def set_param():
    side, param, val = request.args.get('side'), request.args.get('param'), int(request.args.get('val'))
    if side in config: config[side][param] = val
    return "OK"

@app.route('/save')
def save():
    try:
        # 1. Load existing file to preserve all other settings (brightness, contrast, etc.)
        if CAM_CFG_PATH.exists():
            with open(CAM_CFG_PATH, 'r') as f:
                current_file_data = json.load(f)
        else:
            current_file_data = config # Fallback to default if file doesn't exist

        # 2. Update ONLY the exposure from our active runtime config
        for side in ['left', 'right']:
            if side in current_file_data and side in config:
                current_file_data[side]['exposure'] = config[side]['exposure']

        # 3. Write the merged data back
        with open(CAM_CFG_PATH, 'w') as f:
            json.dump(current_file_data, f, indent=4)
        return "OK"
    except Exception as e:
        print(f"Save Error: {e}")
        return "Error", 500

@app.route('/video')
def video():
    def generate():
        global frame_ready, output_frame
        while True:
            # Wait for the AI loop to signal that a new frame is processed
            if not frame_ready or output_frame is None:
                time.sleep(0.01) # Tiny sleep to prevent CPU pinning
                continue
            
            with lock:
                # Encode with higher quality (80+) since we are sending fewer frames
                ret, jpeg = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
                frame_ready = False # Reset the flag after grabbing the frame
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def update_camera(cap, props):
    # Apply all parameters currently held in memory
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    cap.set(cv2.CAP_PROP_BRIGHTNESS, props['brightness'])
    cap.set(cv2.CAP_PROP_CONTRAST, props['contrast'])
    cap.set(cv2.CAP_PROP_EXPOSURE, props['exposure'])
    cap.set(cv2.CAP_PROP_GAIN, props['gain'])
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, props['white_balance'])


def main():
    global output_frame, config, frame_ready
    if not HW_CFG_PATH.exists(): return

    # if not HAS_DISPLAY:
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True), daemon=True).start()

    ai_L = WeedCV(YOLO_PT, SNIPER_PT) if not STREAM_ONLY else None
    ai_R = WeedCV(YOLO_PT, SNIPER_PT) if not STREAM_ONLY else None
    
    with open(HW_CFG_PATH, 'r') as f: hw = json.load(f)
    cap_l = cv2.VideoCapture(hw['cameras']['left']['index'], BACKEND)
    cap_r = cv2.VideoCapture(hw['cameras']['right']['index'], BACKEND)

    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_applied = {"left": None, "right": None}

    while True:
        for side, cap in [("left", cap_l), ("right", cap_r)]:
            if config[side] != last_applied[side]:
                update_camera(cap, config[side])
                last_applied[side] = config[side].copy()

        ret_l, f_l = cap_l.read()
        ret_r, f_r = cap_r.read()

        if ret_l and ret_r:
            if not STREAM_ONLY:
                # Run AI and draw markers
                for ai, frame in [(ai_L, f_l), (ai_R, f_r)]:
                    for (x, y) in ai.return_full(frame):
                        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                        cv2.drawMarker(frame, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
            with lock:
                output_frame = cv2.hconcat([f_l, f_r])
                frame_ready = True

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_l.release(); cap_r.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()