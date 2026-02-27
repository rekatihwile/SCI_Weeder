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
STREAM_ONLY = False  

app = Flask(__name__)
output_frame = None
frame_ready = False 
lock = threading.Lock()

config = {
    "left":  {"brightness": 15, "contrast": 30, "exposure": 350, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100},
    "right": {"brightness": 15, "contrast": 30, "exposure": 350, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100}
}

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
HW_CFG_PATH = BASE_DIR / "hardware_config.json"
CAM_CFG_PATH = BASE_DIR / "camera_config.json"
WEIGHTS_DIR = BASE_DIR / "weights"
YOLO_PT = str(WEIGHTS_DIR / "yolo_w_kale.pt")
SNIPER_PT = str(WEIGHTS_DIR / "sniper.pt")   

IS_WINDOWS = sys.platform.startswith('win')
# Check if a display is available (Windows always true, Linux depends on X11/Wayland display var)
HAS_DISPLAY = os.environ.get('DISPLAY') is not None or IS_WINDOWS
BACKEND = cv2.CAP_MSMF if IS_WINDOWS else cv2.CAP_V4L2

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
        function saveConfig() { fetch('/save').then(r => r.ok ? alert("Saved!") : alert("Failed")); }
    </script>
</head>
<body>
    <h3>🌿 SCI_Weeder Remote Dashboard</h3>
    <div class="stream-box"><img src="/video" class="stream-img"></div>
    <div class="controls-grid">
        {% for side in ['left', 'right'] %}
        <div class="column">
            <strong>{{ side.upper() }} CAMERA</strong>
            {% set sliders = [('brightness', 'B', 0, 32), ('contrast', 'C', 10, 50), ('exposure', 'E', 0, 1000), ('gain', 'G', 0, 255), ('saturation', 'S', 0, 255), ('white_balance', 'WB', 2800, 6500), ('sharpness', 'SH', 0, 255)] %}
            {% for key, label, min, max in sliders %}
            <div class="slider-group">
                <label>{{ label }}: <span id="{{ side }}_{{ key }}_val">{{ config[side][key] }}</span></label>
                <input type="range" min="{{ min }}" max="{{ max }}" value="{{ config[side][key] }}" oninput="updateParam('{{ side }}', '{{ key }}', this.value)">
            </div>
            {% endfor %}
        </div>
        {% endfor %}
    </div>
    <button class="btn-save" onclick="saveConfig()">SAVE TO JSON</button>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE, config=config)

@app.route('/set_param')
def set_param():
    side, param, val = request.args.get('side'), request.args.get('param'), int(request.args.get('val'))
    if side in config: config[side][param] = val
    return "OK"

@app.route('/save')
def save():
    with open(CAM_CFG_PATH, 'w') as f: json.dump(config, f, indent=4)
    return "OK"

@app.route('/video')
def video():
    def generate():
        global frame_ready, output_frame
        while True:
            if not frame_ready or output_frame is None:
                time.sleep(0.01)
                continue
            with lock:
                ret, jpeg = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
                frame_ready = False 
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def update_camera(cap, props):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    cap.set(cv2.CAP_PROP_BRIGHTNESS, props['brightness'])
    cap.set(cv2.CAP_PROP_CONTRAST, props['contrast'])
    cap.set(cv2.CAP_PROP_EXPOSURE, props['exposure'])
    cap.set(cv2.CAP_PROP_GAIN, props['gain'])
    cap.set(cv2.CAP_PROP_SATURATION, props['saturation'])
    cap.set(cv2.CAP_PROP_SHARPNESS, props['sharpness'])
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, props['white_balance'])

def process_frame_visuals(frame, coords, side_config):
    """Draws target markers and HUD directly onto the frame."""
    for det in coords:
        # Unpack based on length to maintain backward compatibility if testing old weights
        if len(det) >= 3:
            x, y, cls_id = det[:3]
        else:
            x, y = det[:2]
            cls_id = 0  # Default fallback
            
        # ID 1 is Kale (Green), IDs 0 & 2 are Weeds (Red)
        color = (0, 255, 0) if cls_id == 1 else (0, 0, 255)

        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.drawMarker(frame, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
        
    # Draw HUD
    c = side_config
    hud = [
        f"B:{c['brightness']} C:{c['contrast']} E:{c['exposure']}",
        f"G:{c['gain']} S:{c['saturation']} WB:{c['white_balance']} SH:{c['sharpness']}"
    ]
    for i, text in enumerate(hud):
        cv2.putText(frame, text, (15, 30 + (i * 30)), 1, 1.2, (0, 255, 0), 2)
    
    return frame

def main():
    global output_frame, config, frame_ready
    if not HW_CFG_PATH.exists():
        print("❌ Error: hardware_config.json not found.")
        return

    # Start Flask if headlessly running
    if not HAS_DISPLAY:
        print("🌐 No display detected. Starting Flask remote dashboard on port 5000...")
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True), daemon=True).start()
    else:
        print("🖥️ Display detected. Starting local OpenCV tuning windows...")

    print("🧠 Initializing AI Targeter...")
    ai_L = WeedCV(YOLO_PT, SNIPER_PT) if not STREAM_ONLY else None
    ai_R = WeedCV(YOLO_PT, SNIPER_PT) if not STREAM_ONLY else None
    
    with open(HW_CFG_PATH, 'r') as f: hw = json.load(f)
    cap_l = cv2.VideoCapture(hw['cameras']['left']['index'], BACKEND)
    cap_r = cv2.VideoCapture(hw['cameras']['right']['index'], BACKEND)

    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 

    if CAM_CFG_PATH.exists():
        with open(CAM_CFG_PATH, 'r') as f: config.update(json.load(f))

    # UI Setup for Local Display
    win_l, win_r = "LEFT_TUNER", "RIGHT_TUNER"
    if HAS_DISPLAY:
        cv2.namedWindow(win_l); cv2.namedWindow(win_r)
        def setup_sliders(win, side):
            c = config[side]
            cv2.createTrackbar("Brightness", win, c["brightness"], 32, lambda x: None)
            cv2.createTrackbar("Contrast",   win, max(0, c["contrast"]-10), 40, lambda x: None)
            exp_init = int(abs(c["exposure"]) / 13 * 100) if IS_WINDOWS else c["exposure"]
            cv2.createTrackbar("Fine_Expos", win, exp_init, 100 if IS_WINDOWS else 1000, lambda x: None)
            cv2.createTrackbar("Gain",       win, c["gain"], 255, lambda x: None)
            cv2.createTrackbar("Saturation", win, c["saturation"], 255, lambda x: None)
            cv2.createTrackbar("WB_Temp",    win, int((c["white_balance"]-2800)/37), 100, lambda x: None)
            cv2.createTrackbar("Sharpness",  win, c["sharpness"], 255, lambda x: None)

        setup_sliders(win_l, "left")
        setup_sliders(win_r, "right")

    last_applied = {"left": None, "right": None}

    while True:
        # Sync trackbars to global config if running locally
        if HAS_DISPLAY:
            for side, win in [("left", win_l), ("right", win_r)]:
                config[side]["brightness"] = cv2.getTrackbarPos("Brightness", win)
                config[side]["contrast"]   = cv2.getTrackbarPos("Contrast", win) + 10
                raw_ex = cv2.getTrackbarPos("Fine_Expos", win)
                config[side]["exposure"]   = -int((raw_ex/100)*13) if IS_WINDOWS else raw_ex
                config[side]["gain"]       = cv2.getTrackbarPos("Gain", win)
                config[side]["saturation"] = cv2.getTrackbarPos("Saturation", win)
                config[side]["white_balance"] = cv2.getTrackbarPos("WB_Temp", win) * 37 + 2800
                config[side]["sharpness"]  = cv2.getTrackbarPos("Sharpness", win)

        # Apply settings if changed
        for side, cap in [("left", cap_l), ("right", cap_r)]:
            if config[side] != last_applied[side]:
                update_camera(cap, config[side])
                last_applied[side] = config[side].copy()
                time.sleep(0.01)

        ret_l, f_l = cap_l.read()
        ret_r, f_r = cap_r.read()

        if ret_l and ret_r:
            coords_L, coords_R = [], []
            if not STREAM_ONLY:
                coords_L = ai_L.return_full(f_l) if ai_L else []
                coords_R = ai_R.return_full(f_r) if ai_R else []

            # Draw everything onto the frames
            f_l = process_frame_visuals(f_l, coords_L, config['left'])
            f_r = process_frame_visuals(f_r, coords_R, config['right'])

            if HAS_DISPLAY:
                cv2.imshow(win_l, f_l)
                cv2.imshow(win_r, f_r)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    with open(CAM_CFG_PATH, 'w') as f:
                        json.dump(config, f, indent=4)
                    print("✅ PERSISTED: Configuration saved.")
                elif key == ord('q'):
                    break
            else:
                # Update globals for Flask stream
                with lock:
                    output_frame = cv2.hconcat([f_l, f_r])
                    frame_ready = True

    cap_l.release(); cap_r.release()
    if HAS_DISPLAY:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()