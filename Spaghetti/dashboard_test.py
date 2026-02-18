import cv2
import json
import numpy as np
from flask import Flask, Response, request, render_template_string
import threading

app = Flask(__name__)

# --- GLOBAL STATE ---
# Initializing with the same default values from your Jetson script
config = {
    "left":  {"brightness": 15, "contrast": 30, "exposure": 350, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100},
    "right": {"brightness": 15, "contrast": 30, "exposure": 350, "gain": 0, "saturation": 64, "white_balance": 4500, "sharpness": 100}
}
output_frame = None
lock = threading.Lock()

# --- HTML TEMPLATE ---
# Modern Dark UI with Two-Column Slider Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SCI_Weeder Remote Dashboard</title>
    <style>
        body { background: #111; color: #eee; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; text-align: center; }
        .container { display: flex; flex-direction: column; align-items: center; }
        .stream-box { width: 90%; max-width: 1200px; border: 3px solid #444; border-radius: 8px; overflow: hidden; margin-bottom: 20px; background: #000; }
        .stream-img { width: 100%; display: block; }
        
        .controls-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; width: 90%; max-width: 1200px; }
        .column { background: #222; padding: 20px; border-radius: 12px; border: 1px solid #333; }
        h2 { color: #00ffcc; border-bottom: 1px solid #444; padding-bottom: 10px; margin-top: 0; }
        
        .slider-group { margin-bottom: 15px; text-align: left; }
        label { display: block; font-size: 0.9em; margin-bottom: 5px; color: #aaa; }
        .slider-row { display: flex; align-items: center; gap: 10px; }
        input[type=range] { flex-grow: 1; cursor: pointer; }
        .val-display { min-width: 45px; font-family: monospace; color: #00ffcc; font-weight: bold; text-align: right; }
        
        .btn-save { margin-top: 30px; padding: 12px 40px; font-size: 1.1em; background: #0088ff; color: white; border: none; border-radius: 5px; cursor: pointer; transition: 0.2s; }
        .btn-save:hover { background: #00aaff; transform: scale(1.05); }
    </style>
    <script>
        function updateParam(side, param, val) {
            // Update the text display next to the slider
            document.getElementById(side + '_' + param + '_val').innerText = val;
            // Send update to Python
            fetch(`/set_param?side=${side}&param=${param}&val=${val}`);
        }
        
        function saveConfig() {
            fetch('/save').then(r => alert("Configuration Saved to JSON!"));
        }
    </script>
</head>
<body>
    <h1>🌿 SCI_Weeder Remote Dashboard</h1>
    
    <div class="container">
        <div class="stream-box">
            <img src="/video" class="stream-img">
        </div>

        <div class="controls-grid">
            {% for side in ['left', 'right'] %}
            <div class="column">
                <h2>{{ side.upper() }} CAMERA</h2>
                
                {% set sliders = [
                    ('brightness', 'Brightness', 0, 32),
                    ('contrast', 'Contrast', 10, 50),
                    ('exposure', 'Exposure', 0, 1000),
                    ('gain', 'Gain', 0, 255),
                    ('saturation', 'Saturation', 0, 255),
                    ('white_balance', 'White Balance', 2800, 6500),
                    ('sharpness', 'Sharpness', 0, 255)
                ] %}

                {% for key, label, min, max in sliders %}
                <div class="slider-group">
                    <label>{{ label }}</label>
                    <div class="slider-row">
                        <input type="range" min="{{ min }}" max="{{ max }}" value="{{ config[side][key] }}" 
                               oninput="updateParam('{{ side }}', '{{ key }}', this.value)">
                        <span id="{{ side }}_{{ key }}_val" class="val-display">{{ config[side][key] }}</span>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endfor %}
        </div>

        <button class="btn-save" onclick="saveConfig()">💾 SAVE PARAMETERS</button>
    </div>
</body>
</html>
"""

# --- WEB ROUTES ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, config=config)

@app.route('/set_param')
def set_param():
    side = request.args.get('side')
    param = request.args.get('param')
    val = int(request.args.get('val'))
    if side in config:
        config[side][param] = val
        # In a real scenario, this is where we'd call cap.set(...)
    return "OK"

@app.route('/save')
def save():
    # Simulate saving to camera_config.json
    print("Saving to JSON:", config)
    return "OK"

@app.route('/video')
def video():
    def generate():
        while True:
            with lock:
                if output_frame is None: continue
                ret, jpeg = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def ai_processing_loop():
    global output_frame
    # Attempt to open local webcam if available, else use black screen
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Create dummy data if no camera is connected to the PC
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "MOCK FEED (No Hardware)", (130, 240), 1, 1.5, (0, 100, 255), 2)
        
        # Split into two to simulate Dual Cameras
        f_l = frame.copy()
        f_r = frame.copy()

        # Simulate Targeter Markers on both
        cv2.drawMarker(f_l, (320, 240), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
        cv2.drawMarker(f_r, (400, 200), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)

        # Apply mock adjustments to visualize changes (Brightness/Contrast only for demo)
        # alpha = contrast, beta = brightness
        c_l = config['left']['contrast'] / 30.0
        b_l = config['left']['brightness']
        f_l = cv2.convertScaleAbs(f_l, alpha=c_l, beta=b_l)

        c_r = config['right']['contrast'] / 30.0
        b_r = config['right']['brightness']
        f_r = cv2.convertScaleAbs(f_r, alpha=c_r, beta=b_r)

        # Concatenate side-by-side
        combined = cv2.hconcat([f_l, f_r])
        
        with lock:
            output_frame = combined

# Start the background CV logic
threading.Thread(target=ai_processing_loop, daemon=True).start()

if __name__ == "__main__":
    print("🚀 Remote Dashboard Test running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)