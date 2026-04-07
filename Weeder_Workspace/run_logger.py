import os
import sys
import time
import json
import cv2
import threading
from datetime import datetime
from pathlib import Path

class DualLogger:
    """Hijacks terminal output and routes it to both the console and a text file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class AsyncVideoRecorder:
    """Runs a dedicated background thread to read from a separate webcam and encode video."""
    def __init__(self, filepath, camera_index=4, fps=15.0):
        self.filepath = str(filepath)
        self.camera_index = camera_index
        self.fps = fps
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._write_loop, daemon=False) 
        self.thread.start()

    def _write_loop(self):
        # Open the dedicated camera
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        
        # 1. SET CODEC STRICTLY FIRST (crucial for Linux USB bandwidth)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        # 2. SET RESOLUTION
        # Let's test at 640x480 first. If this works, change it to 1920x1080 later.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 3. WARMUP DELAY
        time.sleep(1.0)

        if not cap.isOpened():
            print(f"\n[VIDEO ERROR] Could not open camera {self.camera_index} for background recording.")
            self.running = False
            return

        # 4. ROBUST FIRST-FRAME GRAB
        ret = False
        frame = None
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                print(f"[VIDEO] Successfully grabbed stream from camera {self.camera_index}!")
                break
            print(f"[VIDEO] Waiting for camera {self.camera_index} buffer (attempt {i+1}/10)...")
            time.sleep(0.2)

        if not ret:
            print(f"\n[VIDEO ERROR] Failed to grab frame from camera {self.camera_index} after multiple attempts.")
            cap.release()
            self.running = False
            return

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (w, h))
        
        writer.write(frame)

        # Main recording loop
        while self.running:
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
            else:
                time.sleep(0.01)

        writer.release()
        cap.release()

    def stop(self):
        print("\n[VIDEO] Finalizing video file... please wait.")
        self.running = False
        if self.thread:
            self.thread.join()
        print("[VIDEO] Save complete.")

class RunSession:
    """Manages the timestamped folder, stats, and coordinates the loggers."""
    def __init__(self, base_folder="run_data"):
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_folder) / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.image_dir = self.run_dir / "survey_images"
        self.image_dir.mkdir(exist_ok=True)

        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr
        sys.stdout = DualLogger(self.run_dir / "terminal_output.log")
        sys.stderr = DualLogger(self.run_dir / "error_output.log")

        vid_path = self.run_dir / "run_video.avi"
        # Initializes the standalone recorder on Camera 4
        self.recorder = AsyncVideoRecorder(vid_path, camera_index=4)
        
        self.plant_stats = []
        self.current_plant_start = None
        print(f"=== RUN SESSION INITIALIZED: {self.timestamp} ===")

    def start_recording(self):
        print("[VIDEO] Starting background recording thread on Camera 4...")
        self.recorder.start()

    def save_survey_images(self, frameL, frameR, prefix="coarse_survey"):
        cv2.imwrite(str(self.image_dir / f"{prefix}_left.jpg"), frameL)
        cv2.imwrite(str(self.image_dir / f"{prefix}_right.jpg"), frameR)

    def start_plant_timer(self, plant_id):
        self.current_plant_start = time.time()

    def end_plant_timer(self, plant_id, status="Success", final_err_x=0.0, final_err_y=0.0):
        if self.current_plant_start:
            duration = time.time() - self.current_plant_start
            self.plant_stats.append({
                "plant_id": plant_id,
                "duration_sec": round(duration, 3),
                "status": status,
                "final_err_px": {"x": round(final_err_x, 2), "y": round(final_err_y, 2)}
            })
            self.current_plant_start = None

    def end_session(self):
        print("\n[INFO] Saving stats and closing video...")
        total_time = time.time() - self.start_time
        stats = {
            "total_runtime_sec": round(total_time, 3),
            "total_plants": len(self.plant_stats),
            "plant_details": self.plant_stats
        }
        with open(self.run_dir / "runtime_stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        self.recorder.stop()
        sys.stdout = self.stdout_backup
        sys.stderr = self.stderr_backup