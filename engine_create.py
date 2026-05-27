from ultralytics import YOLO

# 1. Load your trained PyTorch model
model = YOLO("/home/eli/LaserWeeder_CleanRuntime/params/cv_weights/yolo26n_seg_best_20260512_224747.pt")

# 2. Export to TensorRT with FP16 precision
print("Starting FP16 TensorRT export... This might take a few minutes.")
model.export(
    format="engine",
    imgsz=1280,
    half=True,
    device=0,
    workspace=4,
    dynamic=False,   # Enables dynamic shapes
    batch=8        # Sets the MAX batch size the engine will support
)
print("Export complete!")