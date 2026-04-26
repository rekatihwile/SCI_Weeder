#!/usr/bin/env python3
"""Export the configured YOLO .pt model to a TensorRT .engine file."""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import CV_WEIGHTS_DIR, DEFAULT_MODEL_PT, MODEL_MAP, YOLO_WARMUP_IMGSZ  # noqa: E402


def _resolve_weight_path(model_name_or_path):
    filename = MODEL_MAP.get(model_name_or_path, model_name_or_path)
    path = Path(filename)
    if not path.is_absolute():
        path = CV_WEIGHTS_DIR / path
    return path


def _check_runtime():
    print(f"[export] python: {sys.executable}")

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is not importable: {exc}") from exc

    print(f"[export] torch: {getattr(torch, '__version__', '?')}")
    print(f"[export] cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This Python environment has CPU-only PyTorch. TensorRT export needs the Jetson/CUDA PyTorch env."
        )
    print(f"[export] cuda device: {torch.cuda.get_device_name(0)}")

    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError(f"TensorRT is not importable in this Python environment: {exc}") from exc
    print(f"[export] tensorrt: {getattr(trt, '__version__', '?')}")

    try:
        import ultralytics
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Ultralytics is not importable in this Python environment: {exc}") from exc
    print(f"[export] ultralytics: {getattr(ultralytics, '__version__', '?')}")

    return YOLO


def _default_output_for(pt_path):
    return pt_path.with_suffix(".engine")


def export_engine(args):
    pt_path = _resolve_weight_path(args.model)
    if not pt_path.exists():
        raise FileNotFoundError(f"PT model not found: {pt_path}")

    out_path = Path(args.output).resolve() if args.output else _default_output_for(pt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    YOLO = _check_runtime()

    export_kwargs = {
        "format": "engine",
        "imgsz": args.imgsz,
        "half": not args.fp32,
        "device": args.device,
    }
    if args.workspace is not None:
        export_kwargs["workspace"] = args.workspace
    if args.batch is not None:
        export_kwargs["batch"] = args.batch

    print(f"[export] source: {pt_path}")
    print(f"[export] target: {out_path}")
    print(f"[export] args: {export_kwargs}")
    print("[export] Building TensorRT engine. First export can take a few minutes...")

    exported = YOLO(str(pt_path)).export(**export_kwargs)
    exported_path = Path(exported)
    if not exported_path.is_absolute():
        exported_path = Path.cwd() / exported_path

    if exported_path.resolve() != out_path.resolve():
        if out_path.exists():
            out_path.unlink()
        shutil.move(str(exported_path), str(out_path))

    print(f"[export] wrote: {out_path}")
    print("\nPut this in config.py after export succeeds:")
    print(f'DEFAULT_MODEL_ENGINE = "{out_path.name}"')
    print("USE_TENSORRT_ENGINE = True")
    print('YOLO_BACKEND = "auto"')
    print("\nThen benchmark it with:")
    print(
        "python dev_tools/benchmark_yolo_backend.py "
        f"--engine {out_path.name} --folder trial_recordings/trial_001_20260425_154837/left"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PT, help="YOLO .pt model name/path. Default: config DEFAULT_MODEL_PT.")
    parser.add_argument("--output", help="Output .engine path. Default: same folder/name as the .pt.")
    parser.add_argument("--imgsz", type=int, default=YOLO_WARMUP_IMGSZ, help="Static TensorRT image size.")
    parser.add_argument("--device", default=0, help="CUDA device for export, usually 0 on the Jetson.")
    parser.add_argument("--fp32", action="store_true", help="Export FP32 instead of FP16.")
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace size in GiB, if supported by this Ultralytics version.")
    parser.add_argument("--batch", type=int, default=None, help="Static batch size, if supported by this Ultralytics version.")
    args = parser.parse_args()
    export_engine(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[export] ERROR: {exc}", file=sys.stderr)
        print(
            "\nYou need one Python environment that has all of these at once:\n"
            "  - CUDA-enabled torch\n"
            "  - tensorrt\n"
            "  - ultralytics\n"
            "  - torchvision\n\n"
            "In this repo today, system python has CUDA/TensorRT but misses Ultralytics/TorchVision, "
            "while .venv has Ultralytics/TorchVision but CPU-only torch.",
            file=sys.stderr,
        )
        raise SystemExit(1)
