#!/usr/bin/env python3
"""Benchmark YOLO .pt vs TensorRT .engine on saved images."""

import argparse
import time
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import AI_CONFIDENCE, DEFAULT_MODEL, DEFAULT_MODEL_ENGINE  # noqa: E402
from vision.detectors.ai_detector import _WeedCVCore, _resolve_weight_path  # noqa: E402


def _collect_images(paths, folder):
    files = []
    if folder:
        root = Path(folder)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(sorted(root.glob(ext)))
    files.extend(Path(p) for p in paths)
    images = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[bench] skipping unreadable image: {path}")
            continue
        images.append((path, img))
    return images


def _resolve_model(spec):
    path = _resolve_weight_path(spec)
    if path is None:
        return None
    return path if path.exists() else None


def _benchmark(label, model_path, images, imgsz, conf, warmup_iters, iters):
    print(f"\n=== {label} ===")
    print(f"model: {model_path}")
    core = _WeedCVCore(
        model_path,
        qpoint_path=None,
        conf=conf,
        verbose=True,
        yolo_backend=label,
    )

    warmup_dt = core.warmup(imgsz=imgsz, iters=warmup_iters)
    print(f"warmup: {warmup_dt:.3f}s ({warmup_iters} iter)")

    total_frames = max(1, int(iters))
    t0 = time.perf_counter()
    total_boxes = 0
    for i in range(total_frames):
        _, img = images[i % len(images)]
        boxes, _ = core._get_filtered_results(img, imgsz=imgsz, conf_override=conf)
        total_boxes += len(boxes)
    dt = time.perf_counter() - t0
    avg_ms = (dt / total_frames) * 1000.0
    print(f"inference: {avg_ms:.2f} ms/frame over {total_frames} iter")
    print(f"avg boxes/frame: {total_boxes / total_frames:.2f}")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="Image paths to benchmark.")
    parser.add_argument("--folder", help="Folder of images/crops to benchmark.")
    parser.add_argument("--pt", default=DEFAULT_MODEL, help="PT model name/path. Default: config DEFAULT_MODEL.")
    parser.add_argument("--engine", default=DEFAULT_MODEL_ENGINE, help="Engine model path/name. Default: config DEFAULT_MODEL_ENGINE.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO imgsz.")
    parser.add_argument("--conf", type=float, default=AI_CONFIDENCE, help="YOLO confidence.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per backend.")
    parser.add_argument("--iters", type=int, default=50, help="Timed inference iterations per backend.")
    args = parser.parse_args()

    images = _collect_images(args.images, args.folder)
    if not images:
        parser.error("provide at least one readable image path or --folder")
    print(f"[bench] loaded {len(images)} image(s)")

    results = {}
    pt_path = _resolve_model(args.pt)
    if pt_path is not None:
        results["pt"] = _benchmark("pt", pt_path, images, args.imgsz, args.conf, args.warmup, args.iters)
    else:
        print(f"[bench] PT model not found: {args.pt}")

    engine_path = _resolve_model(args.engine)
    if engine_path is not None:
        results["engine"] = _benchmark("engine", engine_path, images, args.imgsz, args.conf, args.warmup, args.iters)
    else:
        print(f"[bench] engine model not found: {args.engine}")

    if len(results) == 2:
        speedup = results["pt"] / results["engine"] if results["engine"] > 0 else 0.0
        print(f"\nSpeedup: {speedup:.2f}x (pt / engine)")


if __name__ == "__main__":
    main()
