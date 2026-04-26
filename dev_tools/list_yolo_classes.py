#!/usr/bin/env python3
"""
List class index → name mappings from a YOLO .pt weight file.

Usage:
    python dev_tools/list_yolo_classes.py                    # uses config default
    python dev_tools/list_yolo_classes.py 26_plastic_nano.pt
    python dev_tools/list_yolo_classes.py /abs/path/to/model.pt
"""
import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CV_WEIGHTS_DIR, DEFAULT_MODEL


def resolve_model_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    # Try as-is relative to cwd
    if p.exists():
        return p.resolve()
    # Try in the weights directory, with or without extension
    candidates = [
        CV_WEIGHTS_DIR / name_or_path,
        CV_WEIGHTS_DIR / (name_or_path + ".pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find model: {name_or_path!r}\nSearched: {[str(c) for c in candidates]}")


def main():
    parser = argparse.ArgumentParser(description="Print YOLO class index → name table.")
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model filename or path (default: config DEFAULT_MODEL)",
    )
    args = parser.parse_args()

    model_arg = args.model or DEFAULT_MODEL
    try:
        model_path = resolve_model_path(model_arg)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {model_path}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path), task="segment")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    names = model.names  # dict {int: str}
    if not names:
        print("[WARN] No class names found in model.")
        return

    print(f"\nClasses in {model_path.name} ({len(names)} total):")
    for idx in sorted(names):
        print(f"  {idx:>3}: {names[idx]}")


if __name__ == "__main__":
    main()
