#!/usr/bin/env python3
"""Export a local YOLOv8/v11 ONNX player detector for PixelVision.

Weights stay gitignored under data/models/.

Usage:
    python tools/export_player_model.py
    python tools/export_player_model.py --model yolov8n.pt --out data/models/yolov8n.onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Export YOLO ONNX for PixelVision")
    ap.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name or path")
    ap.add_argument("--out", default="data/models/yolov8n.onnx", help="Destination ONNX path")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERR] ultralytics not installed. pip install ultralytics")
        return 1

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    model = YOLO(args.model)
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))
    if not exported.is_file():
        print(f"[ERR] export did not produce a file: {exported}")
        return 1

    if exported.resolve() != out.resolve():
        shutil.move(str(exported), str(out))

    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    print("PixelVision settings default: player_detector_model_path = data/models/yolov8n.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
