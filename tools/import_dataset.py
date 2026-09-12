#!/usr/bin/env python3
"""Import labeled clips into PixelVision clean/suspicious folders and optionally evaluate.

Usage:
    python tools/import_dataset.py --input data/eval_synthetic/clips --labels data/eval_synthetic/labels.csv --output data --analyze --report data/eval_synthetic/eval_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.anti_cheat_pipeline import AntiCheatPipeline, FrameContext

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Import labeled dataset into PixelVision")
    ap.add_argument("--input", required=True, help="Folder of video clips")
    ap.add_argument("--labels", required=True, help="CSV with filename,label columns")
    ap.add_argument("--output", default="data", help="Output root (default: data)")
    ap.add_argument("--width", type=int, default=960, help="Analysis width")
    ap.add_argument("--height", type=int, default=540, help="Analysis height")
    ap.add_argument("--analyze", action="store_true", help="Run PixelVision on each clip")
    ap.add_argument("--report", default="", help="Optional JSON eval report path")
    ap.add_argument("--max-frames", type=int, default=0, help="Cap frames per clip (0 = all)")
    ap.add_argument("--copy", action="store_true", default=True, help="Copy clips into clean/suspicious")
    ap.add_argument("--no-copy", action="store_false", dest="copy")
    return ap.parse_args()


def read_labels(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fname = (row.get("filename") or row.get("file") or row.get("clip") or "").strip()
            label = (row.get("label") or row.get("cheat_type") or row.get("category") or "").strip().lower()
            if not fname or not label:
                continue
            if label in {"legit", "normal", "pro"}:
                label = "clean"
            if label in {"cheat", "aimbot", "snap", "sticky", "wall", "trigger"}:
                label = "suspicious"
            rows.append(
                {
                    "filename": fname,
                    "label": label,
                    "cheat_type": (row.get("cheat_type") or "").strip().lower(),
                    "notes": (row.get("notes") or "").strip(),
                }
            )
    return rows


def resolve_clip(input_dir: Path, fname: str) -> Path | None:
    candidate = Path(fname)
    options: list[Path] = []
    if candidate.is_absolute():
        options.append(candidate)
    options.append(input_dir / fname)
    options.append(input_dir / candidate.name)
    for path in options:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved.suffix.lower() in VIDEO_EXTS:
            return resolved
    target = candidate.name.lower()
    for path in input_dir.rglob("*"):
        if path.is_file() and path.name.lower() == target and path.suffix.lower() in VIDEO_EXTS:
            return path.resolve()
    return None


def analyze_clip(
    clip_path: Path,
    pipeline: AntiCheatPipeline,
    width: int,
    height: int,
    max_frames: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {"ok": False, "error": "open_failed", "events": [], "frames": 0}

    pipeline.reset_stream_state()
    pipeline.update_target_resolution(width, height)
    events: list[dict[str, Any]] = []
    frame_id = 0
    t0 = time.perf_counter()

    while True:
        if max_frames and frame_id >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        ctx = FrameContext(
            frame=frame,
            timestamp=frame_id / 60.0,
            frame_id=frame_id,
            source=str(clip_path),
            is_duplicate=False,
            analysis_frame=frame,
        )
        event = pipeline.process_frame(ctx)
        if event is not None:
            events.append(
                {
                    "frame_id": event.frame_id,
                    "cheat_category": event.cheat_category,
                    "confidence": float(event.confidence_score),
                }
            )
        frame_id += 1

    cap.release()
    elapsed = time.perf_counter() - t0
    best = max(events, key=lambda e: e["confidence"]) if events else None
    return {
        "ok": True,
        "frames": frame_id,
        "elapsed_sec": round(elapsed, 3),
        "flagged": bool(events),
        "event_count": len(events),
        "best": best,
        "events": events[:20],
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    labels_path = Path(args.labels)
    output_root = Path(args.output)

    if not input_dir.is_dir():
        print(f"[ERR] Input dir not found: {input_dir}")
        return 1
    if not labels_path.is_file():
        print(f"[ERR] Labels CSV not found: {labels_path}")
        return 1

    label_rows = read_labels(labels_path)
    print(f"Loaded {len(label_rows)} labels from {labels_path}")

    clean_dir = output_root / "clean"
    suspicious_dir = output_root / "suspicious"
    clean_dir.mkdir(parents=True, exist_ok=True)
    suspicious_dir.mkdir(parents=True, exist_ok=True)

    pipeline = AntiCheatPipeline(
        target_resolution=(args.width, args.height),
        source_profile="vod_file",
        analysis_stride=1,
        detection_player_class_ids=[0],
        player_detector_model_path=str(ROOT / "data" / "models" / "yolov8n.onnx"),
    )

    results: list[dict[str, Any]] = []
    tp = fp = tn = fn = 0

    for row in label_rows:
        fname = row["filename"]
        label = row["label"]
        clip_path = resolve_clip(input_dir, fname)
        if clip_path is None:
            print(f"  [WARN] Missing clip: {fname}")
            results.append({**row, "status": "missing"})
            continue

        if args.copy:
            dest_dir = clean_dir if label == "clean" else suspicious_dir
            dest = dest_dir / clip_path.name
            if clip_path.resolve() != dest.resolve():
                shutil.copy2(clip_path, dest)
            print(f"  Copied {clip_path.name} -> {dest_dir.name}/")
        else:
            print(f"  Using {clip_path}")

        analysis: dict[str, Any] = {"flagged": False, "event_count": 0}
        if args.analyze:
            print(f"  Analyzing {clip_path.name}...")
            analysis = analyze_clip(clip_path, pipeline, args.width, args.height, args.max_frames)
            flagged = bool(analysis.get("flagged"))
            expected_pos = label != "clean"
            if expected_pos and flagged:
                tp += 1
            elif expected_pos and not flagged:
                fn += 1
            elif (not expected_pos) and flagged:
                fp += 1
            else:
                tn += 1
            best = analysis.get("best") or {}
            print(
                f"    frames={analysis.get('frames')} flagged={flagged} "
                f"events={analysis.get('event_count')} "
                f"best={best.get('cheat_category', '-')} "
                f"conf={float(best.get('confidence', 0) or 0):.3f}"
            )

        results.append(
            {
                **row,
                "status": "ok",
                "clip_path": str(clip_path),
                "analysis": analysis if args.analyze else None,
            }
        )

    train_csv = output_root / "training.csv"
    with train_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "filename",
            "label",
            "cheat_type",
            "notes",
            "flagged",
            "event_count",
            "best_category",
            "best_confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            analysis = item.get("analysis") or {}
            best = analysis.get("best") or {}
            writer.writerow(
                {
                    "filename": item.get("filename", ""),
                    "label": item.get("label", ""),
                    "cheat_type": item.get("cheat_type", ""),
                    "notes": item.get("notes", ""),
                    "flagged": analysis.get("flagged", ""),
                    "event_count": analysis.get("event_count", ""),
                    "best_category": best.get("cheat_category", ""),
                    "best_confidence": best.get("confidence", ""),
                }
            )

    metrics = None
    if args.analyze:
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        metrics = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
        }
        print("\nEval vs labels:")
        print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
        if precision is not None:
            print(f"  precision={precision:.3f}")
        if recall is not None:
            print(f"  recall={recall:.3f}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "input": str(input_dir),
                    "labels": str(labels_path),
                    "output": str(output_root),
                    "metrics": metrics,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Report: {report_path}")

    print(f"\nDone. training.csv -> {train_csv}")
    print(f"  Clean dir files: {len(list(clean_dir.glob('*')))}")
    print(f"  Suspicious dir files: {len(list(suspicious_dir.glob('*')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
