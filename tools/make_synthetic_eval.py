#!/usr/bin/env python3
"""Generate synthetic clean vs mechanical-aim clips for PixelVision eval.

These are not a substitute for real Warzone VODs, but they give an immediate
precision/recall smoke test for the kinematics path (same patterns as
tests/test_aim_tracker.py).

Usage:
    python tools/make_synthetic_eval.py --out data/eval_synthetic
    python tools/import_dataset.py --input data/eval_synthetic/clips --labels data/eval_synthetic/labels.csv --output data --analyze
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _textured_field(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(40, 220, size=(height, width), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (31, 31), 0)


def _paint_hud(gray: np.ndarray) -> np.ndarray:
    """Add high-stdev HUD-ish patches so SceneGate does not skip as no_hud."""
    h, w = gray.shape[:2]
    out = gray.copy()
    out[0 : max(1, h // 5), 0 : max(1, w // 6)] = np.clip(
        out[0 : max(1, h // 5), 0 : max(1, w // 6)].astype(np.int16) + 40, 0, 255
    ).astype(np.uint8)
    cv2.rectangle(out, (4, 4), (max(8, w // 7), max(8, h // 6)), 240, 1)
    out[int(h * 0.78) : h, 0 : max(1, w // 5)] ^= 90
    out[int(h * 0.90) : h, int(w * 0.40) : int(w * 0.60)] ^= 70
    return out


def _write_clip(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError(f"no frames for {path}")
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"could not open writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _clean_clip(seed: int, frames: int, height: int, width: int) -> list[np.ndarray]:
    base = _textured_field(height, width, seed)
    rng = np.random.default_rng(seed + 99)
    ox = oy = 0
    out: list[np.ndarray] = []
    cx, cy = width // 2, height // 2
    for _ in range(frames):
        ox += int(rng.integers(-6, 7))
        oy += int(rng.integers(-6, 7))
        shifted = _paint_hud(np.roll(np.roll(base, oy, axis=0), ox, axis=1))
        bgr = cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR)
        cv2.circle(bgr, (cx, cy), 3, (255, 255, 255), -1)
        out.append(bgr)
    return out


def _suspicious_linear_clip(seed: int, frames: int, height: int, width: int, step: int) -> list[np.ndarray]:
    base = _textured_field(height, width, seed)
    out: list[np.ndarray] = []
    cx, cy = width // 2, height // 2
    for i in range(frames):
        shifted = _paint_hud(np.roll(base, shift=step * i, axis=1))
        bgr = cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR)
        cv2.circle(bgr, (cx, cy), 3, (255, 255, 255), -1)
        out.append(bgr)
    return out


def _suspicious_snap_clip(seed: int, frames: int, height: int, width: int) -> list[np.ndarray]:
    base = _textured_field(height, width, seed)
    out: list[np.ndarray] = []
    cx, cy = width // 2, height // 2
    for i in range(frames):
        shift = 0 if i < frames // 3 else (80 if i < (frames // 3) + 2 else 90)
        shifted = _paint_hud(np.roll(base, shift=shift, axis=1))
        bgr = cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR)
        cv2.circle(bgr, (cx, cy), 3, (255, 255, 255), -1)
        out.append(bgr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Make synthetic PixelVision eval clips")
    ap.add_argument("--out", default="data/eval_synthetic")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--clean", type=int, default=4)
    ap.add_argument("--suspicious", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for i in range(args.clean):
        name = f"clean_{i:02d}.mp4"
        frames = _clean_clip(seed=100 + i, frames=args.frames, height=args.height, width=args.width)
        _write_clip(clips / name, frames, args.fps)
        rows.append({"filename": name, "label": "clean", "cheat_type": "", "notes": "jittery synthetic aim"})

    for i in range(args.suspicious):
        name = f"suspicious_linear_{i:02d}.mp4"
        step = 18 + (i % 3) * 4
        frames = _suspicious_linear_clip(
            seed=200 + i, frames=args.frames, height=args.height, width=args.width, step=step
        )
        _write_clip(clips / name, frames, args.fps)
        rows.append(
            {
                "filename": name,
                "label": "suspicious",
                "cheat_type": "snap",
                "notes": f"linear pan step={step}",
            }
        )

    if args.suspicious > 0:
        name = "suspicious_snap_00.mp4"
        frames = _suspicious_snap_clip(seed=300, frames=args.frames, height=args.height, width=args.width)
        _write_clip(clips / name, frames, args.fps)
        rows.append(
            {
                "filename": name,
                "label": "suspicious",
                "cheat_type": "snap",
                "notes": "discrete camera snap",
            }
        )

    labels_path = out / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label", "cheat_type", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clips under {clips}")
    print(f"Labels: {labels_path}")
    print(
        "Next:\n"
        f"  python tools/import_dataset.py --input {clips} --labels {labels_path} "
        f"--output data --analyze --report {out / 'eval_report.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
