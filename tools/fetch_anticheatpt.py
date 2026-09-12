#!/usr/bin/env python3
"""Dataset notes + helpers for PixelVision.

The old "AntiCheatPT" GitHub link in early drafts does not host gameplay video.
The AntiCheatPT paper's public CS2CD data is tick/demo tables (Parquet/JSON),
which PixelVision cannot score — this console needs pixels (capture / VOD).

What to use instead:
  1) Synthetic smoke eval (kinematics only):
       python tools/make_synthetic_eval.py --out data/eval_synthetic
       python tools/import_dataset.py --input data/eval_synthetic/clips \\
           --labels data/eval_synthetic/labels.csv --output data --analyze \\
           --report data/eval_synthetic/eval_report.json
  2) Your own labeled VODs:
       put clips under data/raw/, fill data/labels_template.csv, then:
       python tools/import_dataset.py --input data/raw --labels data/labels_template.csv \\
           --output data --analyze --report data/eval_report.json

Optional research links (not auto-downloaded):
  - CS2CD (tick data): https://huggingface.co/datasets/CS2CD/Context_window_256
  - GAN-Aimbots Doom videos: https://zenodo.org/record/6345323
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="PixelVision dataset helper (no broken AntiCheatPT download)")
    ap.add_argument("--out", default="data/eval_synthetic", help="Unused; kept for old scripts")
    args = ap.parse_args()
    print(__doc__)
    print(f"(--out={args.out} ignored; see commands above)")
    print("\nTip: start with tools/make_synthetic_eval.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
