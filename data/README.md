# PixelVision dataset storage

Gameplay clips, labels, and notes for reviewing and (optionally) training.

## Structure

- `raw/` – original recordings before labeling
- `clean/` – normal gameplay (gitignored)
- `suspicious/` – suspected cheating gameplay (gitignored)
- `labels/` – JSON metadata
- `logs/` – session JSONL/CSV from the review console (gitignored)
- `models/` – local ONNX weights (gitignored; see `models/README.md`)
- `notes/` – human notes

## Manifest format

Use `manifest.csv` to track each clip. One row per file.

Required columns:
- filename
- label
- game
- map
- mode
- resolution
- fps
- source
- notes

## Example values

- label: clean, aimbot, wallhack, esp, speedhack, macro, unknown
- game: Warzone, Valorant, Apex, CS2, Fortnite, etc.
- mode: ranked, casual, scrim, training

## Eval workflow

Synthetic kinematics smoke test (not a substitute for real VODs):

```bash
python tools/make_synthetic_eval.py --out data/eval_synthetic
python tools/import_dataset.py --input data/eval_synthetic/clips --labels data/eval_synthetic/labels.csv --output data --analyze --report data/eval_synthetic/eval_report.json
```

Real VODs: drop clips in `raw/`, fill `labels_template.csv`, then run `import_dataset.py` the same way.

Note: offline import uses `analysis_stride=1`. Live capture still defaults to stride 3; with SceneGate `no_hud` streaks that can starve kinematics history — keep an eye on it when tuning LIVE.
