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
