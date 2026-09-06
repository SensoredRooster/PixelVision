# PixelVision dataset storage

Use this folder to store gameplay clips, labels, and notes for building your anti-cheat model.

## Structure

- `raw/` – original recordings or clips before labeling
- `clean/` – normal gameplay examples
- `suspicious/` – suspected cheating gameplay examples
- `labels/` – exported label files or JSON metadata
- `notes/` – human notes, screenshots, and review logs

## Manifest format

Use the file `manifest.csv` to track each clip. Add one row per file.

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
- game: Valorant, Apex, CS2, Fortnite, etc.
- mode: ranked, casual, scrim, training

Keep clips organized by label and game so it's easier to train and validate on real examples.
