# PixelVision

Windows **review console** for FPS gameplay. It watches a capture card, webcam,
monitor, or mounted VOD and flags aim that looks mechanical: perfectly straight
camera pans and high-speed locks with no tremor.

This is a **spectator / VOD tool**. It does not inject into a game.

The desktop UI is **PySide6** (not CustomTkinter). Entry point is `python main.py`.

## What it does

- Live capture from DirectShow devices (webcam vs capture card, auto-labeled)
- ffmpeg(dshow) calibration so HDMI cards do not freeze on an unsupported mode
- **Latest-frame LIVE path** — slow analysis/UI/baseline drops frames; it does not queue a backlog
- Status bar shows the **negotiated** capture `width×height @ fps`, not the settings.json request
- VOD playback with seek-to-incident
- Source profiles (`SRC` on the top bar): HDMI game, stream window, VOD file
- Scene gate: skip aim scoring on frozen/black frames, cinematic letterbox, or missing HUD energy
- Ignore rects for facecam / stream chrome / chat (center-point test on tracks)
- Aim scoring via **phase correlation** under the reticle (camera snap, not a moving-sprite tracker)
- Optional YOLO player boxes to corroborate a flag (off in LIVE unless you opt in)
- View modes: STANDARD, HEATMAP, FLAGGED
- Local JSONL event logs, clean-baseline recording, and flagged-clip export

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

ffmpeg must be on `PATH` for capture-card input.

```bash
python -m unittest discover -s tests -v
```

### Player detector (optional)

YOLO is **off** for live capture by default (desktop/task-manager frames would
false-positive). It runs on VOD automatically, or on LIVE if you check
**Analyze this display anyway**.

1. Export a YOLOv8/v11 ONNX file (COCO `person` = class 0 is a starting point;
   a game-tuned model is better):

   ```bash
   pip install ultralytics
   yolo export model=yolov8n.pt format=onnx imgsz=640
   ```

2. Copy it to `data/models/yolov8n.onnx` (or set `player_detector_model_path`).

Weights are gitignored. The console still scores aim without them.

## Review console

Top bar (44px): Mount VOD, Rescan, **SRC** profile, STANDARD/HEATMAP/FLAGGED,
RES/FPS (VOD/manual only), LIVE/VOD pill, Analyze checkbox, clean baseline.

- **SOURCE** card — device or VOD name plus negotiated mode
- **SIGNAL** card — `ok · str 0.25 · tremor 3.7` (or `FREEZE`) at 4 Hz
- Canvas paints at 30 Hz from the newest completed frame
- **INCIDENTS** drawer — collapsed 28px grip; expands on the first flag (capped at 500)

LIVE freeze latch: grayscale absdiff on a max-320px sample, frozen only after
**90** consecutive frames below **1.5**. Motion (HUD, smoke, facecam) clears it
immediately. Empty grab for ~1s shows **Waiting for capture device** and clears
the stale picture.

Clean-baseline recording is skip-if-busy and never stalls the grab loop.

## Source profiles

Set in the UI (`SRC`) or `config/settings.json` → `source_profile`.

| Profile | Use | Ignore rects (normalized) |
|---|---|---|
| `hdmi_game` | Capture card / full-frame game | Facecam `(0.62, 0.42, 0.99, 0.82)` unless `facecam_roi` is set |
| `stream_window` | Kick/Twitch/YouTube in a browser | Top `0–0.10`, bottom `0.88–1.0`, right chat `0.80–1.0` (default on), plus facecam |
| `vod_file` | Mounted clip (default at VOD mount is `stream_window`) | Same as stream window |

Tracks whose bbox **center** sits in an ignore rect are dropped. Kinematics
sample the center ROI only; ignore pixels are zeroed first.

If the frame is not live gameplay (frozen, black, letterbox, no minimap/ammo/stance
energy), kinematics and CheatEvents are skipped. YOLO can still run when Analyze is on.

## Capture settings

`config/settings.json`:

| Key | Meaning |
|---|---|
| `capture_mode` | `camera` or `screen` |
| `source_profile` | `hdmi_game`, `stream_window`, or `vod_file` |
| `stream_chat_ignore` | Include the right-chat ignore rect on stream/VOD (default `true`) |
| `capture_width` / `capture_height` / `capture_fps` | Requested mode; AUTO in the UI lets the card calibrate. Status bar uses what the device actually opened. |
| `player_detector_model_path` | ONNX detector, default `data/models/yolov8n.onnx` |
| `facecam_roi` | Empty = default facecam ignore box |
| `playback_fps` | Mounted-clip FPS; `0` = auto |

HDMI gameplay: `hdmi_game`. Browser stream: `stream_window`.

## How aim is scored

FPS reticles sit at screen center. Aimbots snap the **camera**, not a crosshair
sprite. `CrosshairKinematicsAnalyzer` measures scene translation under the
reticle with phase correlation, then scores path straightness and tremor.

HUD masks are fractional and applied at the **incoming frame size**. Analysis
often runs on a downscaled copy; YOLO boxes stay in analysis space for
corroboration; overlays scale them onto the native capture frame.

## Layout

- `main.py` — entry
- `src/app.py` — Qt bootstrap
- `src/core/frame_source.py` — camera / screen / ffmpeg capture + freeze latch
- `src/core/anti_cheat_pipeline.py` — profiles, ignore rects, scene gate, corroboration
- `src/core/anomaly_detector.py` — phase-correlation aim scoring
- `src/core/hud_masker.py` — Warzone HUD mask + HUD-energy / letterbox checks
- `src/core/object_detector.py` — optional YOLO + IOU tracker
- `src/core/dataset_exporter.py` — clean baseline + flagged clips
- `src/ui/main_window.py` — review console
- `src/ui/workers.py` — capture / playback / analysis / detection threads
- `config/settings.json` — defaults
- `tests/` — aim tracker + coordinate-space unit tests

## Training (optional)

`src/core/train_workflow.py` fits a small classifier on clip-level aim features
(velocity, straightness, tremor, snap size) from `data/clean/` vs
`data/suspicious/`. It does **not** train on the rule engine's own flags.
Torch is only required for this path.

Session logs (`logs/`, `data/logs/`) and recordings (`data/clean/`,
`data/suspicious/`) are gitignored.
