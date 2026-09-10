# PixelVision

Windows review console for FPS gameplay. It watches a capture card, webcam,
monitor, or mounted VOD and flags aim that looks mechanical: perfectly straight
camera pans and high-speed locks with no tremor.

This is a **spectator / VOD tool**. It does not inject into a game.

## What it does

- Live capture from DirectShow devices (webcam vs capture card, auto-labeled)
- ffmpeg(dshow) mode calibration so HDMI cards don't freeze from a too-fast mode
- VOD playback with seek-to-incident
- Source profiles: HDMI game, stream window (ignores chat / facecam / chrome), VOD file
- Aim scoring via phase correlation under the reticle (not a moving-sprite tracker)
- Optional YOLO player boxes to corroborate a flag (off in LIVE unless you opt in)
- Local JSONL event logs and optional clean-baseline / flagged-clip export

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

ffmpeg must be on `PATH` for capture-card input.

### Player detector (optional)

YOLO is off for live desktop capture by default. To enable it on VOD, or on live
with **Analyze this display anyway**:

1. Export a YOLOv8/v11 ONNX file (COCO `person` = class 0 works as a starting point;
   a game-tuned model is better):

   ```bash
   pip install ultralytics
   yolo export model=yolov8n.pt format=onnx imgsz=640
   ```

2. Copy `yolov8n.onnx` to `data/models/yolov8n.onnx`.

Weights are not stored in git.

## Capture

Edit `config/settings.json`:

| Key | Meaning |
|---|---|
| `capture_mode` | `camera` or `screen` |
| `source_profile` | `hdmi_game`, `stream_window`, or `vod_file` |
| `capture_width` / `capture_height` / `capture_fps` | Requested mode; AUTO in the UI lets the card calibrate |
| `player_detector_model_path` | ONNX detector, default `data/models/yolov8n.onnx` |
| `facecam_roi` | Empty = default bottom-right facecam ignore box |

HDMI gameplay: `source_profile: "hdmi_game"`. Twitch/YouTube window: `stream_window`
so chat, title bar, and facecam are ignored.

If a mounted clip plays at the wrong speed, set `playback_fps` to the real FPS
(`0` = auto).

## Layout

- `main.py` — entry
- `src/app.py` — Qt bootstrap
- `src/core/frame_source.py` — camera / screen / ffmpeg capture
- `src/core/anti_cheat_pipeline.py` — scene gates, HUD mask, corroboration
- `src/core/anomaly_detector.py` — reticle + phase-correlation aim scoring
- `src/core/object_detector.py` — optional YOLO + IOU tracker
- `src/ui/main_window.py` — review console
- `config/settings.json` — defaults

## Tests

```bash
python -m unittest discover -s tests -v
```

## Training (optional)

`src/core/train_workflow.py` fits a small classifier on clip-level aim features
(velocity, straightness, tremor, snap size) from `data/clean/` vs
`data/suspicious/`. It does **not** train on the rule engine's own flags.
Torch is only required for this path.
