# PixelVision

PixelVision is a Windows desktop computer vision anti-cheat starter project built in Python.

## Overview

This project provides a lightweight foundation for:
- webcam capture
- motion and anomaly detection
- event logging
- a clean desktop interface for monitoring activity

The goal is to give you a practical starting point for a security or competitive-play monitoring system without locking you into a rigid architecture.

## Features

- Real-time webcam feed
- Basic motion detection using OpenCV
- Local event logging for suspicious activity
- Configurable thresholds via JSON settings
- Desktop UI built with CustomTkinter

## Project layout

- `main.py` – app entry point
- `src/app.py` – app bootstrap
- `src/core/anomaly_detector.py` – motion/anomaly logic
- `src/core/event_logger.py` – local logging helper
- `src/ui/main_window.py` – desktop UI
- `config/settings.json` – default configuration

## Setup

1. Create and activate a virtual environment.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python main.py
```

If a mounted gameplay clip has the wrong playback speed, add an optional
`playback_fps` value to `config/settings.json` and set it to the clip's real FPS.
Use `0` to keep auto-detection.

## Feeding gameplay into the app

The app can watch either:
- a webcam (`capture_mode: "camera"`), or
- a game screen / monitor (`capture_mode: "screen"`)

When camera mode is active, the input dropdown auto-labels discovered devices as
`Webcam` or `Capture Card` based on the DirectShow device name so your webcam and
capture card stay separate in the selector.

To monitor gameplay, edit `config/settings.json` and set:

```json
{
  "capture_mode": "screen",
  "screen_monitor_index": 1,
  "screen_region": [0, 0, 1920, 1080]
}
```

How this works:
- `screen_monitor_index` selects the display to monitor.
- `screen_region` optionally crops the image to just the gameplay window instead of the whole monitor.
- Each captured frame is then processed by the motion detector and logged as an event if it passes the configured motion threshold.

For a game window only, use `screen_region` with the exact coordinates of the game client. You can usually get those dimensions from Windows display settings or by using a tool that reads window bounds.

## Notes

- Use this as a starting point for a privacy-conscious monitoring workflow.
- Keep logs local and review access policies carefully.
- Expand detection logic with object recognition, face tracking, or screen-based checks as your needs grow.
