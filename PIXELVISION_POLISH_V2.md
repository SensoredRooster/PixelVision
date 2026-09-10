# PixelVision polish pass v2

Do not touch workers read loops, anti_cheat_pipeline.process_frame math,
anomaly_detector, object_detector decode, ONNX, or frame_source capture.
No commit.

## 1. LIVE pill
Filled teal, property set from capture-open, not only VOD.
ModePill[mode="live"] = filled #2EE6C7 background, #07090C text, no weak outline.
ModePill[mode="vod"] = filled #E8B86D background, #07090C text.
Same visual weight as the selected STANDARD chip.

## 2. Freeze
Gray 320px absdiff < 1.5 for 90 frames, clear on first motion. One gate. Not kinematics.
Compare consecutive downscaled grayscale frames (max 320px wide), mean absdiff.
Frozen ONLY if absdiff < 1.5 for 90 consecutive frames (~1.5s at 60fps).
Lobby smoke / HUD / facecam motion must clear it immediately.
Do not paint amber status on a one-frame duplicate.
SIGNAL card default is not-frozen; frozen only after the 90-frame hold.

## 3. Facecam
(0.62w, 0.42h)-(0.99w, 0.82h). Ignore bbox centers inside it (center-point test,
not overlap test). Recompute when update_target_resolution fires, but only for
ROIs that were not explicitly configured in settings.

## 4. Paint cap
Canvas 30 Hz, rail 4 Hz, no heatmap compile in STANDARD.
YOLO / detect_and_track stays gated on the analyze checkbox, off by default.

## 5. LIVE chrome
Hide RES/FPS combos in LIVE (show for VOD/manual only).
SIGNAL card: one line, e.g. `ok · str 0.25 · tremor 3.7`.
INCIDENTS 0 grip: 28px, vertical label, hairline only.
Top bar height stays 44; don't wrap.

## 6. Report
py_compile touched files. Report freeze threshold shipped and the ROI tuple. No commit.
