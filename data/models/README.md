# Detector weights

This folder is empty in git on purpose.

## Setup

```bash
pip install ultralytics
python tools/export_player_model.py
```

That writes `yolov8n.onnx` here (gitignored). COCO class `0` (`person`) is the default in `config/settings.json`.
A game-tuned model will beat COCO person boxes for Warzone silhouettes.

The review console runs without weights; kinematics still score aim motion.
YOLO is used only to corroborate a flag with a player box.
