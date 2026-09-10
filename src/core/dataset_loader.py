from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.core.anti_cheat_pipeline import AntiCheatPipeline, FrameContext

_FEATURE_KEYS = (
    "velocity",
    "straightness",
    "tremor_variance",
    "zero_tremor_streak",
    "flow_response",
    "max_step",
)


class PixelVisionDatasetProcessor:
    def __init__(self, target_resolution: tuple[int, int] = (2560, 1440)):
        self.target_res = target_resolution
        self.pipeline = AntiCheatPipeline()

    def process_clip_to_frames(self, video_path: str, label_type: str) -> list[dict[str, Any]]:
        if not os.path.exists(video_path):
            print(f"[ERROR] Video clip target not found: {video_path}")
            return []

        print(f"[DATASET] Processing clip: {os.path.basename(video_path)} | Ground Truth Class: {label_type.upper()}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video file: {video_path}")
            return []

        frame_id = 0
        clip_telemetry_history: list[dict[str, Any]] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            analysis = cv2.resize(frame, (960, 540))

            self.pipeline.process_frame(
                frame_context=FrameContext(
                    frame=frame,
                    timestamp=float(frame_id),
                    frame_id=frame_id,
                    source=f"dataset_{label_type}",
                    is_duplicate=False,
                    analysis_frame=analysis,
                ),
            )
            metrics = self.pipeline.crosshair_analyzer.last_metrics or {}
            clip_telemetry_history.append(
                {
                    "frame_id": frame_id,
                    "label": 1 if label_type == "suspicious" else 0,
                    **{key: float(metrics.get(key, 0.0) or 0.0) for key in _FEATURE_KEYS},
                }
            )

        cap.release()
        return clip_telemetry_history

    def clip_feature_vector(self, history: list[dict[str, Any]]) -> list[float] | None:
        if not history:
            return None
        velocities = [row["velocity"] for row in history]
        straightness = [row["straightness"] for row in history]
        tremor = [row["tremor_variance"] for row in history]
        streaks = [row["zero_tremor_streak"] for row in history]
        responses = [row["flow_response"] for row in history]
        steps = [row["max_step"] for row in history]
        return [
            float(max(velocities)),
            float(max(straightness)),
            float(min(tremor) if tremor else 0.0),
            float(max(streaks)),
            float(max(responses)),
            float(max(steps)),
        ]


class AntiCheatMLDatasetLoader:
    def __init__(self, data_root: str = "data"):
        self.data_root = data_root
        self.processor = PixelVisionDatasetProcessor()
        self.clean_dir = os.path.join(data_root, "clean")
        self.suspicious_dir = os.path.join(data_root, "suspicious")

    def compile_training_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        features_matrix: list[list[float]] = []
        labels_vector: list[int] = []

        for label_name, directory in [("clean", self.clean_dir), ("suspicious", self.suspicious_dir)]:
            if not os.path.exists(directory):
                continue

            for file in os.listdir(directory):
                if not file.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                    continue

                path = os.path.join(directory, file)
                metrics = self.processor.process_clip_to_frames(path, label_type=label_name)
                vector = self.processor.clip_feature_vector(metrics)
                if vector is None:
                    continue
                features_matrix.append(vector)
                labels_vector.append(1 if label_name == "suspicious" else 0)

        if not features_matrix:
            return np.empty((0, len(_FEATURE_KEYS)), dtype=np.float32), np.empty((0,), dtype=np.int64)

        return np.array(features_matrix, dtype=np.float32), np.array(labels_vector, dtype=np.int64)


if __name__ == "__main__":
    os.makedirs("data/clean", exist_ok=True)
    os.makedirs("data/suspicious", exist_ok=True)

    print("[SYSTEM] Starting PixelVision Dataset Compiler Layer...")
    loader = AntiCheatMLDatasetLoader()
    x, y = loader.compile_training_tensors()

    print("\n=====================================================================")
    print("                      DATASET PROCESSING SUMMARY                     ")
    print("=====================================================================")
    print(f"Total clips loaded (X shape): {x.shape}")
    print(f"Labels (y shape): {y.shape}")
    if len(y) > 0:
        print(f" -> Suspicious clips: {np.sum(y == 1)}")
        print(f" -> Clean clips: {np.sum(y == 0)}")
    print("=====================================================================\n")
