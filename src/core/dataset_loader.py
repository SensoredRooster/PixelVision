from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.core.anti_cheat_pipeline import AntiCheatPipeline, FrameContext


class PixelVisionDatasetProcessor:
    def __init__(self, target_resolution: tuple[int, int] = (1920, 1080)):
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

            if frame.shape[1] != self.target_res[0] or frame.shape[0] != self.target_res[1]:
                frame = cv2.resize(frame, self.target_res)

            flagged_event = self.pipeline.process_frame(
                frame_context=FrameContext(
                    frame=frame,
                    timestamp=float(frame_id),
                    frame_id=frame_id,
                    source=f"dataset_{label_type}",
                    is_duplicate=False,
                ),
            )

            current_metrics = {
                "frame_id": frame_id,
                "label": 1 if label_type == "suspicious" else 0,
                "flagged_by_rules": 1 if flagged_event else 0,
                "confidence_score": flagged_event.confidence_score if flagged_event else 0.0,
            }
            clip_telemetry_history.append(current_metrics)

        cap.release()
        return clip_telemetry_history


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
                for entry in metrics:
                    features_matrix.append([float(entry["flagged_by_rules"]), float(entry["confidence_score"])])
                    labels_vector.append(int(entry["label"]))

        if not features_matrix:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)

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
    print(f"Total Structural Frame Profiles Loaded (X Shape): {x.shape}")
    print(f"Total Categorical Ground Labels Compiled (y Shape): {y.shape}")
    if len(y) > 0:
        print(f" -> Total Suspicious Malicious Target Frames Found: {np.sum(y == 1)}")
        print(f" -> Total Clean Validated Reference Frames Found: {np.sum(y == 0)}")
    print("=====================================================================\n")
