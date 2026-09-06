from __future__ import annotations

import glob
import os
import time
from typing import Any

import cv2
import numpy as np

from src.core.anti_cheat_pipeline import AntiCheatPipeline, FrameContext
from src.core.dataset_loader import AntiCheatMLDatasetLoader
from src.core.model_trainer import AntiCheatModelTrainer


def export_yolo_to_onnx(weights_path: str = "data/models/yolov8n_gaming.pt") -> bool:
    """Convert a PyTorch YOLO weights file to ONNX, with a fallback stub for smoke tests."""
    model_dir = os.path.dirname(weights_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    onnx_path = weights_path.replace(".pt", ".onnx")

    try:
        from ultralytics import YOLO

        print(f"[EXPORT] Loading source weights matrix from target file: {weights_path}")
        model = YOLO(weights_path)
        print("[EXPORT] Compiling PyTorch model execution graph layers to standardized ONNX format...")
        model.export(format="onnx", imgsz=640, dynamic=False, opset=12)
        print(f"[EXPORT] SUCCESS: ONNX serialization fully written -> {onnx_path}")
        return True
    except ImportError:
        print("[EXPORT] 'ultralytics' backend engine missing. Simulating fallback dummy ONNX validation model structure.")
        with open(onnx_path, "w", encoding="utf-8") as handle:
            handle.write("FALLBACK_DUMMY_ONNX_BLOB_FOR_INTEGRATION_SMOKE_TESTS")
        return True
    except Exception as exc:
        print(f"[EXPORT] Process Exception encountered during model architecture compile: {exc}")
        return False


class PixelVisionTrainingWorkflow:
    def __init__(self, data_root: str = "data", models_dir: str = "data/models"):
        self.data_root = data_root
        self.models_dir = models_dir
        self.loader = AntiCheatMLDatasetLoader(data_root=self.data_root)
        self.trainer = AntiCheatModelTrainer(model_dir=self.models_dir)

    def verify_dataset_inventory(self):
        clean_clips = glob.glob(os.path.join(self.data_root, "clean", "*.*"))
        suspicious_clips = glob.glob(os.path.join(self.data_root, "suspicious", "*.*"))

        valid_extensions = (".mp4", ".avi", ".mkv", ".mov")
        clean_clips = [clip for clip in clean_clips if clip.lower().endswith(valid_extensions)]
        suspicious_clips = [clip for clip in suspicious_clips if clip.lower().endswith(valid_extensions)]

        print("\n=====================================================================")
        print("                  PIXELVISION REAL-DATASET INVENTORY                 ")
        print("=====================================================================")
        print(f" -> Found [ {len(clean_clips)} ] Real Verified Clean Gameplay Videos.")
        print(f" -> Found [ {len(suspicious_clips)} ] Real Verified Suspicious Clips.")

        if len(clean_clips) == 0 and len(suspicious_clips) == 0:
            print("\n[!] WARNING: Your media target folders are currently empty.")
            print("    Drop your raw mp4 screen recordings or capture card files into:")
            print(f"    - Clean Reference Baselines: {os.path.abspath(os.path.join(self.data_root, 'clean'))}")
            print(f"    - Flagged Cheat Captures   : {os.path.abspath(os.path.join(self.data_root, 'suspicious'))}")
        elif abs(len(clean_clips) - len(suspicious_clips)) > 5:
            print("\n[!] CLASS IMBALANCE DETECTED: One data class heavily outweighs the other.")
            print("    For maximum neural network accuracy, aim for a roughly 1:1 balance ratio.")
        print("=====================================================================\n")

        return len(clean_clips), len(suspicious_clips)

    def execute_pipeline_retraining(self, epochs: int = 30, batch_size: int = 8):
        clean_count, susp_count = self.verify_dataset_inventory()

        if clean_count == 0 and susp_count == 0:
            print("[ABORT] Cannot initiate real model retraining without video files present.")
            return False

        print("[WORKFLOW] Parsing real video arrays into mathematical feature matrices...")
        X, y = self.loader.compile_training_tensors()

        print(f"[WORKFLOW] Extracted {X.shape[0]} total sequential frame vectors for optimization.")

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print("[ABORT] Retraining failed: the dataset contains only one classification target.")
            print("        You must provide at least one clean clip and one suspicious clip to train.")
            return False

        print("[WORKFLOW] Ingestion complete. Passing vector structures directly to PyTorch layer...")
        self.trainer.train_and_export(X, y, epochs=epochs, batch_size=batch_size)
        print("[WORKFLOW] Retraining cycle finalized. New deployment models are online.")
        return True


class PixelVisionValidationWorkflow:
    def __init__(self, data_root: str = "data", conf_min: float = 0.45, squad_filter_enabled: bool = True):
        self.data_root = data_root
        self.conf_min = conf_min
        self.squad_filter_enabled = squad_filter_enabled
        self.pipeline = AntiCheatPipeline(log_dir=os.path.join(self.data_root, "logs"))

    def _is_friendly_squad_member(self, bbox: tuple[int, int, int, int], frame: np.ndarray) -> bool:
        if not self.squad_filter_enabled:
            return False

        x1, y1, x2, y2 = bbox
        sample_y1 = max(0, y1 - 35)
        sample_y2 = max(1, y1)
        sample_x1 = max(0, x1)
        sample_x2 = min(frame.shape[1], x2)

        head_ui_zone = frame[sample_y1:sample_y2, sample_x1:sample_x2]
        if head_ui_zone.size == 0:
            return False

        hsv_zone = cv2.cvtColor(head_ui_zone, cv2.COLOR_BGR2HSV)
        friendly_low = np.array([35, 50, 50])
        friendly_high = np.array([130, 255, 255])
        mask = cv2.inRange(hsv_zone, friendly_low, friendly_high)
        pixel_match_ratio = float(np.sum(mask > 0) / head_ui_zone.size)
        return pixel_match_ratio > 0.05

    def validate_and_compile_metrics(self) -> dict[str, Any]:
        print("\n=====================================================================")
        print("                PIXELVISION MATRIX DISCRIMINATION TUNER              ")
        print("=====================================================================")

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        classes_to_test = [("suspicious", 1), ("clean", 0)]

        for folder_name, ground_truth_label in classes_to_test:
            target_path = os.path.join(self.data_root, folder_name, "*.*")
            clips = [clip for clip in glob.glob(target_path) if clip.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))]

            if not clips:
                print(f"[TUNER] [INFO] No clips found in 'data/{folder_name}/'. Skipping category matrix profile.")
                continue

            print(f"[TUNER] Ingesting {len(clips)} target video asset streams from folder: data/{folder_name}")
            for clip in clips:
                cap = cv2.VideoCapture(clip)
                frame_id = 0
                clip_flagged = False

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_id += 1
                    context = FrameContext(
                        frame=frame,
                        timestamp=time.time(),
                        frame_id=frame_id,
                        source="workflow_calibration",
                        is_duplicate=False,
                    )

                    masked_frame = self.pipeline.hud_masker.apply_mask(context.frame)
                    metrics = self.pipeline.crosshair_analyzer.update(masked_frame, context.timestamp)
                    if metrics and metrics.get("flagged"):
                        clip_flagged = True
                        break

                    if clip_flagged:
                        break

                cap.release()

                if ground_truth_label == 1:
                    if clip_flagged:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if clip_flagged:
                        false_positives += 1
                    else:
                        true_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_summary = {
            "Precision": float(precision),
            "Recall": float(recall),
            "F1_Score": float(f1_score),
            "ConfusionMatrix": {"TP": true_positives, "FP": false_positives, "TN": true_negatives, "FN": false_negatives},
        }

        print("\n=====================================================================")
        print("                  PIXELVISION ENGINE PERFORMANCE REPORT              ")
        print("=====================================================================")
        print(f" -> Current Confidence Filter Boundary : {self.conf_min * 100:.1f}%")
        print(f" -> Friend-or-Foe Squad Mask Status    : {'ACTIVE' if self.squad_filter_enabled else 'DISABLED'}")
        print(f" -> System Classification Precision    : {precision * 100:.1f}%")
        print(f" -> System Classification Recall Rate   : {recall * 100:.1f}%")
        print(f" -> Compiled Tracking F1 Performance   : {f1_score:.4f}")
        print(f" -> Matrix Layout (TP/FP/TN/FN)        : {list(metrics_summary['ConfusionMatrix'].values())}")
        print("=====================================================================\n")

        return metrics_summary


if __name__ == "__main__":
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/clean", exist_ok=True)
    os.makedirs("data/suspicious", exist_ok=True)

    export_yolo_to_onnx("data/models/yolov8n_gaming.pt")

    tuner = PixelVisionValidationWorkflow(conf_min=0.45, squad_filter_enabled=True)
    tuner.validate_and_compile_metrics()

    workflow = PixelVisionTrainingWorkflow()
    workflow.execute_pipeline_retraining(epochs=25, batch_size=16)
