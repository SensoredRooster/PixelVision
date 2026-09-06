from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime as ort

from src.core.anomaly_detector import CrosshairKinematicsAnalyzer
from src.core.hud_masker import WarzoneHUDMasker


@dataclass
class FrameContext:
    frame: np.ndarray
    timestamp: float
    frame_id: int
    source: str
    is_duplicate: bool


@dataclass
class CheatEvent:
    timestamp: str
    frame_id: int
    source_type: str
    cheat_category: str
    confidence_score: float
    telemetry_data: dict[str, Any]


class PixelVisionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"session_{int(datetime.now().timestamp())}.jsonl"

    def commit(self, event: CheatEvent) -> None:
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event)) + "\n")


class AntiCheatPipeline:
    def __init__(self, log_dir: str = "logs", model_path: str = "data/models/pixelvision_detector.onnx"):
        self.logger = PixelVisionLogger(log_dir=log_dir)
        self.model_path = model_path
        self.ort_session = None
        self.yolo_model_ready = False
        self.yolo_model_error: Optional[str] = None
        self.hud_masker = WarzoneHUDMasker()
        self.crosshair_analyzer = CrosshairKinematicsAnalyzer()
        self.frame_counter = 0
        self.last_frame_hash: Optional[bytes] = None
        self.last_tracked_entities: list[dict[str, Any]] = []
        self._initialize_onnx_runtime()

    def _initialize_onnx_runtime(self) -> None:
        if not os.path.exists(self.model_path):
            self.yolo_model_ready = False
            self.yolo_model_error = f"Model not found: {self.model_path}"
            print(f"[PIPELINE] [INFO] ONNX model not found at {self.model_path}.")
            return

        try:
            self.ort_session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self.yolo_model_ready = True
            self.yolo_model_error = None
            print(f"[PIPELINE] [SUCCESS] Live ONNX Inference block loaded: {self.model_path}")
        except Exception as exc:
            self.ort_session = None
            self.yolo_model_ready = False
            self.yolo_model_error = str(exc)
            print(f"[PIPELINE] [ERROR] Failed initialization of ONNX Runtime session: {exc}")

    def _check_duplicate(self, current_frame: np.ndarray) -> bool:
        small = cv2.resize(current_frame, (16, 16), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        current_hash = gray.tobytes()

        if self.last_frame_hash == current_hash:
            return True

        self.last_frame_hash = current_hash
        return False

    def process_frame(self, frame_context: FrameContext) -> Optional[CheatEvent]:
        self.frame_counter += 1
        masked_frame = self.hud_masker.apply_mask(frame_context.frame)
        if self._check_duplicate(masked_frame):
            self.last_tracked_entities = []
            return None

        self.last_tracked_entities = []
        metrics = self.crosshair_analyzer.update(masked_frame, frame_context.timestamp)
        if not metrics or not metrics.get("flagged"):
            return None

        event = CheatEvent(
            timestamp=datetime.fromtimestamp(frame_context.timestamp).isoformat(),
            frame_id=frame_context.frame_id,
            source_type=frame_context.source,
            cheat_category=str(metrics.get("event_type", "crosshair_kinematic_anomaly")),
            confidence_score=float(metrics.get("confidence", 0.0)),
            telemetry_data={
                "crosshair_point": list(metrics.get("point", (0, 0))),
                "velocity": float(metrics.get("velocity", 0.0)),
                "straightness": float(metrics.get("straightness", 0.0)),
                "tremor_variance": float(metrics.get("tremor_variance", 0.0)),
                "zero_tremor_streak": int(metrics.get("zero_tremor_streak", 0)),
                "path": metrics.get("path", []),
                "residuals": metrics.get("residuals", []),
            },
        )
        self.logger.commit(event)
        return event
