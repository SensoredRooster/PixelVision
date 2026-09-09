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
from src.core.dataset_exporter import PixelVisionDatasetExporter
from src.core.hud_masker import WarzoneHUDMasker
from src.core.object_detector import PixelVisionObjectDetector


@dataclass
class FrameContext:
    frame: np.ndarray
    timestamp: float
    frame_id: int
    source: str
    is_duplicate: bool
    preview_frame: np.ndarray | None = None
    analysis_frame: np.ndarray | None = None


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
    def __init__(
        self,
        log_dir: str = "logs",
        model_path: str = "data/models/pixelvision_detector.onnx",
        analysis_stride: int = 3,
        dataset_exporter: PixelVisionDatasetExporter | None = None,
        target_resolution: tuple[int, int] = (2560, 1440),
        player_detector_model_path: str = "data/models/yolov8n_gaming.onnx",
        detection_confidence_threshold: float = 0.45,
        detection_nms_threshold: float = 0.45,
        detection_player_class_ids: list[int] | None = None,
        detection_corroboration_margin_px: int = 12,
    ):
        self.logger = PixelVisionLogger(log_dir=log_dir)
        self.model_path = model_path
        self.ort_session = None
        self.yolo_model_ready = False
        self.yolo_model_error: Optional[str] = None
        self.hud_masker = WarzoneHUDMasker(target_resolution=target_resolution)
        self.crosshair_analyzer = CrosshairKinematicsAnalyzer()
        self.analysis_stride = max(1, int(analysis_stride))
        self.frame_counter = 0
        self.last_frame_hash: Optional[bytes] = None
        self.last_tracked_entities: list[dict[str, Any]] = []
        self.last_telemetry_snapshot: dict[str, Any] | None = None
        self.last_event_type: Optional[str] = None
        self.last_mechanical_lock_detected = False
        self.dataset_exporter = dataset_exporter
        self.player_detector = PixelVisionObjectDetector(
            model_path=player_detector_model_path,
            conf_threshold=detection_confidence_threshold,
            nms_threshold=detection_nms_threshold,
            player_class_ids=detection_player_class_ids if detection_player_class_ids is not None else [],
        )
        self.detector_has_result = False
        self.detection_corroboration_margin_px = detection_corroboration_margin_px
        self._initialize_onnx_runtime()

    @property
    def detector_ready(self) -> bool:
        return self.player_detector.ort_session is not None

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

    def should_analyze_frame(self, frame_id: int) -> bool:
        return frame_id % self.analysis_stride == 0

    def update_detected_entities(
        self, entities: list[dict[str, Any]], analysis_frame_shape: tuple[int, ...], display_frame_shape: tuple[int, ...]
    ) -> None:
        if not entities:
            self.last_tracked_entities = []
            self.detector_has_result = True
            return

        analysis_h, analysis_w = analysis_frame_shape[:2]
        display_h, display_w = display_frame_shape[:2]

        scale_x = display_w / analysis_w if analysis_w > 0 else 1.0
        scale_y = display_h / analysis_h if analysis_h > 0 else 1.0

        rescaled_entities: list[dict[str, Any]] = []
        for entity in entities:
            bbox = entity.get("bbox", (0, 0, 0, 0))
            center = entity.get("center", (0, 0))

            x1, y1, x2, y2 = bbox
            cx, cy = center

            x1_rescaled = int(x1 * scale_x)
            y1_rescaled = int(y1 * scale_y)
            x2_rescaled = int(x2 * scale_x)
            y2_rescaled = int(y2 * scale_y)
            cx_rescaled = int(cx * scale_x)
            cy_rescaled = int(cy * scale_y)

            rescaled_entities.append(
                {
                    "track_id": entity.get("track_id", -1),
                    "bbox": (x1_rescaled, y1_rescaled, x2_rescaled, y2_rescaled),
                    "center": (cx_rescaled, cy_rescaled),
                    "confidence": entity.get("confidence", 0.0),
                    "class_id": entity.get("class_id", -1),
                }
            )

        self.last_tracked_entities = rescaled_entities
        self.detector_has_result = True

    def process_frame(self, frame_context: FrameContext) -> Optional[CheatEvent]:
        self.frame_counter += 1
        source_frame = frame_context.analysis_frame if frame_context.analysis_frame is not None else frame_context.frame
        masked_frame = self.hud_masker.apply_mask(source_frame)
        if self._check_duplicate(masked_frame):
            return None

        metrics = self.crosshair_analyzer.update(masked_frame, frame_context.timestamp)
        self.last_telemetry_snapshot = self.crosshair_analyzer.last_metrics
        if not metrics or not metrics.get("flagged"):
            self.last_event_type = None
            self.last_mechanical_lock_detected = False
            return None

        self.last_event_type = str(metrics.get("event_type", "crosshair_kinematic_anomaly"))
        self.last_mechanical_lock_detected = self.last_event_type == "MECHANICAL_LOCK_NO_TREMOR"

        corroboration_suppressed = False
        associated_track_id = None

        if self.detector_ready and self.detector_has_result:
            crosshair_point = metrics.get("point", (0, 0))
            px, py = crosshair_point

            best_match = None
            best_confidence = 0.0

            for entity in self.last_tracked_entities:
                bbox = entity.get("bbox", (0, 0, 0, 0))
                x1, y1, x2, y2 = bbox
                conf = entity.get("confidence", 0.0)

                x1_padded = max(0, x1 - self.detection_corroboration_margin_px)
                y1_padded = max(0, y1 - self.detection_corroboration_margin_px)
                x2_padded = x2 + self.detection_corroboration_margin_px
                y2_padded = y2 + self.detection_corroboration_margin_px

                if x1_padded <= px <= x2_padded and y1_padded <= py <= y2_padded:
                    if conf > best_confidence:
                        best_confidence = conf
                        best_match = entity

            if best_match is None:
                corroboration_suppressed = True
            else:
                associated_track_id = best_match.get("track_id", None)

        if corroboration_suppressed:
            return None

        event = CheatEvent(
            timestamp=datetime.fromtimestamp(frame_context.timestamp).isoformat(),
            frame_id=frame_context.frame_id,
            source_type=frame_context.source,
            cheat_category=self.last_event_type,
            confidence_score=float(metrics.get("confidence", 0.0)),
            telemetry_data={
                "crosshair_point": list(metrics.get("point", (0, 0))),
                "velocity": float(metrics.get("velocity", 0.0)),
                "straightness": float(metrics.get("straightness", 0.0)),
                "tremor_variance": float(metrics.get("tremor_variance", 0.0)),
                "zero_tremor_streak": int(metrics.get("zero_tremor_streak", 0)),
                "path": metrics.get("path", []),
                "residuals": metrics.get("residuals", []),
                "associated_track_id": associated_track_id,
            },
        )
        self.logger.commit(event)
        return event

    def reset_stream_state(self) -> None:
        self.last_frame_hash = None
        self.last_tracked_entities = []
        self.last_telemetry_snapshot = None
        self.last_event_type = None
        self.last_mechanical_lock_detected = False
        self.detector_has_result = False
        self.crosshair_analyzer.reset()

    def should_export_suspicious_clip(self) -> bool:
        return self.last_mechanical_lock_detected and self.last_event_type == "MECHANICAL_LOCK_NO_TREMOR"

