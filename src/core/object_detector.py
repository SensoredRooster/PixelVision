from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


class PixelVisionObjectDetector:
    def __init__(self, model_path: str = "data/models/yolov8n_gaming.onnx"):
        self.model_path = model_path
        self.ort_session = None
        self.conf_threshold = 0.45
        self.active_tracks: dict[int, tuple[int, int, int, int]] = {}
        self.track_counter = 0
        self._model_warning_emitted = False
        self._load_onnx_model()

    def _load_onnx_model(self) -> None:
        try:
            self.ort_session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            print(f"[DETECTOR] [SUCCESS] Target bounding box detector model loaded: {self.model_path}")
        except Exception as exc:
            print(
                f"[DETECTOR] [WARN] Object detection model absent at {self.model_path}. "
                f"Simulating fallback tracker. ({exc})"
            )
            self.ort_session = None

    def _calculate_iou(
        self,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
        return inter_area / float(box_a_area + box_b_area - inter_area)

    def detect_and_track(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.ort_session is None:
            if not self._model_warning_emitted:
                print(
                    "YOLO MODEL ABSENT - INITIALIZE OBJECT DETECTOR WEIGHTS TO SCAN PLAYERS."
                )
                self._model_warning_emitted = True
            return []

        height, width = frame.shape[:2]
        fresh_detections: list[tuple[int, int, int, int, float]] = []

        try:
            blob = cv2.resize(frame, (640, 640))
            blob = blob.transpose((2, 0, 1))
            blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0

            inputs = {self.ort_session.get_inputs()[0].name: blob}
            outputs = self.ort_session.run(None, inputs)
            predictions = np.asarray(outputs[0])

            if predictions.ndim == 3 and predictions.shape[0] > 0:
                for pred in predictions[0]:
                    if len(pred) < 6:
                        continue
                    conf = float(pred[4])
                    if conf <= self.conf_threshold:
                        continue

                    x_center = float(pred[0]) * width / 640.0
                    y_center = float(pred[1]) * height / 640.0
                    box_w = float(pred[2]) * width / 640.0
                    box_h = float(pred[3]) * height / 640.0

                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    fresh_detections.append((x1, y1, x2, y2, conf))
        except Exception as exc:
            print(f"[DETECTOR] [WARN] Detection inference failed: {exc}")
            return []

        updated_tracks: dict[int, tuple[int, int, int, int]] = {}
        tracked_entities: list[dict[str, Any]] = []

        for det in fresh_detections:
            matched_id = None
            best_iou = 0.30
            for track_id, last_box in self.active_tracks.items():
                iou = self._calculate_iou(det[:4], last_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_id = track_id

            if matched_id is None:
                self.track_counter += 1
                matched_id = self.track_counter

            bbox = det[:4]
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            updated_tracks[matched_id] = bbox
            tracked_entities.append(
                {
                    "track_id": matched_id,
                    "bbox": bbox,
                    "center": (center_x, center_y),
                    "confidence": det[4],
                }
            )

        self.active_tracks = updated_tracks
        return tracked_entities
