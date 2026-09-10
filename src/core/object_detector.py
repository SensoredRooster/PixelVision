from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


class PixelVisionObjectDetector:
    def __init__(
        self,
        model_path: str = "data/models/yolov8n.onnx",
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        player_class_ids: list[int] | None = None,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.player_class_ids = player_class_ids if player_class_ids is not None else []
        if not self.player_class_ids:
            print(
                "[DETECTOR] [WARN] player_class_ids is empty -- detection is filter-closed "
                "and will match zero classes until configured (e.g. [0] for 'person')."
            )
        self.ort_session = None
        self.active_tracks: dict[int, tuple[int, int, int, int, int]] = {}
        self.track_miss_streak: dict[int, int] = {}
        self.track_counter = 0
        self._model_warning_emitted = False
        self._output_layout: str | None = None
        self._load_onnx_model()

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        candidates = [
            model_path,
            "data/models/yolov8n.onnx",
        ]
        seen: set[str] = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.isfile(path):
                return path
        return model_path

    def _load_onnx_model(self) -> None:
        try:
            self.ort_session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            output_shape = self.ort_session.get_outputs()[0].shape
            if len(output_shape) != 3:
                raise ValueError(
                    f"Expected rank-3 YOLO output, got rank {len(output_shape)} "
                    f"(shape {output_shape}). Wrong model file?"
                )
            print(f"[DETECTOR] [SUCCESS] Target bounding box detector model loaded: {self.model_path}")
        except Exception as exc:
            print(
                f"[DETECTOR] [WARN] Object detection model absent at {self.model_path}. "
                f"Simulating fallback tracker. ({exc})"
            )
            self.ort_session = None

    def _detect_output_layout(self, predictions: np.ndarray) -> str:
        if predictions.ndim != 3 or predictions.shape[0] != 1:
            return "unknown"
        axis1_size, axis2_size = predictions.shape[1], predictions.shape[2]
        if axis1_size > axis2_size:
            return "yolov5"
        else:
            return "yolov8"

    def _decode_predictions(self, predictions: np.ndarray) -> list[tuple[int, int, int, int, float, int]]:
        layout = self._detect_output_layout(predictions)
        detections: list[tuple[int, int, int, int, float, int]] = []

        if layout == "yolov5":
            for pred in predictions[0]:
                if len(pred) < 6:
                    continue
                obj_conf = float(pred[4])
                class_scores = pred[5:]
                class_id = int(np.argmax(class_scores))
                confidence = obj_conf * float(class_scores[class_id])
                if confidence <= self.conf_threshold:
                    continue
                if class_id not in self.player_class_ids:
                    continue

                x_center, y_center = float(pred[0]), float(pred[1])
                box_w, box_h = float(pred[2]), float(pred[3])
                detections.append((x_center, y_center, box_w, box_h, confidence, class_id))

        elif layout == "yolov8":
            pred_t = predictions[0].transpose()
            for pred in pred_t:
                if len(pred) < 5:
                    continue
                x_center, y_center, box_w, box_h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
                class_scores = pred[4:]
                class_id = int(np.argmax(class_scores))
                confidence = float(class_scores[class_id])
                if confidence <= self.conf_threshold:
                    continue
                if class_id not in self.player_class_ids:
                    continue

                detections.append((x_center, y_center, box_w, box_h, confidence, class_id))

        return detections

    def _apply_letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(640 / width, 640 / height)
        new_w = int(width * scale)
        new_h = int(height * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2

        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        return padded, scale, pad_x, pad_y

    def _rescale_boxes(
        self,
        detections: list[tuple[int, int, int, int, float, int]],
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_width: int,
        orig_height: int,
    ) -> list[tuple[int, int, int, int, float, int]]:
        rescaled: list[tuple[int, int, int, int, float, int]] = []
        for x_center, y_center, box_w, box_h, conf, class_id in detections:
            x_center_scaled = (x_center - pad_x) / scale
            y_center_scaled = (y_center - pad_y) / scale
            box_w_scaled = box_w / scale
            box_h_scaled = box_h / scale

            x1 = max(0, int(x_center_scaled - box_w_scaled / 2))
            y1 = max(0, int(y_center_scaled - box_h_scaled / 2))
            x2 = min(orig_width - 1, int(x_center_scaled + box_w_scaled / 2))
            y2 = min(orig_height - 1, int(y_center_scaled + box_h_scaled / 2))

            rescaled.append((x1, y1, x2, y2, conf, class_id))

        return rescaled

    def _apply_nms(
        self, boxes: list[tuple[int, int, int, int, float, int]]
    ) -> list[tuple[int, int, int, int, float, int]]:
        if not boxes:
            return []

        boxes_arr = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], dtype=np.float32)
        confidences = np.array([b[4] for b in boxes], dtype=np.float32)

        keep_indices = cv2.dnn.NMSBoxes(boxes_arr.tolist(), confidences.tolist(), self.conf_threshold, self.nms_threshold)

        if isinstance(keep_indices, np.ndarray):
            keep_indices = keep_indices.flatten().tolist()
        elif not isinstance(keep_indices, list):
            keep_indices = []

        kept = [boxes[i] for i in keep_indices]
        return sorted(kept, key=lambda b: b[4], reverse=True)

    def _calculate_iou(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
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
                print("YOLO MODEL ABSENT - INITIALIZE OBJECT DETECTOR WEIGHTS TO SCAN PLAYERS.")
                self._model_warning_emitted = True
            return []

        orig_height, orig_width = frame.shape[:2]
        try:
            padded_frame, scale, pad_x, pad_y = self._apply_letterbox(frame)
            blob = padded_frame.transpose((2, 0, 1))
            blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0

            inputs = {self.ort_session.get_inputs()[0].name: blob}
            outputs = self.ort_session.run(None, inputs)
            predictions = np.asarray(outputs[0])

            raw_detections = self._decode_predictions(predictions)
            rescaled_detections = self._rescale_boxes(
                raw_detections, scale, pad_x, pad_y, orig_width, orig_height
            )
            fresh_detections = self._apply_nms(rescaled_detections)

        except Exception as exc:
            print(f"[DETECTOR] [WARN] Detection inference failed: {exc}")
            return []

        candidate_tracks = set(self.active_tracks.keys())
        updated_tracks: dict[int, tuple[int, int, int, int, int]] = {}
        tracked_entities: list[dict[str, Any]] = []

        for x1, y1, x2, y2, conf, class_id in fresh_detections:
            matched_id = None
            best_iou = 0.30

            for track_id in sorted(candidate_tracks):
                last_box = self.active_tracks[track_id][:4]
                iou = self._calculate_iou((x1, y1, x2, y2), last_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_id = track_id

            if matched_id is not None:
                candidate_tracks.remove(matched_id)
                self.track_miss_streak[matched_id] = 0
            else:
                self.track_counter += 1
                matched_id = self.track_counter
                self.track_miss_streak[matched_id] = 0

            bbox = (x1, y1, x2, y2)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            updated_tracks[matched_id] = (x1, y1, x2, y2, class_id)
            tracked_entities.append(
                {
                    "track_id": matched_id,
                    "bbox": bbox,
                    "center": (center_x, center_y),
                    "confidence": conf,
                    "class_id": class_id,
                }
            )

        for unmatched_id in candidate_tracks:
            self.track_miss_streak[unmatched_id] = self.track_miss_streak.get(unmatched_id, 0) + 1
            if self.track_miss_streak[unmatched_id] <= 5:
                updated_tracks[unmatched_id] = self.active_tracks[unmatched_id]

        for track_id in list(self.track_miss_streak.keys()):
            if track_id not in updated_tracks:
                del self.track_miss_streak[track_id]

        self.active_tracks = updated_tracks
        return tracked_entities
