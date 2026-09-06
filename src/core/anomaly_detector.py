from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class CrosshairSample:
    point: tuple[int, int]
    velocity: float
    straightness: float
    tremor_variance: float
    zero_tremor_streak: int


class CrosshairKinematicsAnalyzer:
    def __init__(
        self,
        window_size: int = 10,
        roi_ratio: float = 0.18,
        velocity_threshold: float = 100.0,
        straightness_threshold: float = 0.995,
        zero_variance_epsilon: float = 1e-9,
    ) -> None:
        self.window_size = window_size
        self.roi_ratio = roi_ratio
        self.velocity_threshold = velocity_threshold
        self.straightness_threshold = straightness_threshold
        self.zero_variance_epsilon = zero_variance_epsilon
        self.point_history: deque[tuple[int, int]] = deque(maxlen=window_size)
        self.time_history: deque[float] = deque(maxlen=window_size)
        self.previous_gray: Optional[np.ndarray] = None
        self.smoothed_point: Optional[tuple[float, float]] = None
        self.zero_tremor_streak = 0

    def _roi_bounds(self, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        roi_half_w = max(24, int(width * self.roi_ratio * 0.5))
        roi_half_h = max(24, int(height * self.roi_ratio * 0.5))
        center_x = width // 2
        center_y = height // 2
        x1 = max(0, center_x - roi_half_w)
        y1 = max(0, center_y - roi_half_h)
        x2 = min(width, center_x + roi_half_w)
        y2 = min(height, center_y + roi_half_h)
        return x1, y1, x2, y2

    def _estimate_crosshair_point(self, frame_gray: np.ndarray) -> tuple[int, int]:
        height, width = frame_gray.shape[:2]
        center = (width // 2, height // 2)

        x1, y1, x2, y2 = self._roi_bounds(frame_gray.shape)
        roi_mask = np.zeros_like(frame_gray, dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255

        features = cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=24,
            qualityLevel=0.01,
            minDistance=7,
            mask=roi_mask,
            blockSize=7,
        )
        if features is not None and len(features) > 0:
            centroid = np.mean(features.reshape(-1, 2), axis=0)
            return (int(round(float(centroid[0]))), int(round(float(centroid[1]))))

        if self.previous_gray is None:
            return center

        delta = cv2.absdiff(self.previous_gray, frame_gray)
        roi = delta[y1:y2, x1:x2]
        if roi.size == 0:
            return center

        _, motion_mask = cv2.threshold(roi, 20, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        weight_sum = float(np.sum(motion_mask))
        if weight_sum <= 0.0:
            return center

        yy, xx = np.indices(motion_mask.shape, dtype=np.float32)
        centroid_x = float(np.sum(xx * motion_mask) / weight_sum) + x1
        centroid_y = float(np.sum(yy * motion_mask) / weight_sum) + y1
        return (int(round(centroid_x)), int(round(centroid_y)))

    def _line_residuals(self, coords: np.ndarray) -> np.ndarray:
        start = coords[0]
        end = coords[-1]
        path = end - start
        path_length = float(np.linalg.norm(path))
        if path_length <= 0.0:
            return np.zeros(len(coords), dtype=np.float32)

        unit = path / path_length
        offsets = coords - start
        projections = np.dot(offsets, unit)
        projected = np.outer(projections, unit) + start
        residuals = np.linalg.norm(coords - projected, axis=1)
        return residuals.astype(np.float32)

    def update(self, frame: np.ndarray, timestamp: float) -> Optional[dict[str, Any]]:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_point = self._estimate_crosshair_point(frame_gray)
        if self.smoothed_point is None:
            self.smoothed_point = (float(raw_point[0]), float(raw_point[1]))
        else:
            self.smoothed_point = (
                (self.smoothed_point[0] * 0.6) + (float(raw_point[0]) * 0.4),
                (self.smoothed_point[1] * 0.6) + (float(raw_point[1]) * 0.4),
            )

        point = (int(round(self.smoothed_point[0])), int(round(self.smoothed_point[1])))
        self.point_history.append(point)
        self.time_history.append(timestamp)
        self.previous_gray = frame_gray.copy()

        if len(self.point_history) < 3:
            return None

        coords = np.array(self.point_history, dtype=np.float32)
        times = np.array(self.time_history, dtype=np.float32)
        step_vectors = np.diff(coords, axis=0)
        step_distances = np.linalg.norm(step_vectors, axis=1)
        elapsed = float(max(times[-1] - times[0], 1e-4))
        displacement = float(np.linalg.norm(coords[-1] - coords[0]))
        total_path_length = float(np.sum(step_distances))
        straightness = float(displacement / total_path_length) if total_path_length > 0.0 else 1.0

        residuals = self._line_residuals(coords)
        tremor_variance = float(np.var(residuals))
        mean_velocity = float(displacement / elapsed)

        if mean_velocity > self.velocity_threshold and tremor_variance <= self.zero_variance_epsilon:
            self.zero_tremor_streak += 1
        else:
            self.zero_tremor_streak = 0

        event_type: Optional[str] = None
        if mean_velocity > self.velocity_threshold and straightness >= self.straightness_threshold:
            event_type = "UNNATURAL_GEOMETRIC_LINE"

        if self.zero_tremor_streak >= 3:
            event_type = "MECHANICAL_LOCK_NO_TREMOR"

        if event_type is None:
            return {
                "flagged": False,
                "point": point,
                "velocity": mean_velocity,
                "straightness": straightness,
                "tremor_variance": tremor_variance,
                "zero_tremor_streak": self.zero_tremor_streak,
                "path": coords.tolist(),
            }

        confidence = min(
            1.0,
            max(
                straightness,
                mean_velocity / max(self.velocity_threshold, 1e-4),
            ),
        )

        return {
            "flagged": True,
            "event_type": event_type,
            "point": point,
            "velocity": mean_velocity,
            "straightness": straightness,
            "tremor_variance": tremor_variance,
            "zero_tremor_streak": self.zero_tremor_streak,
            "confidence": confidence,
            "path": coords.tolist(),
            "residuals": residuals.tolist(),
        }
