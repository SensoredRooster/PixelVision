from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class PixelVisionTelemetryGraph:
    def __init__(self, size: Tuple[int, int] = (260, 140)):
        self.graph_w, self.graph_h = size
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linearity_history: list[float] = []
        self.variance_history: list[float] = []
        self.max_history_points = 45

    def draw_graph_overlay(self, canvas: np.ndarray, linearity: float, variance: float, is_flagged: bool) -> np.ndarray:
        h, w = canvas.shape[:2]
        margin_x = max(20, w - self.graph_w - 20)
        margin_y = max(20, h - self.graph_h - 20)

        self.linearity_history.append(float(linearity))
        self.variance_history.append(float(variance))
        if len(self.linearity_history) > self.max_history_points:
            self.linearity_history.pop(0)
        if len(self.variance_history) > self.max_history_points:
            self.variance_history.pop(0)

        sub_surface = canvas[margin_y : margin_y + self.graph_h, margin_x : margin_x + self.graph_w]
        canvas[margin_y : margin_y + self.graph_h, margin_x : margin_x + self.graph_w] = cv2.addWeighted(
            sub_surface, 0.5, np.zeros_like(sub_surface), 0.5, 0
        )

        grid_color = (0, 255, 0) if not is_flagged else (0, 0, 255)
        cv2.rectangle(canvas, (margin_x, margin_y), (margin_x + self.graph_w, margin_y + self.graph_h), grid_color, 1)
        cv2.putText(
            canvas,
            f"LINEARITY: {linearity:.3f}",
            (margin_x + 10, margin_y + 20),
            self.font,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"TREMOR VAR: {variance:.5f}",
            (margin_x + 10, margin_y + 35),
            self.font,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if len(self.linearity_history) > 1:
            points = []
            step_x = self.graph_w / self.max_history_points

            for idx, var_val in enumerate(self.variance_history):
                x = int(margin_x + (idx * step_x))
                scale_y = min(self.graph_h - 20, int(var_val * 100))
                y = int((margin_y + self.graph_h) - 10 - scale_y)
                points.append((x, y))

            line_color = (0, 0, 255) if is_flagged else (102, 252, 241)
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], line_color, 1, cv2.LINE_AA)

        return canvas
