from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np


class PixelVisionLiveOverlay:
    def __init__(self):
        self.box_color = (0, 255, 0)
        self.alert_color = (0, 0, 255)
        self.text_color = (255, 255, 255)
        self.line_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render_overlays(
        self,
        frame: np.ndarray,
        tracked_entities: list[dict[str, Any]],
        flagged_id: Optional[int] = None,
    ) -> np.ndarray:
        overlay_canvas = frame.copy()

        for entity in tracked_entities:
            track_id = entity.get("track_id")
            bbox = entity.get("bbox", (0, 0, 0, 0))
            confidence = float(entity.get("confidence", 0.0))

            x1, y1, x2, y2 = bbox
            active_color = self.alert_color if track_id == flagged_id else self.box_color

            cv2.rectangle(overlay_canvas, (x1, y1), (x2, y2), active_color, self.line_thickness)

            label_text = f"PV-TRACK #{track_id} ({confidence * 100:.0f}%)"
            (width, height), _ = cv2.getTextSize(label_text, self.font, 0.4, 1)
            cv2.rectangle(overlay_canvas, (x1, max(0, y1 - 20)), (x1 + width, y1), active_color, -1)
            cv2.putText(
                overlay_canvas,
                label_text,
                (x1, max(0, y1 - 5)),
                self.font,
                0.4,
                self.text_color,
                1,
                cv2.LINE_AA,
            )

        return overlay_canvas
