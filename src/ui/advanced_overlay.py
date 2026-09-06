from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional, Tuple

import cv2
import numpy as np


class PixelVisionAdvancedOverlayEngine:
    def __init__(self, target_resolution: Tuple[int, int] = (1920, 1080)):
        self.width, self.height = target_resolution
        self.event_history = deque(maxlen=6)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.heatmap_accumulator = np.zeros((self.height, self.width), dtype=np.float32)

    def log_event(self, frame_id: int, category: str, confidence: float) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.event_history.appendleft(
            {
                "frame": frame_id,
                "type": category.upper(),
                "conf": f"{confidence * 100:.0f}%",
                "time": timestamp,
            }
        )

    def update_heatmap(self, tracked_entities: list[dict[str, Any]], flagged: bool) -> None:
        self.heatmap_accumulator = cv2.multiply(self.heatmap_accumulator, 0.98)

        for entity in tracked_entities:
            bbox = entity.get("bbox", (0, 0, 0, 0))
            center = entity.get("center", (0, 0))
            x1, y1, x2, y2 = bbox
            center_x, center_y = center

            intensity = 25.0 if flagged else 5.0
            radius = int((x2 - x1) / 3) if (x2 - x1) > 0 else 15
            cv2.circle(self.heatmap_accumulator, (center_x, center_y), radius, intensity, -1)

    def generate_heatmap_overlay(self, base_frame: np.ndarray) -> np.ndarray:
        normalized_map = np.clip(self.heatmap_accumulator, 0, 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
        if color_heatmap.shape[:2] != base_frame.shape[:2]:
            color_heatmap = cv2.resize(
                color_heatmap,
                (base_frame.shape[1], base_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        blended_frame = cv2.addWeighted(base_frame, 0.75, color_heatmap, 0.25, 0)
        return blended_frame

    def render_scoreboard_panel(self, canvas: np.ndarray) -> None:
        panel_w, panel_h = 360, 240
        margin = 20

        sub_surface = canvas[margin : margin + panel_h, margin : margin + panel_w]
        black_bg = np.zeros_like(sub_surface)
        canvas[margin : margin + panel_h, margin : margin + panel_w] = cv2.addWeighted(
            sub_surface, 0.4, black_bg, 0.6, 0
        )

        cv2.rectangle(canvas, (margin, margin), (margin + panel_w, margin + panel_h), (0, 255, 0), 1)
        cv2.putText(
            canvas,
            "PV EVENT HISTORY LOG",
            (margin + 15, margin + 25),
            self.font,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.line(canvas, (margin + 10, margin + 35), (margin + panel_w - 10, margin + 35), (0, 255, 0), 1)

        start_y = margin + 60
        for idx, event in enumerate(self.event_history):
            event_str = f"[{event['time']}] FR:{event['frame']} | {event['type']} ({event['conf']})"
            text_color = (0, 0, 255) if "SNAP" in event["type"] else (255, 255, 255)
            cv2.putText(
                canvas,
                event_str,
                (margin + 15, start_y + (idx * 25)),
                self.font,
                0.4,
                text_color,
                1,
                cv2.LINE_AA,
            )

    def compile_display_frame(
        self,
        raw_frame: np.ndarray,
        tracked_entities: list[dict[str, Any]],
        flagged_event: Optional[Any],
        mode: str = "standard",
    ) -> np.ndarray:
        is_flagged = flagged_event is not None
        if is_flagged:
            self.log_event(flagged_event.frame_id, flagged_event.cheat_category, flagged_event.confidence_score)

        self.update_heatmap(tracked_entities, is_flagged)

        if mode == "flagged_only":
            display_canvas = np.zeros_like(raw_frame)
            if is_flagged:
                for entity in tracked_entities:
                    bbox = entity.get("bbox", (0, 0, 0, 0))
                    x1, y1, x2, y2 = bbox
                    display_canvas[y1:y2, x1:x2] = raw_frame[y1:y2, x1:x2]
                    cv2.rectangle(display_canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif mode == "heatmap":
            display_canvas = self.generate_heatmap_overlay(raw_frame)
        else:
            display_canvas = raw_frame.copy()

        self.render_scoreboard_panel(display_canvas)
        return display_canvas


if __name__ == "__main__":
    engine = PixelVisionAdvancedOverlayEngine()
    dummy_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 40
    tracked_entities = [
        {"bbox": (100, 200, 300, 500), "center": (200, 350), "track_id": 1, "confidence": 0.9},
        {"bbox": (500, 400, 700, 800), "center": (600, 600), "track_id": 2, "confidence": 0.7},
    ]
    output = engine.compile_display_frame(dummy_frame, tracked_entities, None, mode="standard")
    print("[SYSTEM] Advanced Overlay Engine structures compiled cleanly.")
    print(output.shape)
