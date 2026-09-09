from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional, Tuple

import cv2
import numpy as np


class PixelVisionAdvancedOverlayEngine:
    def __init__(self, target_resolution: Tuple[int, int] = (2560, 1440)):
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

    def compile_display_frame(
        self,
        raw_frame: np.ndarray,
        tracked_entities: list[dict[str, Any]],
        flagged_event: Optional[Any],
        mode: str = "standard",
        flagged_track_ids: Optional[set[int]] = None,
    ) -> np.ndarray:
        is_flagged = flagged_event is not None
        if is_flagged:
            self.log_event(flagged_event.frame_id, flagged_event.cheat_category, flagged_event.confidence_score)

        if mode == "flagged_only":
            display_canvas = self._generate_flagged_view(raw_frame, tracked_entities, flagged_track_ids or set())
        elif mode == "heatmap":
            self.update_heatmap(tracked_entities, is_flagged)
            display_canvas = self.generate_heatmap_overlay(raw_frame)
        else:
            display_canvas = raw_frame.copy()

        return display_canvas

    def _generate_flagged_view(
        self,
        raw_frame: np.ndarray,
        tracked_entities: list[dict[str, Any]],
        flagged_track_ids: set[int],
    ) -> np.ndarray:
        display_canvas = raw_frame.copy()
        if not flagged_track_ids:
            self._draw_chip(display_canvas, "No flags in this session")
            return display_canvas

        for entity in tracked_entities:
            if entity.get("track_id") not in flagged_track_ids:
                continue
            x1, y1, x2, y2 = entity.get("bbox", (0, 0, 0, 0))
            cv2.rectangle(display_canvas, (x1, y1), (x2, y2), (92, 59, 255), 2)
        return display_canvas

    def _draw_chip(self, canvas: np.ndarray, text: str) -> None:
        (text_w, text_h), _ = cv2.getTextSize(text, self.font, 0.5, 1)
        pad = 8
        x1, y1 = 16, 16
        x2 = min(canvas.shape[1], x1 + text_w + pad * 2)
        y2 = min(canvas.shape[0], y1 + text_h + pad * 2)
        region = canvas[y1:y2, x1:x2]
        canvas[y1:y2, x1:x2] = cv2.addWeighted(region, 0.4, np.zeros_like(region), 0.6, 0)
        cv2.putText(canvas, text, (x1 + pad, y2 - pad), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def generate_track_mask_debug_view(
        self,
        raw_frame: np.ndarray,
        tracked_entities: list[dict[str, Any]],
        is_flagged: bool,
    ) -> np.ndarray:
        """Hidden debug view -- the original reveal-mask behavior, kept for diagnostics only."""
        display_canvas = np.zeros_like(raw_frame)
        if is_flagged:
            for entity in tracked_entities:
                bbox = entity.get("bbox", (0, 0, 0, 0))
                x1, y1, x2, y2 = bbox
                display_canvas[y1:y2, x1:x2] = raw_frame[y1:y2, x1:x2]
                cv2.rectangle(display_canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
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
