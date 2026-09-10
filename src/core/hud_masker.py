from __future__ import annotations

import threading
from typing import Tuple

import cv2
import numpy as np

_HUD_ENERGY_STDEV = 12.0
_HUD_ENERGY_FRAC: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.00, 0.18, 0.28),
    (0.00, 0.74, 0.20, 1.00),
    (0.40, 0.88, 0.60, 1.00),
)


def _content_pixels(width: int, height: int, content_frac: Tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = content_frac
    return (
        max(0, min(width, int(width * x0))),
        max(0, min(height, int(height * y0))),
        max(0, min(width, int(width * x1))),
        max(0, min(height, int(height * y1))),
    )


class WarzoneHUDMasker:
    def __init__(self, target_resolution: Tuple[int, int] = (2560, 1440)):
        self._lock = threading.Lock()
        self.width, self.height = target_resolution
        self.last_letterbox_top = 0
        self.last_letterbox_bottom = 0
        self.mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
        self._generate_warzone_masks()

    def _detect_letterbox_bars(self, frame: np.ndarray) -> Tuple[int, int]:
        row_is_black = np.all(frame == 0, axis=(1, 2))
        non_black_rows = np.flatnonzero(~row_is_black)
        if non_black_rows.size == 0:
            return 0, 0

        top_bar = int(non_black_rows[0])
        bottom_bar = int(row_is_black.shape[0] - 1 - non_black_rows[-1])
        return top_bar, bottom_bar

    def _generate_warzone_masks(self, top_bar: int = 0, bottom_bar: int = 0) -> None:
        self.mask = np.ones((self.height, self.width), dtype=np.uint8) * 255

        cv2.rectangle(self.mask, (0, 0), (460, 410), 0, -1)

        content_bottom = max(0, self.height - bottom_bar)
        bottom_left_top = max(0, content_bottom - 380)
        bottom_right_top = max(0, content_bottom - 360)

        cv2.rectangle(self.mask, (0, bottom_left_top), (500, content_bottom), 0, -1)
        cv2.rectangle(self.mask, (self.width - 520, bottom_right_top), (self.width, content_bottom), 0, -1)
        cv2.rectangle(self.mask, (self.width - 450, 0), (self.width, 240), 0, -1)
        cv2.rectangle(
            self.mask,
            (self.width // 2 - 250, max(0, content_bottom - 140)),
            (self.width // 2 + 250, content_bottom),
            0,
            -1,
        )

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))

            top_bar, bottom_bar = self._detect_letterbox_bars(frame)
            if top_bar != self.last_letterbox_top or bottom_bar != self.last_letterbox_bottom:
                self.last_letterbox_top = top_bar
                self.last_letterbox_bottom = bottom_bar
                self._generate_warzone_masks(top_bar=top_bar, bottom_bar=bottom_bar)

            masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask)
            return masked_frame

    def overlaps_masked_region(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        with self._lock:
            x1c = max(0, min(self.width, x1))
            x2c = max(0, min(self.width, x2))
            y1c = max(0, min(self.height, y1))
            y2c = max(0, min(self.height, y2))
            if x2c <= x1c or y2c <= y1c:
                return False
            region = self.mask[y1c:y2c, x1c:x2c]
            return bool(np.any(region == 0))

    def set_target_resolution(self, width: int, height: int) -> None:
        with self._lock:
            self.width, self.height = width, height
            self.last_letterbox_top = 0
            self.last_letterbox_bottom = 0
            self._generate_warzone_masks()

    def cinematic_letterbox(self, frame: np.ndarray, content_frac: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = _content_pixels(w, h, content_frac)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return False
        ch = crop.shape[0]
        band = max(1, int(ch * 0.10))
        top_mean = float(crop[:band].mean())
        bot_mean = float(crop[-band:].mean())
        mid_mean = float(crop[ch // 3 : (2 * ch) // 3].mean())
        return top_mean < 12.0 and bot_mean < 12.0 and mid_mean > 30.0

    def hud_energy_present(self, frame: np.ndarray, content_frac: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = _content_pixels(w, h, content_frac)
        cw, ch = max(1, x1 - x0), max(1, y1 - y0)
        hits = 0
        for fx0, fy0, fx1, fy1 in _HUD_ENERGY_FRAC:
            rx0 = max(0, min(w, x0 + int(cw * fx0)))
            ry0 = max(0, min(h, y0 + int(ch * fy0)))
            rx1 = max(0, min(w, x0 + int(cw * fx1)))
            ry1 = max(0, min(h, y0 + int(ch * fy1)))
            if rx1 <= rx0 or ry1 <= ry0:
                continue
            roi = gray[ry0:ry1, rx0:rx1]
            if roi.size == 0:
                continue
            if float(roi.std()) >= _HUD_ENERGY_STDEV:
                hits += 1
        return hits >= 2


if __name__ == "__main__":
    masker = WarzoneHUDMasker(target_resolution=(2560, 1440))
    dummy_gameplay = np.ones((1440, 2560, 3), dtype=np.uint8) * 255
    output = masker.apply_mask(dummy_gameplay)

    print("[SYSTEM] Warzone HUD mask array matrix built cleanly.")
    print(" -> Displaying mask overlay output window. Close window or press 'q' to end.")

    cv2.imshow("PixelVision HUD Mask Layout Validation", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
