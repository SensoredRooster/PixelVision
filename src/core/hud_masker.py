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

# Original 2560x1440 pixel HUD boxes, converted to fractions of the content
# area (between letterbox bars). apply_mask paints these at the incoming
# frame's actual size — it never resizes the frame.
_HUD_MASK_FRAC: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.00, 0.180, 0.285),  # top-left minimap / squad
    (0.00, 0.736, 0.195, 1.000),  # bottom-left weapon
    (0.797, 0.750, 1.000, 1.000),  # bottom-right
    (0.824, 0.000, 1.000, 0.167),  # top-right
    (0.402, 0.903, 0.598, 1.000),  # bottom-center compass / kill ticker
)


def _content_pixels(width: int, height: int, content_frac: Tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = content_frac
    return (
        max(0, min(width, int(width * x0))),
        max(0, min(height, int(height * y0))),
        max(0, min(width, int(width * x1))),
        max(0, min(height, int(height * y1))),
    )


def scale_point(
    point: tuple[float, float] | tuple[int, int],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[int, int]:
    """Map a point from one frame size to another. (0, 0) stays origin-aligned."""
    x, y = point
    fw, fh = from_size
    tw, th = to_size
    if fw <= 0 or fh <= 0:
        return int(round(x)), int(round(y))
    return int(round(x * tw / fw)), int(round(y * th / fh))


def scale_bbox(
    bbox: tuple[int, int, int, int],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    sx1, sy1 = scale_point((x1, y1), from_size, to_size)
    sx2, sy2 = scale_point((x2, y2), from_size, to_size)
    return (sx1, sy1, sx2, sy2)


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
        content_top = max(0, int(top_bar))
        content_bottom = max(content_top + 1, self.height - max(0, int(bottom_bar)))
        content_height = max(1, content_bottom - content_top)
        content_width = max(1, self.width)

        for fx0, fy0, fx1, fy1 in _HUD_MASK_FRAC:
            x1 = int(content_width * fx0)
            y1 = content_top + int(content_height * fy0)
            x2 = int(content_width * fx1)
            y2 = content_top + int(content_height * fy1)
            cv2.rectangle(self.mask, (x1, y1), (x2, y2), 0, -1)

    def _sync_mask_to_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        top_bar, bottom_bar = self._detect_letterbox_bars(frame)
        if (
            width == self.width
            and height == self.height
            and top_bar == self.last_letterbox_top
            and bottom_bar == self.last_letterbox_bottom
            and self.mask is not None
            and self.mask.shape[:2] == (height, width)
        ):
            return
        self.width, self.height = width, height
        self.last_letterbox_top = top_bar
        self.last_letterbox_bottom = bottom_bar
        self._generate_warzone_masks(top_bar=top_bar, bottom_bar=bottom_bar)

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Zero HUD pixels in-place-equivalent; output is the same HxW as input."""
        with self._lock:
            self._sync_mask_to_frame(frame)
            return cv2.bitwise_and(frame, frame, mask=self.mask)

    def overlaps_masked_region(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int | None = None,
        height: int | None = None,
    ) -> bool:
        """True if the bbox *center* sits inside a HUD fraction (same test as facecam)."""
        frame_w = int(width) if width is not None else self.width
        frame_h = int(height) if height is not None else self.height
        if frame_w <= 0 or frame_h <= 0:
            return False
        cx = (float(x1) + float(x2)) * 0.5
        cy = (float(y1) + float(y2)) * 0.5
        nx = cx / frame_w
        ny = cy / frame_h
        for fx0, fy0, fx1, fy1 in _HUD_MASK_FRAC:
            if fx0 <= nx <= fx1 and fy0 <= ny <= fy1:
                return True
        return False

    def set_target_resolution(self, width: int, height: int) -> None:
        with self._lock:
            self.width, self.height = max(1, int(width)), max(1, int(height))
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
