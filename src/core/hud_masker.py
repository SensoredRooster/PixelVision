from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class WarzoneHUDMasker:
    def __init__(self, target_resolution: Tuple[int, int] = (1920, 1080)):
        self.width, self.height = target_resolution
        self.last_letterbox_top = 0
        self.last_letterbox_bottom = 0
        self.mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
        self._generate_warzone_masks()

    def _detect_letterbox_bars(self, frame: np.ndarray) -> Tuple[int, int]:
        black_rows = np.all(frame == 0, axis=2)

        top_bar = 0
        for row_is_black in black_rows:
            if not bool(np.all(row_is_black)):
                break
            top_bar += 1

        bottom_bar = 0
        for row_is_black in reversed(black_rows):
            if not bool(np.all(row_is_black)):
                break
            bottom_bar += 1

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
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        top_bar, bottom_bar = self._detect_letterbox_bars(frame)
        if top_bar or bottom_bar:
            self.last_letterbox_top = top_bar
            self.last_letterbox_bottom = bottom_bar
            self._generate_warzone_masks(top_bar=top_bar, bottom_bar=bottom_bar)
        else:
            self.last_letterbox_top = 0
            self.last_letterbox_bottom = 0
            self._generate_warzone_masks()

        masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask)
        return masked_frame


if __name__ == "__main__":
    masker = WarzoneHUDMasker(target_resolution=(1920, 1080))
    dummy_gameplay = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    output = masker.apply_mask(dummy_gameplay)

    print("[SYSTEM] Warzone HUD mask array matrix built cleanly.")
    print(" -> Displaying mask overlay output window. Close window or press 'q' to end.")

    cv2.imshow("PixelVision HUD Mask Layout Validation", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
