from __future__ import annotations

from typing import Optional

import numpy as np


LIVE = "Live"
HELD = "Held"
RAW_LIVE = "live"


class SceneGate:
    N = 20

    def __init__(self, hysteresis_frames: int | None = None) -> None:
        self.N = int(hysteresis_frames) if hysteresis_frames else 20
        self.published = LIVE
        self.reason = RAW_LIVE
        self._streak = 0
        self._streak_kind = "live"
        self.last_live_frame: Optional[np.ndarray] = None
        self.frame_index = 0
        self.last_live_frame_index = 0

    @property
    def state(self) -> str:
        return self.published

    @property
    def published_state(self) -> str:
        return self.published

    @property
    def is_live(self) -> bool:
        return self.published == LIVE

    @property
    def is_held(self) -> bool:
        return self.published == HELD

    @property
    def raw_reason(self) -> str:
        return self.reason

    @property
    def gate_reason(self) -> str:
        return self.reason

    def reset(self) -> None:
        self.published = LIVE
        self.reason = RAW_LIVE
        self._streak = 0
        self._streak_kind = "live"
        self.last_live_frame = None
        self.frame_index = 0
        self.last_live_frame_index = 0

    def update(self, raw: str | None, frame: np.ndarray | None = None) -> str:
        self.frame_index += 1
        is_live = raw in (None, "", RAW_LIVE)
        kind = "live" if is_live else "held"
        if kind != self._streak_kind:
            self._streak_kind = kind
            self._streak = 1
        else:
            self._streak += 1

        if is_live:
            self.reason = RAW_LIVE
            if frame is not None:
                self.last_live_frame = frame
                self.last_live_frame_index = self.frame_index
        else:
            self.reason = str(raw)

        if self.published == LIVE and kind == "held" and self._streak >= self.N:
            self.published = HELD
        elif self.published == HELD and kind == "live" and self._streak >= self.N:
            self.published = LIVE
        return self.published

    def display_frame(self, current: np.ndarray) -> np.ndarray:
        if self.published == HELD and self.last_live_frame is not None:
            return self.last_live_frame
        return current
