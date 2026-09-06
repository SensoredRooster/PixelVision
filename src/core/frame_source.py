from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from mss import mss


class FrameSource:
    def __init__(self, settings: dict[str, Any]):
        self.mode = settings.get("capture_mode", "camera").lower()
        self.capture = None
        self.screen = None

        if self.mode == "screen":
            self.screen = mss()
            self.monitor_index = int(settings.get("screen_monitor_index", 1))
            self.region = settings.get("screen_region")
            if self.region is not None:
                self.region = tuple(int(value) for value in self.region)
        else:
            camera_index = int(settings.get("camera_index", 0))
            preferred_capture_backend = cv2.CAP_DSHOW
            self.capture = cv2.VideoCapture(camera_index, preferred_capture_backend)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(settings.get("capture_width", 640)))
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(settings.get("capture_height", 480)))

    def read(self):
        if self.mode == "screen":
            if self.screen is None:
                return False, None

            if self.region is not None:
                shot = self.screen.grab(self.region)
            else:
                monitor = self.screen.monitors[self.monitor_index]
                shot = self.screen.grab(monitor)

            frame = np.array(shot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return True, frame

        if self.capture is None:
            return False, None

        return self.capture.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.screen = None
