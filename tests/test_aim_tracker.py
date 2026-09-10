from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.anomaly_detector import CrosshairKinematicsAnalyzer


def _bgr_from_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _textured_field(height: int, width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(40, 220, size=(height, width), dtype=np.uint8)
    # Low-frequency blobs so phase correlation has something to lock onto.
    blur = cv2.GaussianBlur(noise, (31, 31), 0)
    return blur


class AimTrackerTest(unittest.TestCase):
    def test_static_scene_is_not_flagged(self) -> None:
        analyzer = CrosshairKinematicsAnalyzer()
        base = _textured_field(540, 960, seed=1)
        flagged = 0
        last = None
        for i in range(20):
            last = analyzer.update(_bgr_from_gray(base), timestamp=float(i))
            if last and last.get("flagged"):
                flagged += 1
        self.assertEqual(flagged, 0)
        self.assertIsNotNone(analyzer.last_metrics)
        self.assertLess(float(analyzer.last_metrics["velocity"]), 2.0)

    def test_linear_camera_pan_flags_geometric_line(self) -> None:
        analyzer = CrosshairKinematicsAnalyzer()
        base = _textured_field(540, 960, seed=2)
        flagged_event = None
        for i in range(16):
            shifted = np.roll(base, shift=20 * i, axis=1)
            result = analyzer.update(_bgr_from_gray(shifted), timestamp=float(i))
            if result and result.get("flagged"):
                flagged_event = result
                break
        self.assertIsNotNone(flagged_event)
        self.assertIn(flagged_event["event_type"], {"UNNATURAL_GEOMETRIC_LINE", "MECHANICAL_LOCK_NO_TREMOR"})
        self.assertGreaterEqual(float(flagged_event["straightness"]), 0.96)

    def test_jittery_walk_has_tremor_and_does_not_lock(self) -> None:
        analyzer = CrosshairKinematicsAnalyzer()
        base = _textured_field(540, 960, seed=3)
        rng = np.random.default_rng(9)
        ox, oy = 0, 0
        lock_flags = 0
        last = None
        for i in range(24):
            ox += int(rng.integers(-6, 7))
            oy += int(rng.integers(-6, 7))
            shifted = np.roll(np.roll(base, oy, axis=0), ox, axis=1)
            last = analyzer.update(_bgr_from_gray(shifted), timestamp=float(i))
            if last and last.get("event_type") == "MECHANICAL_LOCK_NO_TREMOR":
                lock_flags += 1
        self.assertIsNotNone(analyzer.last_metrics)
        self.assertGreater(float(analyzer.last_metrics["tremor_variance"]), 0.45)
        self.assertEqual(lock_flags, 0)

    def test_reticle_refines_to_bright_dot_at_center(self) -> None:
        analyzer = CrosshairKinematicsAnalyzer()
        gray = _textured_field(540, 960, seed=4)
        cx, cy = 480, 270
        cv2.circle(gray, (cx, cy), 3, 255, -1)
        frame = _bgr_from_gray(gray)
        analyzer.update(frame, timestamp=0.0)
        analyzer.update(frame, timestamp=0.05)
        point = analyzer.last_metrics["point"]
        self.assertLessEqual(abs(point[0] - cx), 4)
        self.assertLessEqual(abs(point[1] - cy), 4)

    def test_static_reticle_does_not_pin_scene_tracking(self) -> None:
        analyzer = CrosshairKinematicsAnalyzer()
        base = _textured_field(540, 960, seed=6)
        cx, cy = 480, 270
        for i in range(16):
            shifted = np.roll(base, shift=8 * i, axis=1)
            cv2.circle(shifted, (cx, cy), 4, 255, -1)
            analyzer.update(_bgr_from_gray(shifted), timestamp=float(i))
        self.assertIsNotNone(analyzer.last_metrics)
        self.assertGreater(float(analyzer.last_metrics["velocity"]), 4.0)
        self.assertGreater(float(analyzer.last_metrics["tremor_variance"]), 0.0)

    def test_aim_point_defaults_to_frame_center(self) -> None:
        analyzer = CrosshairKinematicsAnalyzer()
        gray = _textured_field(360, 640, seed=5)
        analyzer.update(_bgr_from_gray(gray), timestamp=0.0)
        point = analyzer.last_metrics["point"]
        self.assertEqual(point, (320, 180))


if __name__ == "__main__":
    unittest.main()
