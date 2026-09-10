from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.hud_masker import WarzoneHUDMasker, scale_bbox, scale_point


class ScaleHelpersTest(unittest.TestCase):
    def test_scale_point_doubles(self) -> None:
        self.assertEqual(scale_point((100, 50), (960, 540), (1920, 1080)), (200, 100))

    def test_scale_bbox_roundtrip_identity(self) -> None:
        bbox = (10, 20, 110, 220)
        size = (640, 360)
        self.assertEqual(scale_bbox(bbox, size, size), bbox)

    def test_origin_stays_origin(self) -> None:
        self.assertEqual(scale_point((0, 0), (960, 540), (2560, 1440)), (0, 0))


class HudMaskerCoordinateTest(unittest.TestCase):
    def test_apply_mask_preserves_frame_shape(self) -> None:
        masker = WarzoneHUDMasker(target_resolution=(2560, 1440))
        frame = np.full((540, 960, 3), 180, dtype=np.uint8)
        masked = masker.apply_mask(frame)
        self.assertEqual(masked.shape, (540, 960, 3))

    def test_hud_fractions_scale_with_resolution(self) -> None:
        large = WarzoneHUDMasker(target_resolution=(2560, 1440))
        small = WarzoneHUDMasker(target_resolution=(1280, 720))
        # Top-left HUD fraction is (0, 0)-(0.18, 0.285). Center of that box
        # must count as HUD at every resolution.
        self.assertTrue(large.overlaps_masked_region(100, 100, 200, 200, width=2560, height=1440))
        self.assertTrue(small.overlaps_masked_region(50, 50, 100, 100, width=1280, height=720))
        # Dead-center of the 16:9 frame is gameplay, not HUD.
        self.assertFalse(large.overlaps_masked_region(1260, 700, 1300, 740, width=2560, height=1440))
        self.assertFalse(small.overlaps_masked_region(630, 350, 650, 370, width=1280, height=720))

    def test_apply_mask_zeros_top_left_hud(self) -> None:
        masker = WarzoneHUDMasker(target_resolution=(1920, 1080))
        frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        masked = masker.apply_mask(frame)
        # A pixel inside the top-left HUD fraction should be blacked out.
        self.assertEqual(int(masked[40, 40].sum()), 0)
        # Center pixel should survive.
        self.assertGreater(int(masked[540, 960].sum()), 0)


class PipelineSpaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.modules.setdefault("onnxruntime", MagicMock())

    def _pipeline(self):
        from src.core.anti_cheat_pipeline import AntiCheatPipeline

        return AntiCheatPipeline(
            log_dir="/tmp/pv-test-logs",
            player_detector_model_path="/tmp/does-not-exist.onnx",
            target_resolution=(1920, 1080),
        )

    def test_entities_stay_analysis_and_scale_for_overlay(self) -> None:
        pipeline = self._pipeline()
        entities = [
            {
                "track_id": 7,
                "bbox": (430, 200, 530, 340),
                "center": (480, 270),
                "confidence": 0.9,
                "class_id": 0,
            }
        ]
        pipeline.update_detected_entities(entities, (540, 960, 3), (1080, 1920, 3))
        analysis = pipeline.get_analysis_entities()
        self.assertEqual(analysis[0]["bbox"], (430, 200, 530, 340))
        self.assertEqual(analysis[0]["center"], (480, 270))
        display = pipeline.get_tracked_entities()
        self.assertEqual(display[0]["bbox"], (860, 400, 1060, 680))
        self.assertEqual(display[0]["center"], (960, 540))

    def test_process_frame_does_not_upscale_analysis_frame(self) -> None:
        from src.core.anti_cheat_pipeline import FrameContext

        pipeline = self._pipeline()
        native = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        analysis = np.full((540, 960, 3), 40, dtype=np.uint8)
        # Texture in HUD-energy zones so scene-skip doesn't trip "no_hud".
        analysis[10:80, 10:80] = 200
        analysis[420:520, 10:80] = 180
        analysis[480:530, 420:540] = 160
        native[20:160, 20:160] = 200
        native[840:1040, 20:160] = 180
        native[960:1060, 840:1080] = 160

        ctx = FrameContext(
            frame=native,
            timestamp=1.0,
            frame_id=3,
            source="test",
            is_duplicate=False,
            analysis_frame=analysis,
        )
        pipeline.process_frame(ctx)
        self.assertEqual(pipeline._analysis_size, (960, 540))
        self.assertEqual(pipeline._display_size, (1920, 1080))
        gray = pipeline.crosshair_analyzer.previous_gray
        if gray is not None:
            self.assertEqual(gray.shape, (540, 960))


if __name__ == "__main__":
    unittest.main()
