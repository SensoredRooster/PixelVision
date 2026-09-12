from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.game_profiles import list_game_profiles, load_game_profile
from src.core.hud_masker import HUDMasker


class GameProfileTests(unittest.TestCase):
    def test_warzone_and_generic_exist(self) -> None:
        ids = {p.id for p in list_game_profiles()}
        self.assertIn("warzone", ids)
        self.assertIn("generic", ids)

    def test_generic_skips_hud_energy_gate(self) -> None:
        masker = HUDMasker(target_resolution=(640, 360), game_profile="generic")
        blackish = np.zeros((360, 640, 3), dtype=np.uint8)
        self.assertTrue(masker.hud_energy_present(blackish))
        self.assertFalse(masker.require_hud_energy)

    def test_warzone_still_requires_energy(self) -> None:
        profile = load_game_profile("warzone")
        self.assertTrue(profile.require_hud_energy)
        masker = HUDMasker(target_resolution=(640, 360), game_profile=profile)
        flat = np.full((360, 640, 3), 128, dtype=np.uint8)
        self.assertFalse(masker.hud_energy_present(flat))


if __name__ == "__main__":
    unittest.main()
