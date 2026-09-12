import unittest

import numpy as np

from src.core.scene_gate import HELD, LIVE, SceneGate


def _frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


class SceneGateTests(unittest.TestCase):
    def test_short_black_burst_stays_live(self) -> None:
        gate = SceneGate()
        live = _frame(40)
        black = _frame(0)
        states = []
        for _ in range(30):
            states.append(gate.update("live", live))
        for _ in range(5):
            states.append(gate.update("black", black))
        for _ in range(30):
            states.append(gate.update("live", live))
        self.assertTrue(all(s == LIVE for s in states))
        self.assertIs(gate.display_frame(black), black)

    def test_twenty_five_black_holds_on_frame_fifty(self) -> None:
        gate = SceneGate()
        live = _frame(40)
        black = _frame(0)
        for _ in range(30):
            gate.update("live", live)
        self.assertEqual(gate.last_live_frame_index, 30)
        published = None
        for i in range(25):
            published = gate.update("black", black)
            if published == HELD:
                self.assertEqual(30 + i + 1, 50)
                break
        self.assertEqual(published, HELD)
        self.assertEqual(gate.last_live_frame_index, 30)
        np.testing.assert_array_equal(gate.display_frame(black), live)

    def test_held_reason_can_change_without_resetting_counter(self) -> None:
        gate = SceneGate()
        for _ in range(20):
            gate.update("no_hud", _frame(10))
        self.assertEqual(gate.published, HELD)
        self.assertEqual(gate.reason, "no_hud")
        for _ in range(20):
            gate.update("black", _frame(0))
        self.assertEqual(gate.published, HELD)
        self.assertEqual(gate.reason, "black")

    def test_five_live_frames_do_not_reenter(self) -> None:
        gate = SceneGate()
        for _ in range(25):
            gate.update("black", _frame(0))
        self.assertEqual(gate.published, HELD)
        for _ in range(5):
            gate.update("live", _frame(40))
        self.assertEqual(gate.published, HELD)
        for _ in range(25):
            gate.update("black", _frame(0))
        self.assertEqual(gate.published, HELD)


if __name__ == "__main__":
    unittest.main()
