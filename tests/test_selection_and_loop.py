from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wavehands.application.loop_station import LoopStationService
from wavehands.application.selection_service import HoverSelectionService
from wavehands.config import SelectionConfig


class SelectionServiceTests(unittest.TestCase):
    def test_hover_selects_after_threshold(self) -> None:
        service = HoverSelectionService(
            SelectionConfig(
                hover_seconds=0.1,
                cooldown_seconds=0.0,
                stable_frames=1,
            )
        )
        t0 = 10.0
        r1 = service.update(candidate_index=2, now=t0)
        self.assertFalse(r1.just_selected)

        r2 = service.update(candidate_index=2, now=t0 + 0.11)
        self.assertTrue(r2.just_selected)
        self.assertEqual(r2.selected_index, 2)


class LoopStationTests(unittest.TestCase):
    def test_record_then_playback_emits_due_event(self) -> None:
        loop = LoopStationService(min_loop_seconds=0.2)
        t0 = 100.0
        mode = loop.cycle(t0)  # idle -> recording
        self.assertEqual(mode, "recording")

        loop.record_note_event(
            frequency_hz=440.0,
            duration_seconds=0.25,
            velocity=0.7,
            now=t0 + 0.05,
        )
        mode = loop.cycle(t0 + 0.30)  # recording -> playing
        self.assertEqual(mode, "playing")
        self.assertEqual(len(loop.state.layers), 1)
        self.assertEqual(len(loop.state.layers[0].events), 1)

        # First poll after start initializes timing window and emits nothing.
        due0 = loop.poll_due_events(t0 + 0.31)
        self.assertEqual(due0, [])

        # Move forward enough to pass the event offset (0.05s) and expect emission.
        due1 = loop.poll_due_events(t0 + 0.37)
        self.assertEqual(len(due1), 1)
        self.assertAlmostEqual(due1[0].frequency_hz, 440.0, places=6)


if __name__ == "__main__":
    unittest.main()

