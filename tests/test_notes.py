from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wavehands.domain.notes import build_scale, note_frequency_from_scale_index


class NotesScaleTests(unittest.TestCase):
    def test_do_major_names(self) -> None:
        scale = build_scale("Do", "Mayor", "Sostenidos (#)")
        self.assertEqual(scale.note_names, ("Do", "Re", "Mi", "Fa", "Sol", "La", "Si"))

    def test_re_minor_with_flats(self) -> None:
        scale = build_scale("Re", "Menor", "Bemoles (b)")
        self.assertEqual(scale.note_names, ("Re", "Mi", "Fa", "Sol", "La", "Sib", "Do"))

    def test_a4_frequency_from_scale(self) -> None:
        # Root La in index 0 should map to MIDI 69 => 440Hz with octave_shift 0.
        scale = build_scale("La", "Mayor", "Sostenidos (#)")
        freq = note_frequency_from_scale_index(0, active_scale=scale, octave_shift=0)
        self.assertAlmostEqual(freq, 440.0, places=6)


if __name__ == "__main__":
    unittest.main()

