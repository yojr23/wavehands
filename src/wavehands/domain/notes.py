import math

NOTE_NAMES = ("Do", "Re", "Mi", "Fa", "Sol", "La", "Si")
CHORD_NAMES = ("Maj", "Min", "Dim", "Aug", "7", "Maj7", "Sus2", "Sus4")
CHORD_INTERVALS = {
    "Maj": (0, 4, 7),
    "Min": (0, 3, 7),
    "Dim": (0, 3, 6),
    "Aug": (0, 4, 8),
    "7": (0, 4, 7, 10),
    "Maj7": (0, 4, 7, 11),
    "Sus2": (0, 2, 7),
    "Sus4": (0, 5, 7),
}
NOTE_TO_MIDI_C4 = {
    "Do": 60,
    "Re": 62,
    "Mi": 64,
    "Fa": 65,
    "Sol": 67,
    "La": 69,
    "Si": 71,
}


def note_name_from_index(index: int) -> str:
    return NOTE_NAMES[index % len(NOTE_NAMES)]


def note_frequency(note_name: str, octave_shift: int = 0, a4_hz: float = 440.0) -> float:
    midi = NOTE_TO_MIDI_C4[note_name] + (octave_shift * 12)
    return a4_hz * math.pow(2.0, (midi - 69) / 12.0)


def note_frequency_from_index(index: int, octave_shift: int = 0, a4_hz: float = 440.0) -> float:
    name = note_name_from_index(index)
    return note_frequency(name, octave_shift=octave_shift, a4_hz=a4_hz)


def chord_name_from_index(index: int) -> str:
    return CHORD_NAMES[index % len(CHORD_NAMES)]


def chord_intervals_from_index(index: int) -> tuple[int, ...]:
    chord_name = chord_name_from_index(index)
    return CHORD_INTERVALS[chord_name]
