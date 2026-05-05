import math
from dataclasses import dataclass

NOTE_NAMES = ("Do", "Re", "Mi", "Fa", "Sol", "La", "Si")
ROOT_NOTE_OPTIONS = ("Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si")
ACCIDENTAL_OPTIONS = ("Sostenidos (#)", "Bemoles (b)")
SCALE_NAMES = ("Mayor", "Menor", "Pentatonica Mayor", "Pentatonica Menor", "Blues", "Cromatica")
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
CHROMATIC_SHARP_NAMES = ("Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si")
CHROMATIC_FLAT_NAMES = ("Do", "Reb", "Re", "Mib", "Mi", "Fa", "Solb", "Sol", "Lab", "La", "Sib", "Si")

NOTE_TO_SEMITONE = {
    "Do": 0,
    "Do#": 1,
    "Reb": 1,
    "Re": 2,
    "Re#": 3,
    "Mib": 3,
    "Mi": 4,
    "Fa": 5,
    "Fa#": 6,
    "Solb": 6,
    "Sol": 7,
    "Sol#": 8,
    "Lab": 8,
    "La": 9,
    "La#": 10,
    "Sib": 10,
    "Si": 11,
}

SCALE_INTERVALS = {
    "Mayor": (0, 2, 4, 5, 7, 9, 11),
    "Menor": (0, 2, 3, 5, 7, 8, 10),
    "Pentatonica Mayor": (0, 2, 4, 7, 9),
    "Pentatonica Menor": (0, 3, 5, 7, 10),
    "Blues": (0, 3, 5, 6, 7, 10),
    "Cromatica": tuple(range(12)),
}


@dataclass(frozen=True)
class ActiveScale:
    root_note: str
    scale_name: str
    accidental_mode: str
    note_names: tuple[str, ...]
    semitone_offsets: tuple[int, ...]

    @property
    def root_semitone(self) -> int:
        return NOTE_TO_SEMITONE[self.root_note]


def note_name_from_index(index: int) -> str:
    return NOTE_NAMES[index % len(NOTE_NAMES)]


def note_frequency(note_name: str, octave_shift: int = 0, a4_hz: float = 440.0) -> float:
    semitone = NOTE_TO_SEMITONE[_normalize_note_name(note_name)]
    midi = 60 + semitone + (octave_shift * 12)
    return a4_hz * math.pow(2.0, (midi - 69) / 12.0)


def note_frequency_from_index(index: int, octave_shift: int = 0, a4_hz: float = 440.0) -> float:
    name = note_name_from_index(index)
    return note_frequency(name, octave_shift=octave_shift, a4_hz=a4_hz)


def build_active_scale(root_note: str, scale_name: str, accidental_mode: str) -> ActiveScale:
    root = _normalize_note_name(root_note)
    intervals = SCALE_INTERVALS.get(scale_name, SCALE_INTERVALS["Mayor"])
    use_flats = accidental_mode == "Bemoles (b)"
    chromatic = CHROMATIC_FLAT_NAMES if use_flats else CHROMATIC_SHARP_NAMES
    root_semitone = NOTE_TO_SEMITONE[root]
    names = tuple(chromatic[(root_semitone + semitone) % 12] for semitone in intervals)
    return ActiveScale(
        root_note=root,
        scale_name=scale_name if scale_name in SCALE_INTERVALS else "Mayor",
        accidental_mode=accidental_mode if accidental_mode in ACCIDENTAL_OPTIONS else ACCIDENTAL_OPTIONS[0],
        note_names=names,
        semitone_offsets=intervals,
    )


def build_scale(root_note: str, scale_name: str, accidental_mode: str = ACCIDENTAL_OPTIONS[0]) -> ActiveScale:
    return build_active_scale(root_note=root_note, scale_name=scale_name, accidental_mode=accidental_mode)


def note_name_from_scale_index(index: int, active_scale: ActiveScale) -> str:
    return active_scale.note_names[index % len(active_scale.note_names)]


def note_frequency_from_scale_index(
    index: int,
    active_scale: ActiveScale,
    octave_shift: int = 0,
    a4_hz: float = 440.0,
) -> float:
    semitone_in_scale = active_scale.semitone_offsets[index % len(active_scale.semitone_offsets)]
    midi = 60 + active_scale.root_semitone + semitone_in_scale + (octave_shift * 12)
    return a4_hz * math.pow(2.0, (midi - 69) / 12.0)


def chord_name_from_index(index: int) -> str:
    return CHORD_NAMES[index % len(CHORD_NAMES)]


def chord_intervals_from_index(index: int) -> tuple[int, ...]:
    chord_name = chord_name_from_index(index)
    return CHORD_INTERVALS[chord_name]


def _normalize_note_name(note_name: str) -> str:
    value = note_name.strip()
    if value in NOTE_TO_SEMITONE:
        return value
    return "Do"
