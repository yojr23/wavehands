from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Point2D:
    x: int
    y: int


@dataclass
class HandPointer:
    point: Point2D
    label: str = "Unknown"


@dataclass
class SelectionState:
    hovered_index: Optional[int] = None
    hover_started_at: float = 0.0
    stable_frames: int = 0
    selected_index: Optional[int] = None
    last_selected_at: float = 0.0


@dataclass
class PerformanceSettings:
    volume: float = 0.5
    note_duration_seconds: float = 0.6
    octave_shift: int = 0
    sustain: bool = False
    root_note: str = "Do"
    scale_name: str = "Mayor"
    accidental_mode: str = "Sostenidos (#)"
    instrument_name: str = "Sine"
    show_camera: bool = True
