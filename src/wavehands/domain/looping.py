from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class LoopEvent:
    offset_seconds: float
    frequency_hz: float
    duration_seconds: float
    velocity: float


@dataclass
class LoopLayer:
    events: List[LoopEvent] = field(default_factory=list)

