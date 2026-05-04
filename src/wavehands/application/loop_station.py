from dataclasses import dataclass, field
from typing import List

from wavehands.domain.looping import LoopEvent, LoopLayer


@dataclass
class LoopState:
    mode: str = "idle"  # idle | recording | playing | overdubbing
    loop_length_seconds: float = 0.0
    layers: List[LoopLayer] = field(default_factory=list)


class LoopStationService:
    def __init__(self, min_loop_seconds: float = 0.75) -> None:
        self.state = LoopState()
        self._min_loop_seconds = min_loop_seconds
        self._recording_started_at = 0.0
        self._playback_started_at = 0.0
        self._pending_events: List[LoopEvent] = []
        self._last_poll_at = 0.0
        self._play_cursors: List[int] = []

    def cycle(self, now: float) -> str:
        if self.state.mode == "idle":
            self._start_recording(now)
            return self.state.mode
        if self.state.mode == "recording":
            self._stop_base_recording(now)
            return self.state.mode
        if self.state.mode == "playing":
            self._start_overdub(now)
            return self.state.mode
        if self.state.mode == "overdubbing":
            self._stop_overdub(now)
            return self.state.mode
        return self.state.mode

    def clear(self) -> None:
        self.state = LoopState()
        self._recording_started_at = 0.0
        self._playback_started_at = 0.0
        self._pending_events = []
        self._last_poll_at = 0.0
        self._play_cursors = []

    def record_note_event(self, frequency_hz: float, duration_seconds: float, velocity: float, now: float) -> None:
        if self.state.mode not in ("recording", "overdubbing"):
            return

        if self.state.mode == "recording":
            offset = now - self._recording_started_at
        else:
            offset = (now - self._playback_started_at) % self.state.loop_length_seconds

        event = LoopEvent(
            offset_seconds=max(0.0, offset),
            frequency_hz=max(20.0, frequency_hz),
            duration_seconds=max(0.05, duration_seconds),
            velocity=max(0.0, min(1.0, velocity)),
        )
        self._pending_events.append(event)

    def poll_due_events(self, now: float) -> List[LoopEvent]:
        if self.state.mode not in ("playing", "overdubbing"):
            self._last_poll_at = now
            return []
        if self.state.loop_length_seconds <= 0.0 or not self.state.layers:
            self._last_poll_at = now
            return []
        if self._last_poll_at == 0.0:
            self._last_poll_at = now
            return []

        prev_pos = (self._last_poll_at - self._playback_started_at) % self.state.loop_length_seconds
        curr_pos = (now - self._playback_started_at) % self.state.loop_length_seconds
        wrapped = curr_pos < prev_pos

        if len(self._play_cursors) != len(self.state.layers):
            self._play_cursors = [0 for _ in self.state.layers]

        due: List[LoopEvent] = []
        for layer_idx, layer in enumerate(self.state.layers):
            events = layer.events
            if not events:
                continue

            cursor = max(0, min(self._play_cursors[layer_idx], len(events)))
            if wrapped:
                while cursor < len(events):
                    due.append(events[cursor])
                    cursor += 1
                cursor = 0
                while cursor < len(events) and events[cursor].offset_seconds <= curr_pos:
                    due.append(events[cursor])
                    cursor += 1
            else:
                while cursor < len(events) and events[cursor].offset_seconds <= curr_pos:
                    if events[cursor].offset_seconds > prev_pos:
                        due.append(events[cursor])
                    cursor += 1

            self._play_cursors[layer_idx] = cursor

        self._last_poll_at = now
        return due

    def _start_recording(self, now: float) -> None:
        self.clear()
        self.state.mode = "recording"
        self._recording_started_at = now
        self._pending_events = []
        self._last_poll_at = now
        self._play_cursors = []

    def _stop_base_recording(self, now: float) -> None:
        loop_length = max(self._min_loop_seconds, now - self._recording_started_at)
        first_layer = LoopLayer(events=sorted(self._pending_events, key=lambda ev: ev.offset_seconds))
        self.state.layers = [first_layer] if first_layer.events else []
        self.state.loop_length_seconds = loop_length
        self.state.mode = "playing" if self.state.layers else "idle"
        self._playback_started_at = now
        self._pending_events = []
        self._last_poll_at = now
        self._play_cursors = [0 for _ in self.state.layers]

    def _start_overdub(self, now: float) -> None:
        if not self.state.layers or self.state.loop_length_seconds <= 0.0:
            return
        self.state.mode = "overdubbing"
        self._recording_started_at = now
        self._pending_events = []

    def _stop_overdub(self, _now: float) -> None:
        overdub_layer = LoopLayer(events=sorted(self._pending_events, key=lambda ev: ev.offset_seconds))
        if overdub_layer.events:
            self.state.layers.append(overdub_layer)
        self._pending_events = []
        self.state.mode = "playing"
        self._play_cursors = [0 for _ in self.state.layers]
