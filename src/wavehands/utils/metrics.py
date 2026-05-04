from __future__ import annotations

import logging
import time
from dataclasses import dataclass


LOGGER = logging.getLogger("wavehands")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass
class RuntimeCounters:
    frames: int = 0
    frames_with_hands: int = 0
    note_changes: int = 0
    chord_changes: int = 0
    sustain_toggles: int = 0
    loop_state_changes: int = 0


class RuntimeMetrics:
    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self._start = time.time()
        self._last_log = self._start
        self.counters = RuntimeCounters()
        self._last_audio_callback_count = 0.0

    def tick_frame(self, has_hands: bool) -> None:
        self.counters.frames += 1
        if has_hands:
            self.counters.frames_with_hands += 1

    def maybe_log(
        self,
        selected_note: str | None,
        selected_chord: str | None,
        loop_mode: str,
        audio_metrics: dict[str, float],
    ) -> None:
        now = time.time()
        elapsed = now - self._last_log
        if elapsed < self.interval_seconds:
            return

        frames = max(1, self.counters.frames)
        fps = frames / elapsed
        hand_ratio = self.counters.frames_with_hands / frames

        callback_count = audio_metrics.get("callback_count", 0.0)
        callback_rate = (callback_count - self._last_audio_callback_count) / max(elapsed, 1e-6)
        self._last_audio_callback_count = callback_count

        LOGGER.info(
            (
                "fps=%.1f hands=%.0f%% note=%s chord=%s loop=%s "
                "note_changes=%d chord_changes=%d sustain_toggles=%d loop_changes=%d "
                "audio_cb_rate=%.1f/s audio_xruns=%.0f audio_cpu=%.2f voices=%.0f"
            ),
            fps,
            hand_ratio * 100.0,
            selected_note or "--",
            selected_chord or "--",
            loop_mode,
            self.counters.note_changes,
            self.counters.chord_changes,
            self.counters.sustain_toggles,
            self.counters.loop_state_changes,
            callback_rate,
            audio_metrics.get("xrun_count", 0.0),
            audio_metrics.get("callback_cpu_avg", 0.0),
            audio_metrics.get("loop_voice_count", 0.0),
        )

        self.counters.frames = 0
        self.counters.frames_with_hands = 0
        self._last_log = now
