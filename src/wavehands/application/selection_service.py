from dataclasses import dataclass
from typing import Optional

from wavehands.config import SelectionConfig
from wavehands.domain.models import SelectionState


@dataclass
class SelectionResult:
    selected_index: Optional[int] = None
    just_selected: bool = False


class HoverSelectionService:
    def __init__(self, config: SelectionConfig) -> None:
        self._config = config
        self.state = SelectionState()

    def update(self, candidate_index: Optional[int], now: float) -> SelectionResult:
        if candidate_index is None:
            self.state.hovered_index = None
            self.state.hover_started_at = now
            self.state.stable_frames = 0
            return SelectionResult(selected_index=self.state.selected_index, just_selected=False)

        if self.state.hovered_index != candidate_index:
            self.state.hovered_index = candidate_index
            self.state.hover_started_at = now
            self.state.stable_frames = 1
            return SelectionResult(selected_index=self.state.selected_index, just_selected=False)

        self.state.stable_frames += 1

        if self.state.stable_frames < self._config.stable_frames:
            return SelectionResult(selected_index=self.state.selected_index, just_selected=False)

        hover_time = now - self.state.hover_started_at
        cooldown_time = now - self.state.last_selected_at
        ready = hover_time >= self._config.hover_seconds and cooldown_time >= self._config.cooldown_seconds
        if not ready:
            return SelectionResult(selected_index=self.state.selected_index, just_selected=False)

        self.state.selected_index = candidate_index
        self.state.last_selected_at = now
        self.state.stable_frames = 0
        self.state.hover_started_at = now
        return SelectionResult(selected_index=candidate_index, just_selected=True)
