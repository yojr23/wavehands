from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from wavehands.domain.models import PerformanceSettings


@dataclass
class SliderControl:
    label: str
    x: int
    y: int
    width: int
    min_value: float
    max_value: float
    value: float

    @property
    def track_y(self) -> int:
        return self.y + 24

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= (self.x + self.width) and self.y <= py <= (self.y + 40)

    def set_from_x(self, px: int) -> None:
        ratio = (px - self.x) / float(self.width)
        ratio = max(0.0, min(1.0, ratio))
        self.value = self.min_value + ratio * (self.max_value - self.min_value)

    def draw(self, frame: np.ndarray) -> None:
        cv2.putText(frame, self.label, (self.x, self.y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (236, 242, 249), 1)
        cv2.line(frame, (self.x, self.track_y), (self.x + self.width, self.track_y), (75, 92, 114), 2)
        ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        knob_x = int(self.x + ratio * self.width)
        cv2.circle(frame, (knob_x, self.track_y), 8, (0, 190, 255), -1)

    def set_geometry(self, x: int, y: int, width: int) -> None:
        self.x = x
        self.y = y
        self.width = max(80, width)


@dataclass
class OptionSelector:
    label: str
    x: int
    y: int
    width: int
    height: int
    options: List[str]
    selected_index: int = 1

    def option_rect(self, index: int) -> Tuple[int, int, int, int]:
        option_w = self.width // len(self.options)
        ox = self.x + (index * option_w)
        return ox, self.y + 18, option_w - 4, self.height

    def on_click(self, px: int, py: int) -> bool:
        for idx in range(len(self.options)):
            ox, oy, ow, oh = self.option_rect(idx)
            if ox <= px <= ox + ow and oy <= py <= oy + oh:
                self.selected_index = idx
                return True
        return False

    def selected_value(self) -> str:
        return self.options[self.selected_index]

    def draw(self, frame: np.ndarray) -> None:
        cv2.putText(frame, self.label, (self.x, self.y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (236, 242, 249), 1)
        for idx, option in enumerate(self.options):
            ox, oy, ow, oh = self.option_rect(idx)
            active = idx == self.selected_index
            fill = (20, 133, 203) if active else (56, 71, 89)
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), fill, -1)
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (129, 148, 173), 1)
            cv2.putText(frame, option, (ox + 8, oy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (243, 249, 255), 1)

    def set_geometry(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = max(120, width)
        self.height = max(18, height)


@dataclass
class ToggleControl:
    label: str
    x: int
    y: int
    checked: bool = False

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + 180 and self.y <= py <= self.y + 24

    def toggle(self) -> None:
        self.checked = not self.checked

    def draw(self, frame: np.ndarray) -> None:
        box_color = (0, 180, 120) if self.checked else (56, 71, 89)
        cv2.rectangle(frame, (self.x, self.y), (self.x + 20, self.y + 20), box_color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + 20, self.y + 20), (129, 148, 173), 1)
        cv2.putText(frame, self.label, (self.x + 28, self.y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (236, 242, 249), 1)

    def set_geometry(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


@dataclass
class ActionButton:
    label: str
    x: int
    y: int
    width: int
    height: int
    fill_color: tuple[int, int, int] = (56, 71, 89)
    was_clicked: bool = False

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def click(self) -> None:
        self.was_clicked = True

    def consume_click(self) -> bool:
        if not self.was_clicked:
            return False
        self.was_clicked = False
        return True

    def draw(self, frame: np.ndarray) -> None:
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), self.fill_color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), (129, 148, 173), 1)
        cv2.putText(frame, self.label, (self.x + 12, self.y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (243, 249, 255), 1)

    def set_geometry(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = max(90, width)
        self.height = max(22, height)


class ControlPanel:
    def __init__(self, panel_x: int, panel_width: int) -> None:
        content_x = panel_x + 24
        slider_width = panel_width - 52

        self.volume_slider = SliderControl(
            label="Volumen",
            x=content_x,
            y=80,
            width=slider_width,
            min_value=0.0,
            max_value=1.0,
            value=0.5,
        )
        self.duration_slider = SliderControl(
            label="Longitud (seg)",
            x=content_x,
            y=150,
            width=slider_width,
            min_value=0.1,
            max_value=2.0,
            value=0.6,
        )
        self.range_selector = OptionSelector(
            label="Rango Octava",
            x=content_x,
            y=220,
            width=slider_width,
            height=30,
            options=["-1", "0", "+1", "+2"],
            selected_index=1,
        )
        self.sustain_toggle = ToggleControl(
            label="Sustain",
            x=content_x,
            y=300,
            checked=False,
        )
        self.record_toggle_button = ActionButton(
            label="Grabar",
            x=content_x,
            y=338,
            width=150,
            height=30,
            fill_color=(17, 132, 81),
        )
        self.record_stop_button = ActionButton(
            label="Detener",
            x=content_x + 160,
            y=338,
            width=120,
            height=30,
            fill_color=(173, 61, 61),
        )
        self._active_slider: Optional[SliderControl] = None
        self._volume_revision = 0
        self._record_state = "idle"
        self.layout(panel_x, panel_width, 360)

    def to_settings(self) -> PerformanceSettings:
        range_map = {"-1": -1, "0": 0, "+1": 1, "+2": 2}
        octave_shift = range_map[self.range_selector.selected_value()]
        return PerformanceSettings(
            volume=self.volume_slider.value,
            note_duration_seconds=self.duration_slider.value,
            octave_shift=octave_shift,
            sustain=self.sustain_toggle.checked,
        )

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.volume_slider.contains(x, y):
                self._active_slider = self.volume_slider
                previous = self.volume_slider.value
                self._active_slider.set_from_x(x)
                if self.volume_slider.value != previous:
                    self._volume_revision += 1
                return
            if self.duration_slider.contains(x, y):
                self._active_slider = self.duration_slider
                self._active_slider.set_from_x(x)
                return
            if self.range_selector.on_click(x, y):
                return
            if self.sustain_toggle.contains(x, y):
                self.sustain_toggle.toggle()
                return
            if self.record_toggle_button.contains(x, y):
                self.record_toggle_button.click()
                return
            if self.record_stop_button.contains(x, y):
                self.record_stop_button.click()
                return

        if event == cv2.EVENT_MOUSEMOVE and self._active_slider is not None:
            previous = self._active_slider.value
            self._active_slider.set_from_x(x)
            if self._active_slider is self.volume_slider and self._active_slider.value != previous:
                self._volume_revision += 1
            return

        if event == cv2.EVENT_LBUTTONUP:
            self._active_slider = None

    def layout(self, panel_x: int, panel_width: int, canvas_height: int) -> None:
        content_x = panel_x + 16
        content_w = max(120, panel_width - 32)
        y = 74
        gap = max(10, canvas_height // 36)

        self.volume_slider.set_geometry(content_x, y, content_w)
        y += 40 + gap
        self.duration_slider.set_geometry(content_x, y, content_w)
        y += 40 + gap
        self.range_selector.set_geometry(content_x, y, content_w, 30)
        y += 56 + gap
        self.sustain_toggle.set_geometry(content_x, y)
        y += 36 + gap
        left_w = max(96, int(content_w * 0.56))
        right_w = max(84, content_w - left_w - 10)
        self.record_toggle_button.set_geometry(content_x, y, left_w, 30)
        self.record_stop_button.set_geometry(content_x + left_w + 10, y, right_w, 30)

    def draw(self, frame: np.ndarray, panel_x: int) -> None:
        panel_h = frame.shape[0]
        cv2.rectangle(frame, (panel_x, 0), (frame.shape[1], panel_h), (15, 23, 34), -1)
        cv2.putText(frame, "Controles", (panel_x + 16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 252, 255), 2)

        self.volume_slider.draw(frame)
        self.duration_slider.draw(frame)
        self.range_selector.draw(frame)
        self.sustain_toggle.draw(frame)
        self.record_toggle_button.draw(frame)
        self.record_stop_button.draw(frame)

    def set_sustain(self, enabled: bool) -> None:
        self.sustain_toggle.checked = enabled

    def consume_record_toggle(self) -> bool:
        return self.record_toggle_button.consume_click()

    def consume_record_stop(self) -> bool:
        return self.record_stop_button.consume_click()

    def consume_clear_loop(self) -> bool:
        # Compatibilidad con controladores que usan el flujo "Clear Loop".
        return self.record_stop_button.consume_click()

    def set_record_state(self, state: str) -> None:
        self._record_state = state
        if state == "recording":
            self.record_toggle_button.label = "Pausar"
            self.record_toggle_button.fill_color = (190, 134, 34)
            self.record_stop_button.fill_color = (173, 61, 61)
        elif state == "paused":
            self.record_toggle_button.label = "Reanudar"
            self.record_toggle_button.fill_color = (17, 132, 81)
            self.record_stop_button.fill_color = (173, 61, 61)
        else:
            self.record_toggle_button.label = "Grabar"
            self.record_toggle_button.fill_color = (17, 132, 81)
            self.record_stop_button.fill_color = (93, 106, 124)

    @property
    def volume_revision(self) -> int:
        return self._volume_revision
