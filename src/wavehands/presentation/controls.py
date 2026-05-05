from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from wavehands.domain.models import PerformanceSettings
from wavehands.domain.notes import ACCIDENTAL_OPTIONS, ROOT_NOTE_OPTIONS, SCALE_NAMES


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
class CycleSelector:
    label: str
    x: int
    y: int
    width: int
    height: int
    options: List[str]
    selected_index: int = 0

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def on_click(self, px: int, py: int) -> bool:
        if not self.contains(px, py) or len(self.options) <= 1:
            return False
        arrow_w = min(28, self.width // 4)
        if px <= self.x + arrow_w:
            self.selected_index = (self.selected_index - 1) % len(self.options)
            return True
        if px >= self.x + self.width - arrow_w:
            self.selected_index = (self.selected_index + 1) % len(self.options)
            return True
        return False

    def selected_value(self) -> str:
        return self.options[self.selected_index]

    def draw(self, frame: np.ndarray) -> None:
        label_y = self.y + 12
        row_y = self.y + 18
        cv2.putText(frame, self.label, (self.x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (236, 242, 249), 1)
        cv2.rectangle(frame, (self.x, row_y), (self.x + self.width, row_y + self.height), (56, 71, 89), -1)
        cv2.rectangle(frame, (self.x, row_y), (self.x + self.width, row_y + self.height), (129, 148, 173), 1)

        arrow_w = min(28, self.width // 4)
        left_center = (self.x + arrow_w // 2, row_y + (self.height // 2))
        right_center = (self.x + self.width - arrow_w // 2, row_y + (self.height // 2))
        cv2.putText(frame, "<", (left_center[0] - 5, left_center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (243, 249, 255), 1)
        cv2.putText(frame, ">", (right_center[0] - 5, right_center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (243, 249, 255), 1)

        value = self.selected_value()
        (text_w, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        text_x = self.x + (self.width - text_w) // 2
        text_y = row_y + int(self.height * 0.68)
        cv2.putText(frame, value, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (243, 249, 255), 1)

    def set_geometry(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = max(120, width)
        self.height = max(20, height)


@dataclass
class ToggleControl:
    label: str
    x: int
    y: int
    width: int = 180
    checked: bool = False

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + 24

    def toggle(self) -> None:
        self.checked = not self.checked

    def draw(self, frame: np.ndarray) -> None:
        box_color = (0, 180, 120) if self.checked else (56, 71, 89)
        cv2.rectangle(frame, (self.x, self.y), (self.x + 20, self.y + 20), box_color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + 20, self.y + 20), (129, 148, 173), 1)
        cv2.putText(frame, self.label, (self.x + 28, self.y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (236, 242, 249), 1)

    def set_geometry(self, x: int, y: int, width: Optional[int] = None) -> None:
        self.x = x
        self.y = y
        if width is not None:
            self.width = max(80, width)


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
        self.root_selector = CycleSelector(
            label="Tonica",
            x=content_x,
            y=272,
            width=slider_width,
            height=24,
            options=list(ROOT_NOTE_OPTIONS),
            selected_index=0,
        )
        self.scale_selector = CycleSelector(
            label="Escala",
            x=content_x,
            y=316,
            width=slider_width,
            height=24,
            options=list(SCALE_NAMES),
            selected_index=0,
        )
        self.accidental_selector = CycleSelector(
            label="Alteraciones",
            x=content_x,
            y=360,
            width=slider_width,
            height=24,
            options=list(ACCIDENTAL_OPTIONS),
            selected_index=0,
        )
        self.instrument_selector = CycleSelector(
            label="Instrumento",
            x=content_x,
            y=404,
            width=slider_width,
            height=24,
            options=["Sine", "Piano", "Drums"],
            selected_index=0,
        )
        self.sustain_toggle = ToggleControl(
            label="Sustain",
            x=content_x,
            y=444,
            width=120,
            checked=False,
        )
        self.camera_toggle = ToggleControl(
            label="Camara ON",
            x=content_x + 128,
            y=444,
            width=150,
            checked=True,
        )
        self.record_toggle_button = ActionButton(
            label="Grabar",
            x=content_x,
            y=480,
            width=150,
            height=30,
            fill_color=(17, 132, 81),
        )
        self.record_stop_button = ActionButton(
            label="Detener",
            x=content_x + 160,
            y=480,
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
            root_note=self.root_selector.selected_value(),
            scale_name=self.scale_selector.selected_value(),
            accidental_mode=self.accidental_selector.selected_value(),
            instrument_name=self.instrument_selector.selected_value(),
            show_camera=self.camera_toggle.checked,
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
            if self.root_selector.on_click(x, y):
                return
            if self.scale_selector.on_click(x, y):
                return
            if self.accidental_selector.on_click(x, y):
                return
            if self.instrument_selector.on_click(x, y):
                return
            if self.sustain_toggle.contains(x, y):
                self.sustain_toggle.toggle()
                return
            if self.camera_toggle.contains(x, y):
                self.camera_toggle.toggle()
                self.camera_toggle.label = "Camara ON" if self.camera_toggle.checked else "Camara OFF"
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
        y = 62
        gap = max(4, canvas_height // 96)

        self.volume_slider.set_geometry(content_x, y, content_w)
        y += 40 + gap
        self.duration_slider.set_geometry(content_x, y, content_w)
        y += 40 + gap
        self.range_selector.set_geometry(content_x, y, content_w, 30)
        y += 52 + gap
        self.root_selector.set_geometry(content_x, y, content_w, 24)
        y += 44 + gap
        self.scale_selector.set_geometry(content_x, y, content_w, 24)
        y += 44 + gap
        self.accidental_selector.set_geometry(content_x, y, content_w, 24)
        y += 44 + gap
        self.instrument_selector.set_geometry(content_x, y, content_w, 24)
        y += 42 + gap

        toggle_gap = 10
        toggle_w = max(90, (content_w - toggle_gap) // 2)
        self.sustain_toggle.set_geometry(content_x, y, toggle_w)
        self.camera_toggle.set_geometry(content_x + toggle_w + toggle_gap, y, toggle_w)
        y += 30 + gap

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
        self.root_selector.draw(frame)
        self.scale_selector.draw(frame)
        self.accidental_selector.draw(frame)
        self.instrument_selector.draw(frame)
        self.sustain_toggle.draw(frame)
        self.camera_toggle.draw(frame)
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
