import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from wavehands.domain.models import Point2D


@dataclass(frozen=True)
class WheelPalette:
    outline: tuple[int, int, int] = (90, 90, 90)
    selected_fill: tuple[int, int, int] = (0, 185, 255)
    hover_fill: tuple[int, int, int] = (70, 140, 180)
    segment_border: tuple[int, int, int] = (200, 200, 200)
    text: tuple[int, int, int] = (250, 250, 250)
    pointer_ring: tuple[int, int, int] = (0, 130, 220)
    pulse: tuple[int, int, int] = (255, 220, 90)
    center_fill: tuple[int, int, int] = (30, 30, 30)
    hold_text: tuple[int, int, int] = (230, 230, 230)


class NoteWheel:
    def __init__(
        self,
        center: Point2D,
        radius: int,
        note_names: Sequence[str],
        palette: Optional[WheelPalette] = None,
    ) -> None:
        self.center = center
        self.radius = radius
        self.note_names = list(note_names)
        self.palette = palette or WheelPalette()
        self._start_angle = -math.pi / 2.0  # Sector 0 arriba (12 en punto).
        self._sector_size = (2 * math.pi) / max(1, len(self.note_names))
        self._sector_deg = 360.0 / max(1, len(self.note_names))
        self._segment_angles: list[tuple[float, float]] = []
        self._text_positions: list[tuple[int, int]] = []
        self._rebuild_geometry_cache()

    def set_geometry(self, center: Point2D, radius: int) -> None:
        new_radius = max(24, radius)
        if self.center == center and self.radius == new_radius:
            return
        self.center = center
        self.radius = new_radius
        self._rebuild_geometry_cache()

    def point_to_note_index(self, point: Point2D) -> Optional[int]:
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dist = math.hypot(dx, dy)
        if dist > self.radius:
            return None

        angle = math.atan2(dy, dx)
        normalized = (angle - self._start_angle + 2 * math.pi) % (2 * math.pi)
        return int(normalized / self._sector_size)

    def draw(
        self,
        frame: np.ndarray,
        hovered_index: Optional[int],
        selected_index: Optional[int],
        hover_progress: float,
        pulse_strength: float = 0.0,
        pointer_inside: bool = False,
    ) -> None:
        sectors = len(self.note_names)
        outline_color = self.palette.outline
        text_scale = max(0.35, min(0.7, self.radius / 170.0))
        text_thick = 2 if text_scale >= 0.55 else 1

        for idx, note in enumerate(self.note_names):
            start, end = self._segment_angles[idx]
            color = outline_color
            thickness = 2

            if selected_index == idx:
                color = self.palette.selected_fill
                thickness = -1
            elif hovered_index == idx:
                color = self.palette.hover_fill
                thickness = -1

            cv2.ellipse(
                frame,
                (self.center.x, self.center.y),
                (self.radius, self.radius),
                0.0,
                start,
                end,
                color,
                thickness,
            )

            if selected_index == idx or hovered_index == idx:
                cv2.ellipse(
                    frame,
                    (self.center.x, self.center.y),
                    (self.radius, self.radius),
                    0.0,
                    start,
                    end,
                    self.palette.segment_border,
                    1,
                )

            tx, ty = self._text_positions[idx]
            cv2.putText(frame, note, (tx - 14, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, text_scale, self.palette.text, text_thick)

        cv2.circle(frame, (self.center.x, self.center.y), self.radius + 2, self.palette.segment_border, 2)

        if pointer_inside:
            cv2.circle(frame, (self.center.x, self.center.y), self.radius + 10, self.palette.pointer_ring, 2)

        if pulse_strength > 0.0:
            pulse_radius = self.radius + 16 + int(18 * pulse_strength)
            pulse_color = (
                int(self.palette.pulse[0] * pulse_strength),
                int(self.palette.pulse[1] * pulse_strength),
                int(self.palette.pulse[2] * pulse_strength),
            )
            cv2.circle(frame, (self.center.x, self.center.y), pulse_radius, pulse_color, 2)
        cv2.circle(frame, (self.center.x, self.center.y), int(self.radius * 0.26), self.palette.center_fill, -1)
        cv2.putText(
            frame,
            f"Hold: {int(hover_progress * 100):02d}%",
            (self.center.x - 52, self.center.y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.34, text_scale * 0.75),
            self.palette.hold_text,
            1,
        )

    def _rebuild_geometry_cache(self) -> None:
        sectors = len(self.note_names)
        self._segment_angles = []
        self._text_positions = []
        base_deg = math.degrees(self._start_angle)
        for idx in range(sectors):
            start = base_deg + (idx * self._sector_deg)
            end = start + self._sector_deg
            self._segment_angles.append((start, end))
            mid_angle = self._start_angle + (idx + 0.5) * self._sector_size
            tx = int(self.center.x + (self.radius * 0.67) * math.cos(mid_angle))
            ty = int(self.center.y + (self.radius * 0.67) * math.sin(mid_angle))
            self._text_positions.append((tx, ty))
