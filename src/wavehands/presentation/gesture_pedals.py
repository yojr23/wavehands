import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from wavehands.domain.models import Point2D


@dataclass
class PedalState:
    is_active: bool = False
    hover_started_at: float = 0.0
    cooldown_until: float = 0.0
    pulse_started_at: float = 0.0


class GesturePedal:
    def __init__(self, center: Point2D, radius: int, label: str, hold_seconds: float = 0.45, cooldown_seconds: float = 0.35) -> None:
        self.center = center
        self.radius = radius
        self.label = label
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.state = PedalState()

    def set_geometry(self, center: Point2D, radius: int) -> None:
        self.center = center
        self.radius = max(20, radius)

    def update(self, pointer: Optional[Point2D], now: float) -> bool:
        inside = pointer is not None and self._contains(pointer)
        if now < self.state.cooldown_until:
            return False

        if not inside:
            self.state.is_active = False
            self.state.hover_started_at = now
            return False

        if not self.state.is_active:
            self.state.is_active = True
            self.state.hover_started_at = now
            return False

        if now - self.state.hover_started_at >= self.hold_seconds:
            self.state.is_active = False
            self.state.cooldown_until = now + self.cooldown_seconds
            self.state.pulse_started_at = now
            self.state.hover_started_at = now
            return True

        return False

    def hover_progress(self, now: float) -> float:
        if not self.state.is_active:
            return 0.0
        elapsed = now - self.state.hover_started_at
        return max(0.0, min(1.0, elapsed / self.hold_seconds))

    def pulse_strength(self, now: float) -> float:
        if self.state.pulse_started_at == 0.0:
            return 0.0
        age = now - self.state.pulse_started_at
        if age > 0.35:
            return 0.0
        return 1.0 - (age / 0.35)

    def draw(self, frame: np.ndarray, now: float, active_text: str) -> None:
        pulse = self.pulse_strength(now)
        hover = self.hover_progress(now)

        base_color = (48, 48, 48)
        cv2.circle(frame, (self.center.x, self.center.y), self.radius, base_color, -1)
        cv2.circle(frame, (self.center.x, self.center.y), self.radius, (150, 150, 150), 2)

        if hover > 0.0:
            end_angle = int(360 * hover)
            cv2.ellipse(
                frame,
                (self.center.x, self.center.y),
                (self.radius + 4, self.radius + 4),
                -90.0,
                0.0,
                float(end_angle),
                (0, 200, 255),
                3,
            )

        if pulse > 0.0:
            glow_radius = self.radius + int(26 * pulse)
            glow_color = (int(255 * pulse), int(180 * pulse), int(80 * pulse))
            cv2.circle(frame, (self.center.x, self.center.y), glow_radius, glow_color, 2)

        cv2.putText(frame, self.label, (self.center.x - 42, self.center.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (238, 238, 238), 1)
        cv2.putText(frame, active_text, (self.center.x - 52, self.center.y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 230), 1)

    def _contains(self, point: Point2D) -> bool:
        return math.hypot(point.x - self.center.x, point.y - self.center.y) <= self.radius
