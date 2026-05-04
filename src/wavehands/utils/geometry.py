import math
from typing import Optional

from wavehands.domain.models import Point2D


def distance(p1: Point2D, p2: Point2D) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def point_in_circle(point: Point2D, center: Point2D, radius: int) -> bool:
    return distance(point, center) <= radius


def point_to_sector(point: Point2D, center: Point2D, radius: int, sectors: int) -> Optional[int]:
    if not point_in_circle(point, center, radius):
        return None

    dx = point.x - center.x
    dy = point.y - center.y
    angle = math.atan2(dy, dx)
    angle = (angle + 2 * math.pi) % (2 * math.pi)
    sector_angle = (2 * math.pi) / sectors
    sector = int(angle / sector_angle)
    return sector

