from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class EllipseRecord:
    cx: float
    cy: float
    a: float
    b: float
    theta: float
    score: float = 1.0

    def canonical(self) -> "EllipseRecord":
        a, b, theta = float(self.a), float(self.b), float(self.theta)
        if b > a:
            a, b = b, a
            theta += math.pi / 2.0
        theta = ((theta + math.pi / 2.0) % math.pi) - math.pi / 2.0
        return replace(self, a=a, b=b, theta=theta)


@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    size: int
    scale: float
    pad_x: float
    pad_y: float

    @classmethod
    def create(cls, width: int, height: int, size: int) -> "LetterboxTransform":
        scale = min(size / width, size / height)
        return cls(width, height, size, scale, (size - width * scale) / 2.0, (size - height * scale) / 2.0)

    def image(self, image: np.ndarray) -> np.ndarray:
        width = int(round(self.original_width * self.scale))
        height = int(round(self.original_height * self.scale))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        left = int(round(self.pad_x))
        top = int(round(self.pad_y))
        canvas[top : top + height, left : left + width] = resized
        return canvas

    def ellipse(self, ellipse: EllipseRecord) -> EllipseRecord:
        return replace(
            ellipse,
            cx=ellipse.cx * self.scale + self.pad_x,
            cy=ellipse.cy * self.scale + self.pad_y,
            a=ellipse.a * self.scale,
            b=ellipse.b * self.scale,
        )

    def inverse_ellipse(self, ellipse: EllipseRecord) -> EllipseRecord:
        return replace(
            ellipse,
            cx=(ellipse.cx - self.pad_x) / self.scale,
            cy=(ellipse.cy - self.pad_y) / self.scale,
            a=ellipse.a / self.scale,
            b=ellipse.b / self.scale,
        ).canonical()


def horizontal_flip(ellipse: EllipseRecord, width: int) -> EllipseRecord:
    return EllipseRecord(width - 1 - ellipse.cx, ellipse.cy, ellipse.a, ellipse.b, -ellipse.theta, ellipse.score).canonical()


def draw_ellipses(image: np.ndarray, ellipses: Iterable[EllipseRecord], color=(0, 0, 255)) -> np.ndarray:
    result = image.copy()
    for ellipse in ellipses:
        e = ellipse.canonical()
        cv2.ellipse(
            result,
            ((e.cx, e.cy), (2.0 * e.a, 2.0 * e.b), math.degrees(e.theta)),
            color,
            2,
        )
        if e.score != 1.0:
            cv2.putText(result, f"{e.score:.2f}", (round(e.cx), round(e.cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return result


def ellipse_overlap(lhs: EllipseRecord, rhs: EllipseRecord, step: float = 0.2) -> float:
    lhs_eq = _shape_to_equation(lhs)
    rhs_eq = _shape_to_equation(rhs)
    y_min = math.floor(max(lhs.cy - max(lhs.a, lhs.b), rhs.cy - max(rhs.a, rhs.b)))
    y_max = math.ceil(min(lhs.cy + max(lhs.a, lhs.b), rhs.cy + max(rhs.a, rhs.b)))
    if y_min >= y_max:
        return 0.0
    overlap = 0.0
    y = float(y_min)
    while y <= y_max + 1e-6:
        lrange = _range_at_y(lhs_eq, y)
        rrange = _range_at_y(rhs_eq, y)
        if lrange is not None and rrange is not None:
            overlap += max(0.0, min(lrange[1], rrange[1]) - max(lrange[0], rrange[0]))
        y += step
    intersection = overlap * step
    union = math.pi * lhs.a * lhs.b + math.pi * rhs.a * rhs.b - intersection
    return intersection / union if union > 0.0 else 0.0


def _shape_to_equation(ellipse: EllipseRecord) -> tuple[float, ...]:
    e = ellipse.canonical()
    ct, st = math.cos(e.theta), math.sin(e.theta)
    s2 = math.sin(2.0 * e.theta)
    ct2, st2 = ct * ct, st * st
    ai, bi = 1.0 / (e.a * e.a), 1.0 / (e.b * e.b)
    values = [
        ct2 * ai + st2 * bi,
        -0.5 * s2 * (bi - ai),
        ct2 * bi + st2 * ai,
        (-e.cx * st2 + e.cy * s2 / 2.0) * bi - (e.cx * ct2 + e.cy * s2 / 2.0) * ai,
        (-e.cy * ct2 + e.cx * s2 / 2.0) * bi - (e.cy * st2 + e.cx * s2 / 2.0) * ai,
        ((e.cx * ct + e.cy * st) / e.a) ** 2 + ((e.cy * ct - e.cx * st) / e.b) ** 2 - 1.0,
    ]
    scale = 1.0 / math.sqrt(abs(values[0] * values[2] - values[1] * values[1]))
    return tuple(value * scale for value in values)


def _range_at_y(eq: tuple[float, ...], y: float) -> tuple[float, float] | None:
    a, b, c, d, e, f = eq
    delta = (b * y + d) ** 2 - a * (c * y * y + 2.0 * e * y + f)
    if delta < 0.0:
        return None
    x1 = (-(b * y + d) - math.sqrt(delta)) / a
    x2 = (-(b * y + d) + math.sqrt(delta)) / a
    return (min(x1, x2), max(x1, x2))
