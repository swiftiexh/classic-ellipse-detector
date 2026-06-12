"""Lightweight CenterNet-style ellipse detector."""

from .geometry import EllipseRecord
from .model import EllipseCenterNet

__all__ = ["EllipseRecord", "EllipseCenterNet"]
