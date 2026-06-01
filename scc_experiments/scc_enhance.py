"""SCC-Enhance: Adaptive image preprocessing for ellipse detection.

Applies CLAHE + bilateral filter to enhance edge visibility before detection.
Pure wrapper — does not modify C++ source.
"""
from __future__ import annotations

import cv2
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnhanceParams:
    name: str
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0


LIGHT = EnhanceParams(
    name="light",
    clahe_clip_limit=2.0,
    clahe_tile_grid=(8, 8),
    bilateral_d=5,
    bilateral_sigma_color=50.0,
    bilateral_sigma_space=50.0,
)

MEDIUM = EnhanceParams(
    name="medium",
    clahe_clip_limit=3.0,
    clahe_tile_grid=(8, 8),
    bilateral_d=7,
    bilateral_sigma_color=75.0,
    bilateral_sigma_space=75.0,
)

STRONG = EnhanceParams(
    name="strong",
    clahe_clip_limit=4.0,
    clahe_tile_grid=(8, 8),
    bilateral_d=9,
    bilateral_sigma_color=100.0,
    bilateral_sigma_space=100.0,
)

ALL_PARAMS = [LIGHT, MEDIUM, STRONG]


def enhance_image(image: cv2.Mat, params: EnhanceParams) -> cv2.Mat:
    """Apply CLAHE + bilateral filter to enhance edge visibility."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip_limit,
        tileGridSize=params.clahe_tile_grid,
    )
    enhanced = clahe.apply(gray)

    enhanced = cv2.bilateralFilter(
        enhanced,
        d=params.bilateral_d,
        sigmaColor=params.bilateral_sigma_color,
        sigmaSpace=params.bilateral_sigma_space,
    )

    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return result


def enhance_and_save(
    src: Path, dst: Path, params: EnhanceParams
) -> None:
    """Load image from src, enhance, save to dst."""
    image = cv2.imread(str(src))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {src}")
    enhanced = enhance_image(image, params)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), enhanced)
