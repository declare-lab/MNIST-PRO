"""Exploration and coverage metrics.

`stroke_coverage` can be computed using either the full glimpse windows or only the
readable portion of each window (excluding the border ring):

    coverage_mode="windows"   the union of the full glimpse windows
    coverage_mode="readable"  the union minus the border ring
"""

from __future__ import annotations

import numpy as np

from .rendering import BORDER_WIDTH, STROKE_VALUE, CanvasSpec

COVERAGE_MODES = ("windows", "readable")


def window_mask(shape, windows, box_size: int) -> np.ndarray:
    """Union of the glimpse windows, as a boolean mask."""
    m = np.zeros(shape, dtype=bool)
    for x, y in windows:
        m[y : y + box_size, x : x + box_size] = True
    return m


def readable_mask(
    shape, windows, box_size: int, border_width: int = BORDER_WIDTH
) -> np.ndarray:
    """Union of the parts of each window that showed canvas content.

    The outline occupies `border_width` pixels inside each window edge, so content
    is visible only in the inset rectangle. A pixel counts as readable if it is
    inside the inset of *any* window.
    """
    m = np.zeros(shape, dtype=bool)
    b = border_width
    h, w = shape
    for x, y in windows:
        y0, y1 = min(y + b, h), max(y + box_size - b, 0)
        x0, x1 = min(x + b, w), max(x + box_size - b, 0)
        if y1 > y0 and x1 > x0:
            m[y0:y1, x0:x1] = True
    return m


def coverage_mask(
    shape,
    windows,
    box_size: int,
    mode: str = "windows",
    border_width: int = BORDER_WIDTH,
) -> np.ndarray:
    if mode == "windows":
        return window_mask(shape, windows, box_size)
    if mode == "readable":
        return readable_mask(shape, windows, box_size, border_width)
    raise ValueError(
        f"unknown coverage mode {mode!r}; expected one of {COVERAGE_MODES}"
    )


def stroke_coverage(
    canvas: np.ndarray,
    windows,
    box_size: int,
    mode: str = "windows",
    border_width: int = BORDER_WIDTH,
) -> float:
    """Fraction of stroke pixels inside the covered region, in [0, 1]."""
    stroke = canvas == STROKE_VALUE
    total = int(stroke.sum())
    if total == 0:
        return 0.0
    m = coverage_mask(canvas.shape, windows, box_size, mode, border_width)
    return float((stroke & m).sum()) / total


def area_coverage(
    canvas: np.ndarray, windows, box_size: int, mode: str = "windows"
) -> float:
    m = coverage_mask(canvas.shape, windows, box_size, mode)
    return float(m.sum()) / m.size


def exploration_stats(canvas: np.ndarray, windows, spec: CanvasSpec) -> dict:
    """Calculate observation, window, revisits, and stroke coverage statistics."""
    unique = sorted(set(map(tuple, windows)))
    revisits = len(windows) - len(unique)
    cov_w = stroke_coverage(canvas, unique, spec.box_size, "windows")
    cov_r = stroke_coverage(canvas, unique, spec.box_size, "readable")
    return {
        "n_observations": len(windows),
        "n_unique_windows": len(unique),
        "n_revisits": revisits,
        "revisit_rate": (revisits / len(windows)) if windows else 0.0,
        "stroke_coverage": round(cov_w, 6),
        "stroke_coverage_readable": round(cov_r, 6),
        "border_occlusion": round(cov_w - cov_r, 6),
        "area_coverage": round(area_coverage(canvas, unique, spec.box_size), 6),
    }
