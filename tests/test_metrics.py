"""Coverage metrics, including the border-occlusion discrepancy.

Nothing in the original test suite touched the coverage metric, although it is the
figure reported in the exploration-efficiency table.
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

from mnist_pro.metrics import (area_coverage, exploration_stats, readable_mask,
                               stroke_coverage, window_mask)
from mnist_pro.rendering import CanvasSpec

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
with open(os.path.join(GOLDEN, "manifest.json")) as f:
    MANIFEST = json.load(f)


def canvas_of(entry):
    return np.array(Image.open(os.path.join(GOLDEN, entry["canvas"])).convert("L"))


def test_window_mask_covers_exactly_the_box():
    m = window_mask((224, 224), [(0, 0)], 64)
    assert m.sum() == 64 * 64
    assert m[0, 0] and m[63, 63] and not m[64, 64]


def test_full_canvas_gives_unit_coverage():
    entry = MANIFEST[0]
    canvas = canvas_of(entry)
    h, w = canvas.shape
    windows = [(x, y) for y in range(0, h, 64) for x in range(0, w, 64)]
    assert stroke_coverage(canvas, windows, 64) == pytest.approx(1.0)


def test_no_windows_gives_zero_coverage():
    assert stroke_coverage(canvas_of(MANIFEST[0]), [], 64) == 0.0


def test_readable_mask_is_inset_by_the_border_width():
    m = readable_mask((224, 224), [(0, 0)], 64, border_width=2)
    assert m.sum() == 60 * 60
    assert not m[0, 0] and not m[1, 1] and m[2, 2] and m[61, 61] and not m[62, 62]


def test_readable_coverage_never_exceeds_window_coverage():
    """The border hides content, so the honest figure is never the larger one."""
    for entry in MANIFEST:
        canvas = canvas_of(entry)
        windows = [(x, y) for y in range(0, canvas.shape[0] - 63, 32)
                   for x in range(0, canvas.shape[1] - 63, 32)][:12]
        w = stroke_coverage(canvas, windows, 64, mode="windows")
        r = stroke_coverage(canvas, windows, 64, mode="readable")
        assert r <= w + 1e-9


def test_border_occlusion_is_reported_not_hidden():
    entry = MANIFEST[0]
    canvas = canvas_of(entry)
    spec = CanvasSpec(digits=entry["digits"])
    windows = [tuple(o["window"]) for o in entry["observations"]]
    stats = exploration_stats(canvas, windows, spec)
    assert stats["border_occlusion"] == pytest.approx(
        stats["stroke_coverage"] - stats["stroke_coverage_readable"], abs=1e-9)
    assert stats["border_occlusion"] >= 0


def test_overlapping_windows_recover_what_one_border_hid():
    """Because glimpses overlap at step 32 < box 64, a pixel hidden under one
    border is usually interior to a neighbouring window."""
    shape = (224, 224)
    single = readable_mask(shape, [(0, 0)], 64)
    overlapped = readable_mask(shape, [(0, 0), (32, 0), (0, 32)], 64)
    hidden_then_recovered = (~single) & overlapped
    assert hidden_then_recovered.sum() > 0


def test_exploration_stats_counts_revisits():
    canvas = canvas_of(MANIFEST[0])
    spec = CanvasSpec(digits=MANIFEST[0]["digits"])
    stats = exploration_stats(canvas, [(0, 0), (32, 0), (0, 0)], spec)
    assert stats["n_observations"] == 3
    assert stats["n_unique_windows"] == 2
    assert stats["n_revisits"] == 1


def test_unknown_coverage_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown coverage mode"):
        stroke_coverage(canvas_of(MANIFEST[0]), [(0, 0)], 64, mode="whatever")


def test_area_coverage_is_a_fraction():
    v = area_coverage(canvas_of(MANIFEST[0]), [(0, 0)], 64)
    assert 0.0 < v < 1.0
