"""The renderer must be byte-identical to the implementation that produced the
released results.

This replaces `tests/test_env_render.py`, which asserted nothing -- it saved two PNGs
for a human to look at, and had gone stale: it called `env.step('right')` while
`step()` required JSON, so running it raised `ValueError` before reaching either save.

Fixtures are real observations from released runs. `tests/make_golden.py` regenerates
them; re-baselining is therefore a deliberate, reviewable act.
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

from mnist_pro.rendering import (CanvasSpec, deterministic_start, render_composite,
                                 render_observation)

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
with open(os.path.join(GOLDEN, "manifest.json")) as f:
    MANIFEST = json.load(f)

CASES = [(e, o) for e in MANIFEST for o in e["observations"]]
IDS = [f"{e['tag']}_ep{e['episode']}_step{o['step']}" for e, o in CASES]


def _spec(entry):
    return CanvasSpec(digits=entry["digits"], image_size=entry["image_size"],
                      box_size=entry["box_size"], step_size=entry["step_size"])


def _canvas(entry):
    return np.array(Image.open(os.path.join(GOLDEN, entry["canvas"])).convert("L"))


@pytest.mark.parametrize("entry,obs", CASES, ids=IDS)
def test_observation_matches_released_bytes(entry, obs):
    canvas = _canvas(entry)
    x, y = obs["window"]
    rendered = np.array(render_observation(canvas, x, y, _spec(entry)).convert("RGB"))
    expected = np.array(Image.open(os.path.join(GOLDEN, obs["file"])).convert("RGB"))
    assert rendered.shape == expected.shape
    assert np.array_equal(rendered, expected), (
        f"{obs['file']}: {int((rendered != expected).sum())} channel values differ")


@pytest.mark.parametrize("entry", MANIFEST, ids=[f"{e['tag']}_ep{e['episode']}" for e in MANIFEST])
def test_canvas_dimensions(entry):
    canvas = _canvas(entry)
    spec = _spec(entry)
    assert canvas.shape == spec.shape
    assert canvas.shape[1] == entry["image_size"] * entry["digits"]


@pytest.mark.parametrize("entry", MANIFEST, ids=[f"{e['tag']}_ep{e['episode']}" for e in MANIFEST])
def test_canvas_is_binarised(entry):
    """Strokes at 0, background at 255, nothing in between."""
    assert set(np.unique(_canvas(entry)).tolist()) <= {0, 255}


@pytest.mark.parametrize("entry", MANIFEST, ids=[f"{e['tag']}_ep{e['episode']}" for e in MANIFEST])
def test_deterministic_start_reproduces_step_zero(entry):
    """The content-hashed start must land where the released run started."""
    step0 = next((o for o in entry["observations"] if o["step"] == 0), None)
    if step0 is None:
        pytest.skip("no step 0 in fixture")
    assert deterministic_start(_canvas(entry), _spec(entry)) == tuple(step0["window"])


@pytest.mark.parametrize("entry", MANIFEST, ids=[f"{e['tag']}_ep{e['episode']}" for e in MANIFEST])
def test_full_composite_equals_canvas(entry):
    """Revealing every window must reproduce the canvas exactly, in RGB."""
    canvas, spec = _canvas(entry), _spec(entry)
    h, w = canvas.shape
    windows = [(x, y)
               for y in range(0, h - spec.box_size + 1, spec.box_size)
               for x in range(0, w - spec.box_size + 1, spec.box_size)]
    windows += [(w - spec.box_size, y) for y in range(0, h - spec.box_size + 1, spec.box_size)]
    windows += [(x, h - spec.box_size) for x in range(0, w - spec.box_size + 1, spec.box_size)]
    windows.append((w - spec.box_size, h - spec.box_size))
    out = np.array(render_composite(canvas, windows, spec, draw_border=False).convert("RGB"))
    assert np.array_equal(out, np.stack([canvas] * 3, axis=-1))


def test_border_is_drawn_over_the_window():
    """The outline sits on top of the glimpse, so it hides canvas content.

    Pinned because it is the source of the coverage discrepancy that
    `metrics.stroke_coverage` exposes: the published figure counts pixels the agent
    could not actually read.
    """
    entry = MANIFEST[0]
    canvas, spec = _canvas(entry), _spec(entry)
    x, y = entry["observations"][0]["window"]
    with_border = np.array(render_observation(canvas, x, y, spec, draw_border=True))
    without = np.array(render_observation(canvas, x, y, spec, draw_border=False))
    assert not np.array_equal(with_border, without)
    edge = with_border[y, x:x + spec.box_size]
    assert np.all(edge == np.array([0, 229, 255], dtype=np.uint8))
