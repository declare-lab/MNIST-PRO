"""Parity with the released runs, from raw MNIST through to the canvas.

`test_renderer_golden.py` proves observations are rendered identically *given* a
canvas. These tests close the remaining link: that episode `i` still draws the same
source images as the released run, and that building a canvas from those images
reproduces the stored `original.png` byte-for-byte.

Skipped unless MNIST and the released logs are both present, so the suite still runs
in a bare checkout.
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

from mnist_pro.dataset import images_for, load_mnist, sample_balanced
from mnist_pro.rendering import CanvasSpec, build_canvas

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.environ.get("MNIST_PRO_LOGS", os.path.join(REPO, "..", "main_table_logs"))
DATA = os.environ.get("MNIST_PRO_DATA", os.path.join(REPO, "..", "data"))

RUNS = {
    1: "gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42"
       "_MemoryVisionAgent_hist1_evalsets10_20260824_110242",
    2: "multidigit_2_gemini-3.7-flash_img224_box64_step32_maxsteps78_seed42"
       "_MultiDigitMemoryVisionAgent_hist1_evalsets10_20260824_114120",
}

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(os.path.join(DATA, "MNIST")) and os.path.isdir(LOGS)),
    reason="needs MNIST under --data-dir and the released logs; "
           "set MNIST_PRO_DATA / MNIST_PRO_LOGS to override")


@pytest.fixture(scope="module")
def dataset():
    return load_mnist(DATA)


def released_episodes(digits):
    path = os.path.join(LOGS, RUNS[digits], "results_summary.json")
    with open(path) as f:
        return json.load(f)["episodes"]


@pytest.mark.parametrize("digits", [1, 2])
def test_sampler_reproduces_released_episode_mapping(dataset, digits):
    """Episode ids must still refer to the same images, or every published episode
    id silently changes meaning."""
    episodes = released_episodes(digits)
    specs = sample_balanced(dataset, num_sets=10, digits=digits, seed=42)
    assert len(specs) >= len(episodes)
    for released, spec in zip(episodes, specs):
        expected = (tuple(released["indices"]) if digits > 1
                    else (released["eval_index"],))
        assert spec.indices == expected, f"episode {released['episode_id']}"
        assert spec.label == str(released["label"])


@pytest.mark.parametrize("digits", [1, 2])
def test_build_canvas_reproduces_released_original_png(dataset, digits):
    """Invert, bilinear resize, threshold 200 -- byte-identical to what was run."""
    episodes = released_episodes(digits)[:5]
    specs = sample_balanced(dataset, num_sets=10, digits=digits, seed=42)
    spec_geom = CanvasSpec(digits=digits, image_size=224, box_size=64, step_size=32)
    for released, spec in zip(episodes, specs):
        built = build_canvas(images_for(dataset, spec), spec_geom)
        stored = np.array(Image.open(os.path.join(
            LOGS, RUNS[digits], f"episode_{released['episode_id']}",
            "original.png")).convert("L"))
        assert built.shape == stored.shape
        assert np.array_equal(built, stored), (
            f"episode {released['episode_id']}: "
            f"{int((built != stored).sum())} pixels differ")


def test_max_steps_matches_every_released_run():
    """The step budget must equal what each released run was actually given.

    Recomputing it wrongly would silently change how long agents may explore, which
    changes coverage and therefore every downstream number.
    """
    import re

    from mnist_pro.matrix import parse_run_dir
    from mnist_pro.runner import default_max_steps

    checked = 0
    for name in sorted(os.listdir(LOGS)):
        match = re.search(r"_maxsteps(\d+)_", name)
        cell = parse_run_dir(name)
        if not (match and cell):
            continue
        assert default_max_steps(cell) == int(match.group(1)), name
        checked += 1
    assert checked > 10, "expected to check many released runs"
