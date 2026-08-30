"""The evaluation driver, exercised without a network or a dataset.

The driver is where an episode, an agent and the recorder meet, so its failure modes
are the expensive ones: a backend outage that aborts a sweep, a run whose condition
cannot be recovered afterwards, a control that has drifted from the main path.
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

from mnist_pro import runner
from mnist_pro.backends import BackendError
from mnist_pro.dataset import EpisodeSpec
from mnist_pro.matrix import Cell


class FakeDataset:
    """Stands in for torchvision MNIST: indexable, yields (image, label)."""

    def __init__(self, n=40):
        self.items = []
        for i in range(n):
            a = np.zeros((28, 28), dtype=np.uint8)
            a[6:22, 9 + (i % 3):13 + (i % 3)] = 255
            a[6:10, 9:19] = 255
            self.items.append((Image.fromarray(a, mode="L"), i % 10))

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)


class ScriptedBackend:
    def __init__(self, script):
        self.script = list(script)
        self.n = 0

    def generate(self, system_instruction, contents):
        self.n += 1
        value = self.script[min(self.n - 1, len(self.script) - 1)]
        if isinstance(value, Exception):
            raise value
        return value, {"total_tokens": 5}


@pytest.fixture
def cell():
    return Cell(model="fake", digits=1, memory="event_logging", horizon=1,
                turn_mode="turn_based", harness="turn_based")


def patch_backend(monkeypatch, script):
    backend = ScriptedBackend(script)
    monkeypatch.setattr(runner, "get_backend", lambda *a, **k: backend)
    return backend


def test_max_steps_matches_the_released_runs(cell):
    """36 at one digit, 78 at two -- the values the published runs were given."""
    assert runner.default_max_steps(cell) == 36
    assert runner.default_max_steps(Cell(model="m", digits=2)) == 78


@pytest.mark.parametrize("box,step,digits,expected", [
    (32, 16, 1, 13 * 13),
    (128, 64, 1, 3 * 3),
    (64, 32, 3, 20 * 6),
])
def test_max_steps_follows_the_original_formula(box, step, digits, expected):
    cell = Cell(model="m", digits=digits, box_size=box, step_size=step)
    assert runner.default_max_steps(cell) == expected


def test_run_dir_name_stays_compatible_with_released_naming(cell):
    name = runner.run_dir_name(cell, evalsets=10, timestamp="20260824_105544")
    assert name == ("fake_img224_box64_step32_maxsteps36_seed42"
                    "_DefaultVisionAgent_hist1_evalsets10_20260824_105544")
    two = runner.run_dir_name(Cell(model="fake", digits=2), 10, "20260824_105544")
    assert two.startswith("multidigit_2_fake_img224")
    assert "_MultiDigitMemoryVisionAgent_" in two


def test_episode_writes_every_artefact(tmp_path, cell, monkeypatch):
    patch_backend(monkeypatch, ['{"action": "move", "direction": "right"}',
                                '{"action": "answer", "value": 0}'])
    spec = EpisodeSpec(0, (0,), (0,))
    out = runner.run_episode(spec, cell, FakeDataset(), str(tmp_path), max_steps=10)

    ep = tmp_path / "episode_0"
    assert (ep / "original.png").exists()
    assert (ep / "step_0.png").exists() and (ep / "step_2.png").exists()
    assert (ep / "trajectory.json").exists()
    assert json.loads((ep / "response_1.json").read_text())["usage"]["total_tokens"] == 5

    assert out["success"] is True
    assert out["termination_reason"] == "answered"
    assert out["n_moves"] == 1
    assert out["n_observations"] == 2
    assert out["error"] is None
    assert "stroke_coverage" in out and "stroke_coverage_readable" in out
    assert out["border_occlusion"] >= 0


def test_backend_failure_is_recorded_not_fatal(tmp_path, cell, monkeypatch):
    """A backend that exhausts its retries used to raise and kill the whole sweep."""
    patch_backend(monkeypatch, [BackendError("provider down")])
    out = runner.run_episode(EpisodeSpec(3, (3,), (3,)), cell, FakeDataset(),
                             str(tmp_path), max_steps=5)
    assert out["error"] is not None
    assert "BackendError" in out["error"]
    assert (tmp_path / "episode_3" / "trajectory.json").exists()


def test_step_limit_is_recorded_as_truncation(tmp_path, cell, monkeypatch):
    patch_backend(monkeypatch, ['{"action": "move", "direction": "right"}'])
    out = runner.run_episode(EpisodeSpec(1, (1,), (1,)), cell, FakeDataset(),
                             str(tmp_path), max_steps=4)
    assert out["termination_reason"] == "step_limit"
    assert out["answer"] is None
    assert out["success"] is False


def test_unparseable_output_is_recorded_as_invalid_action(tmp_path, cell, monkeypatch):
    patch_backend(monkeypatch, ["I would rather not say"])
    out = runner.run_episode(EpisodeSpec(2, (2,), (2,)), cell, FakeDataset(),
                             str(tmp_path), max_steps=6)
    assert out["termination_reason"] == "invalid_action"
    assert out["answer"] is None


def test_run_cell_writes_config_and_summary(tmp_path, cell, monkeypatch):
    patch_backend(monkeypatch, ['{"action": "answer", "value": 0}'])
    monkeypatch.setattr(runner, "load_mnist", lambda *a, **k: FakeDataset())
    out_root = runner.run_cell(cell, results_dir=str(tmp_path), evalsets=1,
                               workers=2, limit=4, with_control=False)

    config = json.loads(open(os.path.join(out_root, "run_config.json")).read())
    assert config["memory"] == "event_logging" and config["max_steps"] == 36

    summary = json.loads(open(os.path.join(out_root, "results_summary.json")).read())
    assert summary["metrics"]["total_episodes"] == 4
    assert summary["metrics"]["failed_episodes"] == 0
    assert summary["metrics"]["accuracy"] is not None
    assert "termination_reasons" in summary["metrics"]
    assert len(summary["episodes"]) == 4
    assert [e["episode_id"] for e in summary["episodes"]] == [0, 1, 2, 3]


def test_run_cell_is_discoverable_by_the_matrix(tmp_path, cell, monkeypatch):
    """A run must be found by run_config.json, not by parsing its directory name."""
    from mnist_pro.matrix import discover_runs
    patch_backend(monkeypatch, ['{"action": "answer", "value": 0}'])
    monkeypatch.setattr(runner, "load_mnist", lambda *a, **k: FakeDataset())
    runner.run_cell(cell, results_dir=str(tmp_path), evalsets=1, limit=2,
                    with_control=False)
    found = discover_runs(str(tmp_path))
    assert len(found) == 1
    assert found[0].source == "run_config.json"
    assert found[0].cell.memory == "event_logging"


def test_control_uses_the_same_agent_as_the_main_path(monkeypatch, cell):
    """The control used to be a separate function in each driver, free to drift."""
    patch_backend(monkeypatch, ['{"action": "answer", "value": 0}'])
    out = runner.run_control(EpisodeSpec(0, (0,), (0,)), cell, FakeDataset())
    assert out["control_prediction"] == "0"
    assert out["control_success"] is True
    assert out["raw_response"]


def test_no_border_option_changes_the_observations(tmp_path, cell, monkeypatch):
    patch_backend(monkeypatch, ['{"action": "answer", "value": 0}'])
    runner.run_episode(EpisodeSpec(0, (0,), (0,)), cell, FakeDataset(),
                       str(tmp_path / "with"), max_steps=5, draw_border=True)
    runner.run_episode(EpisodeSpec(0, (0,), (0,)), cell, FakeDataset(),
                       str(tmp_path / "without"), max_steps=5, draw_border=False)
    a = np.array(Image.open(tmp_path / "with" / "episode_0" / "step_0.png"))
    b = np.array(Image.open(tmp_path / "without" / "episode_0" / "step_0.png"))
    assert not np.array_equal(a, b)
