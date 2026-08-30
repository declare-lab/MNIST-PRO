"""The matrix must make an incomplete study visible.

The motivating case is real: across every released run there is no Textual Belief
State cell at an unbounded horizon. Unbounded horizon appears only with Event
Logging. A horizon comparison drawn from those logs therefore also swaps the memory
configuration, and the two effects cannot be separated.
"""

import json
import os

import pytest

from mnist_pro.matrix import (Cell, confounds, discover_runs, load_matrix,
                              parse_run_dir, status)

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "main_table.yaml")


def test_config_expands_to_cells():
    cells = load_matrix(CONFIG)
    assert len(cells) > 50
    assert all(isinstance(c, Cell) for c in cells)
    assert len({c.key() for c in cells}) == len(cells)   # no duplicates


def test_config_crosses_horizon_with_every_memory_config():
    """The gap that the released study has, and this config must not."""
    cells = load_matrix(CONFIG)
    turn_based = [c for c in cells if c.harness == "turn_based"]
    for memory in ("visual_buffer", "event_logging", "textual_belief_state",
                   "metric_grid_map"):
        horizons = {c.horizon for c in turn_based if c.memory == memory}
        assert {1, -1} <= horizons, f"{memory} is not declared at both horizons"


def test_confound_detector_catches_the_released_design():
    """Reproduce the released matrix and confirm it is flagged."""
    released = [
        Cell(model="gemini-3.7-flash", digits=d, memory="event_logging", horizon=-1)
        for d in (1, 2)
    ] + [
        Cell(model="gemini-3.7-flash", digits=d, memory=m, horizon=1)
        for d in (1, 2) for m in ("textual_belief_state", "metric_grid_map")
    ]
    warns = confounds(released)
    assert any("memory and horizon" in w for w in warns), warns


def test_confound_detector_is_quiet_on_a_crossed_design():
    crossed = [Cell(model="m", memory=mem, horizon=h)
               for mem in ("event_logging", "textual_belief_state")
               for h in (1, -1)]
    assert not [w for w in confounds(crossed) if "memory and horizon" in w]


@pytest.mark.parametrize("name,expected", [
    ("gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42"
     "_MemoryVisionAgent_hist1_evalsets10_20260824_110242",
     dict(model="gemini-3.7-flash", digits=1, memory="textual_belief_state", horizon=1)),
    ("multidigit_2_gemini-3.7-flash_img224_box64_step32_maxsteps78_seed42"
     "_MultiDigitDefaultVisionAgent_hist-1_evalsets10_20260824_111931",
     dict(model="gemini-3.7-flash", digits=2, memory="event_logging", horizon=-1)),
])
def test_legacy_directory_names_parse(name, expected):
    cell = parse_run_dir(name)
    assert cell is not None
    for k, v in expected.items():
        assert getattr(cell, k) == v


def test_unparseable_directory_name_is_ignored():
    assert parse_run_dir("some_random_folder") is None


def test_status_reports_present_and_missing(tmp_path):
    run = tmp_path / ("gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42"
                      "_MemoryVisionAgent_hist1_evalsets10_20260824_110242")
    run.mkdir()
    (run / "results_summary.json").write_text(json.dumps(
        {"metrics": {"accuracy": 0.54}, "episodes": [{"episode_id": 0}]}))

    matrix = [Cell(model="gemini-3.7-flash", memory="textual_belief_state", horizon=1),
              Cell(model="gemini-3.7-flash", memory="textual_belief_state", horizon=-1)]
    st = status(matrix, str(tmp_path))
    assert len(st["present"]) == 1
    assert len(st["missing"]) == 1
    assert st["missing"][0]["cell"].horizon == -1


def test_run_config_is_preferred_over_the_directory_name(tmp_path):
    run = tmp_path / "an_opaque_name"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps(
        {"model": "m", "digits": 2, "memory": "metric_grid_map", "horizon": -1,
         "harness": "mcp", "arm": "A2"}))
    (run / "results_summary.json").write_text(json.dumps({"metrics": {}, "episodes": []}))
    found = discover_runs(str(tmp_path))
    assert len(found) == 1
    assert found[0].source == "run_config.json"
    assert found[0].cell.harness == "mcp" and found[0].cell.arm == "A2"


def test_invalid_axis_values_are_rejected():
    with pytest.raises(ValueError, match="unknown harness"):
        Cell(model="m", harness="carrier-pigeon")
    with pytest.raises(ValueError, match="unknown arm"):
        Cell(model="m", arm="A9")
