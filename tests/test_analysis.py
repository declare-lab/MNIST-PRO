"""Analysis must work against any results directory.

The original table scripts opened a hardcoded `results/` and iterated a literal list
of model names, so the released `main_table_logs/` could not be analysed by the
repository's own code.
"""

import csv
import json
import os

from mnist_pro.analysis import latest_per_cell, load_results, main_table, to_csv


def write_run(root, name, metrics, episodes=1, config=None):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    if not config:
        digits = 2 if ("multidigit_2" in name or "MultiDigit" in name) else 1
        memory = "image_only_baseline"
        if "MemoryVisionAgent" in name:
            memory = "textual_state"
        elif "SpatialMemoryVisionAgent" in name:
            memory = "metric_grid_map"
        model = "gemini-3.7-flash" if "gemini" in name else "m"
        horizon = 1 if "_hist1" in name else -1
        config = {
            "model": model,
            "digits": digits,
            "memory": memory,
            "horizon": horizon,
            "turn_mode": "natural" if "Natural" in name else "turn_based",
            "harness": "natural" if "Natural" in name else "turn_based",
            "arm": "A0",
            "box_size": 64,
            "step_size": 32,
            "image_size": 224,
            "seed": 42
        }
    with open(os.path.join(d, "run_config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(d, "results_summary.json"), "w") as f:
        json.dump({"metrics": metrics,
                   "episodes": [{"episode_id": i} for i in range(episodes)]}, f)
    return d


def test_load_results_takes_any_directory(tmp_path):
    root = tmp_path / "anywhere_at_all"
    root.mkdir()
    write_run(str(root), "gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42"
                         "_MemoryVisionAgent_hist1_evalsets10_20260824_110242",
              {"accuracy": 0.54, "average_steps": 7.92})
    results = load_results(str(root))
    assert len(results) == 1
    assert results[0].accuracy == 0.54
    assert results[0].cell.memory == "textual_state"


def test_no_hardcoded_results_path():
    """No module-level constant pins a directory or a model list.

    Checked on the parsed syntax tree rather than the raw text, so prose describing
    the old behaviour does not trip the assertion.
    """
    import ast
    import inspect

    from mnist_pro import analysis, matrix
    for module in (analysis, matrix):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not names:
                continue
            if isinstance(node.value, ast.Constant) and node.value.value == "results":
                raise AssertionError(
                    f"{module.__name__}: {names} pins a results directory")
            if any(n.lower() in {"models", "model_list"} for n in names):
                raise AssertionError(f"{module.__name__}: {names} pins a model list")


def test_main_table_separates_levels(tmp_path):
    root = str(tmp_path)
    write_run(root, "gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42"
                    "_MemoryVisionAgent_hist1_evalsets10_20260824_110242",
              {"accuracy": 0.54, "average_steps": 7.92})
    write_run(root, "multidigit_2_gemini-3.7-flash_img224_box64_step32_maxsteps78"
                    "_seed42_MultiDigitMemoryVisionAgent_hist1_evalsets10_20260824_114120",
              {"accuracy": 0.06, "average_steps": 34.84})
    results = load_results(root)
    assert [r["accuracy"] for r in main_table(results, digits=1)] == [0.54]
    assert [r["accuracy"] for r in main_table(results, digits=2)] == [0.06]


def test_csv_includes_both_coverage_definitions(tmp_path):
    root = str(tmp_path)
    write_run(root, "opaque", {"accuracy": 0.5, "average_stroke_coverage": 0.61,
                               "average_stroke_coverage_readable": 0.59},
              config={"model": "m", "digits": 1, "memory": "image_only_baseline",
                      "horizon": -1})
    out = os.path.join(root, "out.csv")
    to_csv(load_results(root), out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["average_stroke_coverage"] == "0.61"
    assert rows[0]["average_stroke_coverage_readable"] == "0.59"


def test_latest_per_cell_deduplicates(tmp_path):
    root = str(tmp_path)
    base = ("gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42"
            "_MemoryVisionAgent_hist1_evalsets10_")
    a = write_run(root, base + "20260824_110242", {"accuracy": 0.1})
    b = write_run(root, base + "20260825_110242", {"accuracy": 0.9})
    os.utime(b, (2_000_000_000, 2_000_000_000))
    best = latest_per_cell(load_results(root))
    assert len(best) == 1
    assert list(best.values())[0].accuracy == 0.9


def test_missing_summary_is_skipped(tmp_path):
    (tmp_path / "gemini-3.7-flash_img224_box64_step32_maxsteps36_seed42"
                "_MemoryVisionAgent_hist1_evalsets10_20260824_110242").mkdir()
    assert load_results(str(tmp_path)) == []
