# MNIST-PRO

An evaluation framework for **active-glimpse visual reasoning**: an agent sees a
64×64 window onto a masked MNIST canvas, moves it around, and eventually commits to
an answer. The question the benchmark asks is not whether a model can read a digit —
it can, at 0.97–0.98 on the unmasked canvas — but whether it can *integrate what it
saw across turns*.

This is a restructuring of the original research code, driven by twelve specific
defects found while analysing the released logs. Each one is listed below with the
test that now pins it.

```bash
pip install -e ".[dev,data,models,analysis]"
export GEMINI_API_KEY=...          # read from the environment, never written to a file

mnist-pro run --model gemini-3.7-flash --digits 2 --memory textual_belief_state --horizon 1
mnist-pro matrix --config configs/main_table.yaml --results-dir main_table_logs
mnist-pro analyse --results-dir main_table_logs --csv results.csv
```

## What changed, and why

| # | Defect in the original | Fix | Pinned by |
|---|---|---|---|
| 1 | `MnistActiveVisionEnv` and `MultiDigitActiveVisionEnv` were **92% textually identical** (161 / 171 lines) | one `ActiveGlimpseEnv`; `digits` is a parameter | `test_env_api.py::test_single_env_class_handles_both_levels` |
| 2 | `evaluate.py` and `evaluate_multidigit.py` were **83% identical** (495 / 526 lines) and had drifted | one `runner.py` | — |
| 3 | **15 agent classes** differing mostly by prompt string | one `GlimpseAgent` + four `MemorySpec` values | `test_agents.py` (22 tests) |
| 4 | A malformed action **raised** and killed the run (9 × `raise ValueError`) | scored as `invalid_action`, episode terminates | `test_malformed_actions_are_scored_not_raised` |
| 5 | Forced answer, parse failure and a real answer of −1 **shared the sentinel `-1`** | `truncated` for the step limit; an explicit invalid action for parse failure; no sentinel | `test_step_limit_truncates_without_inventing_an_answer` |
| 6 | Termination reason recoverable only by matching the string `"Forced answer: maximum steps reached"` | `TerminationReason` enum in `info` | `test_env_api.py` |
| 7 | Visited `(x, y)` **never logged** — recoverable only by replaying an md5-seeded RNG | `TrajectoryRecorder` logs windows, latency and token usage | `test_recorder_captures_windows_actions_and_usage` |
| 8 | `tests/test_env_render.py` asserted nothing **and was stale** — it called `env.step('right')`, which raised | golden fixtures from released runs, compared byte-for-byte | `test_renderer_golden.py` (29 tests) |
| 9 | The coverage metric — the number in the exploration table — was **untested** | `metrics.py` with both definitions | `test_metrics.py` (10 tests) |
| 10 | Table scripts hardcoded `results_dir = "results"` and a literal model list, so `main_table_logs/` **could not be analysed by the repo's own code** | every entry point takes `--results-dir` | `test_no_hardcoded_results_path` |
| 11 | **No Textual Belief State run at unbounded horizon exists anywhere** — horizon and memory are perfectly confounded | declarative matrix reports missing cells and unsupported claims | `test_confound_detector_catches_the_released_design` |
| 12 | The cyan border is drawn **over** each glimpse, hiding its outer 2px, so published coverage overstates what was read by ~0.017 (L2) / 0.022 (L1) | both definitions computed and reported; `--no-border` removes the cause | `test_readable_coverage_never_exceeds_window_coverage` |

## Reproducibility

The renderer is pinned against real observations from the released runs. `pytest`
re-renders each fixture from its canvas and requires a **byte-for-byte** match,
including that the content-hashed start position lands where the original run
started. Re-baselining requires deliberately running `tests/make_golden.py`.

```
90 passed
```

## The environment

Gymnasium's API, so the wrappers do the bookkeeping the original had to remember:

```python
from mnist_pro import CanvasSpec, make_env

env = make_env(images, label="58", spec=CanvasSpec(digits=2), max_steps=78)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step({"action": "move", "direction": "right"})
print(info["termination_reason"], env.record.windows)
```

`reset(seed=...)` is new. The original start was fixed per image by a content hash,
so sensitivity to where the agent starts could not be measured at all.

## The memory taxonomy

| config | carried between turns |
|---|---|
| `visual_buffer` | images only, no text record |
| `event_logging` | prior actions |
| `textual_belief_state` | prior actions and thoughts |
| `metric_grid_map` | prior actions, thoughts and a structured spatial map |

`horizon` bounds the images retained (`-1` is unbounded). Prompt text is carried over
verbatim from the original classes and asserted in tests, so new runs stay comparable
with published ones.

## Tool-use harnesses

`mnist_pro/harness/` vendors the working Antigravity implementation: an MCP server
exposing `move` / `view_image` / `submit`, plus the controllers and preflight checks
that produced the released six-arm results. Arms `A0` / `A1` / `A2` are a
cross-episode axis — nothing carried, persistent notes, notes plus correctness
feedback.

**Harness results belong in their own table.** A tool-use harness brings its own
scaffolding, so "memory" is no longer controlled, and merging those numbers into the
memory-taxonomy table would void the axis that table is built on. See
[docs/harness.md](docs/harness.md).

## Layout

```
mnist_pro/
  env.py          ActiveGlimpseEnv, TerminationReason
  rendering.py    canvas construction and observation rendering
  wrappers.py     TimeLimit, TrajectoryRecorder
  metrics.py      coverage, both definitions
  agents/         one agent, four declarative memory specs
  backends.py     Gemini / OpenAI / OpenRouter / Vertex, with usage accounting
  matrix.py       declarative run matrix, gap and confound reporting
  runner.py       the single evaluation driver
  analysis.py     result loading and tables, for any results directory
  harness/        vendored MCP + Antigravity controllers
configs/main_table.yaml
tests/            90 tests, golden fixtures from released runs
```
