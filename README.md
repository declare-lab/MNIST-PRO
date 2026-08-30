# MNIST-PRO

An evaluation framework for **active-glimpse visual reasoning**. An agent sees a small
window onto a masked MNIST canvas, moves it around one step at a time, and eventually
commits to an answer. The benchmark measures whether a model can integrate what it saw
across turns — not whether it can read a digit, which it can do at 0.97–0.98 on the
unmasked canvas.

## Install

```bash
pip install -e ".[dev,data,models,analysis]"
```

Extras: `data` pulls torch/torchvision for MNIST, `models` the provider SDKs,
`analysis` matplotlib/pandas, `dev` pytest. The core package needs only numpy,
Pillow, PyYAML and gymnasium.

API keys are read from the environment and never written to any file:

```bash
export GEMINI_API_KEY=...        # also OPENAI_API_KEY, OPENROUTER_API_KEY
```

## Quickstart

Evaluate one condition:

```bash
mnist-pro run --model gemini-3.7-flash --digits 2 --memory textual_belief_state --horizon 1
```

Summarise any directory of runs:

```bash
mnist-pro analyse --results-dir results --csv results.csv
```

Check the study against its declared design:

```bash
mnist-pro matrix --config configs/main_table.yaml --results-dir results
```

## Concepts

**Levels.** `--digits 1` is a 224×224 canvas with one digit; `--digits 2` is 448×224
with two, answered as a left-to-right string such as `"58"`. Any digit count works —
level is a parameter, not a fork.

**Glimpse geometry.** `--box-size` is the window side (default 64) and `--step-size`
how far one move travels (default 32). Because the step is half the box, consecutive
glimpses overlap.

**Memory configurations** — what the agent carries between turns:

| `--memory` | carried |
|---|---|
| `visual_buffer` | images only, no textual record of its own actions |
| `event_logging` | its prior actions |
| `textual_belief_state` | prior actions and thoughts |
| `metric_grid_map` | prior actions, thoughts, and a structured spatial map |

**Horizon.** `--horizon` bounds how many past images stay in context; `-1` is
unbounded. Memory configuration and horizon are independent axes.

**Turn mode.** `--turn-mode turn_based` (default) re-renders a textual memory each
turn. `--turn-mode natural` keeps the real conversation transcript, with observations
interleaved in their own turns; it requires `--horizon -1`.

## Python API

The environment follows Gymnasium's interface:

```python
from mnist_pro import CanvasSpec, make_env

env = make_env(images, label="58", spec=CanvasSpec(digits=2), max_steps=78)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step({"action": "move", "direction": "right"})
obs, reward, terminated, truncated, info = env.step({"action": "answer", "value": "58"})

print(info["termination_reason"])   # answered | step_limit | invalid_action
print(env.record.windows)           # every window position the agent was shown
print(env.record.usage_totals)      # token counts for the episode
```

Actions are `{"action": "move", "direction": ...}` or `{"action": "answer", "value": ...}`,
accepted as a dict or a JSON string. A malformed action ends the episode as
`invalid_action` rather than raising, so a bad model output costs an episode, not the run.

`reset(seed=...)` varies the start position. With no seed the start is derived from a
hash of the canvas, which is what the released runs used.

Wrappers do the bookkeeping:

```python
from mnist_pro import ActiveGlimpseEnv, TimeLimit, TrajectoryRecorder

env = TrajectoryRecorder(TimeLimit(ActiveGlimpseEnv(images, label), max_steps=36))
```

`TimeLimit` reports the step limit through `truncated`, without inventing an answer.
`TrajectoryRecorder` logs window positions, actions, rewards, per-step latency and
token usage; `env.record.to_dict()` is what lands in `trajectory.json`.

Agents:

```python
from mnist_pro import AgentConfig, GlimpseAgent
from mnist_pro.backends import get_backend

agent = GlimpseAgent(get_backend("gemini-3.7-flash"),
                     AgentConfig(memory="textual_belief_state", digits=2, horizon=1))
action, raw_response, usage = agent.act(obs)
raw, prediction, usage = agent.predict_full_image(canvas_image)   # unmasked control
```

## Coverage metrics

`stroke_coverage` has two definitions, and runs report both:

```python
from mnist_pro.metrics import exploration_stats
exploration_stats(canvas, windows, spec)
# stroke_coverage           union of the full glimpse windows
# stroke_coverage_readable  the same, minus the 2px ring the window outline covers
# border_occlusion          the difference
```

The outline is drawn *over* each glimpse, so the outer two pixels were never legible.
The gap is about 0.017 at two digits and 0.022 at one. Pass `--no-border` to remove the
cause entirely, at the price of comparability with runs that had it.

## Run outputs

```
results/<run-name>/
  run_config.json         the exact condition; nothing is recovered by parsing a name
  results_summary.json    metrics and one entry per episode
  episode_<i>/
    original.png          the unmasked canvas
    step_<n>.png          every observation the agent was shown
    trajectory.json       windows, actions, rewards, latency, usage, termination reason
    response_<n>.json     the raw model response, saved before anything parses it
```

## The evaluation matrix

`configs/main_table.yaml` declares the conditions the study intends to cover, as a
cross-product. `mnist-pro matrix` reports which exist, which are missing, and which
axes the config claims to vary independently but does not:

```
declared cells: 168    runs found: 51
PRESENT (51) ... MISSING (117)
CONFOUNDS
  ! harness and arm are not fully crossed: 8 of 12 combinations declared
```

`expect_crossed:` lists the axis pairs that must be fully crossed, so a warning means
a claim is unsupported rather than that some ablation is deliberately partial.

Directory names from earlier versions are parsed automatically, so existing logs can
be analysed without conversion.

## Tool-use harnesses

`mnist_pro/harness/` holds the MCP server and Antigravity controllers, where the agent
calls `move` / `view_image` / `submit` as tools instead of emitting JSON. Arms `A0`,
`A1` and `A2` vary what carries *between* episodes: nothing, a persistent `NOTES.md`,
or notes plus a correctness receipt.

Harness results belong in their own table — a harness supplies its own context
management, so memory is no longer the controlled variable. See
[docs/harness.md](docs/harness.md).

## Tests

```bash
pytest
```

94 tests. The renderer is pinned byte-for-byte against real observations from released
runs, and — when MNIST and a log directory are present — canvas construction and
episode sampling are checked against them too, so a refactor cannot silently change
what is being measured. `tests/make_golden.py` re-baselines the fixtures; that is a
deliberate act, not a side effect.

## Layout

```
mnist_pro/
  env.py          ActiveGlimpseEnv, TerminationReason
  rendering.py    canvas construction and observation rendering
  wrappers.py     TimeLimit, TrajectoryRecorder
  metrics.py      coverage and exploration statistics
  agents/         one agent, four declarative memory specs
  backends.py     Gemini, OpenAI, OpenRouter, Vertex
  dataset.py      deterministic episode sampling
  matrix.py       declarative run matrix
  runner.py       the evaluation driver
  analysis.py     result loading and tables
  harness/        MCP server and tool-use controllers
configs/          the declared evaluation matrix
docs/             harness and migration notes
```

Migrating from the pre-`mnist_pro` layout: [docs/migration.md](docs/migration.md).
