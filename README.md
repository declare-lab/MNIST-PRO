# MNIST-PRO

An evaluation framework for active-glimpse visual reasoning.

An agent sees a small window onto a masked MNIST canvas, moves it one step at a time,
and eventually commits to an answer. The benchmark is not about reading a digit —
models do that at 0.97–0.98 on the unmasked canvas. It is about whether a model can
hold together what it saw across many turns.

## Install

```bash
pip install -e ".[dev,data,models,analysis]"
```

The core package needs only numpy, Pillow, PyYAML and gymnasium. The extras add
torchvision for MNIST (`data`), the provider SDKs (`models`), matplotlib and pandas
(`analysis`), and pytest (`dev`).

Credentials are read from the environment and never written to a file:

```bash
export GEMINI_API_KEY=...
```

## Quickstart

Run one condition:

```bash
mnist-pro run --model gemini-3.7-flash --digits 2
```

Summarise a directory of runs:

```bash
mnist-pro analyse --results-dir results --csv results.csv
```

See which harnesses can run on this machine:

```bash
mnist-pro harness
```

Check a study against its declared design:

```bash
mnist-pro matrix --config configs/main_table.yaml --results-dir results
```

## How an episode works

The canvas is one or more MNIST digits, upscaled and binarised to black strokes on
white. Everything outside the glimpse window is masked dark gray. The agent moves the
window and then answers.

`--digits 1` gives a 224×224 canvas answered with a single digit. `--digits 2` gives
448×224 answered as a left-to-right string such as `"58"`. Any digit count works.

`--box-size` is the window side, 64 by default, and `--step-size` is how far one move
travels, 32 by default. Because the step is half the window, consecutive glimpses
overlap.

## Memory configurations

What the agent is allowed to carry from one turn to the next. This is the axis the
benchmark's main table is built on.

- `visual_buffer` — past images only, with no written record of its own actions
- `event_logging` — its past actions
- `textual_belief_state` — its past actions and thoughts
- `metric_grid_map` — its past actions, thoughts, and a structured spatial map

`--horizon` bounds how many past images stay in context, and `-1` means unbounded.
Memory configuration and horizon vary independently.

## Turn modes

`--turn-mode natural` is the default. The real conversation transcript is kept:
observations stay in their own turns and the model's replies stay between them. It
implies `--horizon -1`, because the transcript is complete by construction.

`--turn-mode turn_based --horizon 1` re-renders a written summary of past actions each
turn instead. This is what the earlier published runs used, and is how to reproduce
them.

## Harnesses

A harness is how the agent is driven. There are two families.

**In-process.** `natural` and `turn_based`. The framework holds the conversation and
parses the model's JSON replies. Memory is fully controlled: the memory configuration
and the horizon say exactly what the model may carry between turns.

**Tool use.** `mcp`, `antigravity`, `claude_code` and `deepseek`. The agent calls
`move`, `view_image` and `submit` over MCP. There is no reply to parse, so parsing
failures disappear as a category — but the runtime brings its own context management,
so memory is no longer controlled. Those results are reported separately; see
[docs/harness.md](docs/harness.md).

`mnist-pro harness` prints what each one needs and whether it can run here. `mcp`
needs nothing external. `antigravity` and `deepseek` need a credential. `claude_code`
needs the `claude` binary on PATH.

Gemini is driven through the Antigravity managed agent, which is how the released
six-arm results were produced. The controller posts an interaction, the agent replies
with pending tool calls, and the controller executes them through the MCP server.

## Cross-episode arms

Arms decide what carries between episodes, and apply to the tool-use harnesses.

- `A0` — nothing carries over
- `A1` — a persistent `NOTES.md` carries over
- `A2` — notes carry over, and each submission returns a correctness receipt

A2 makes the suite a learning experiment rather than a static benchmark.

## Python API

The environment follows Gymnasium's interface.

```python
from mnist_pro import CanvasSpec, make_env

env = make_env(images, label="58", spec=CanvasSpec(digits=2), max_steps=78)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step({"action": "move", "direction": "right"})
obs, reward, terminated, truncated, info = env.step({"action": "answer", "value": "58"})

info["termination_reason"]   # answered, step_limit, or invalid_action
env.record.windows           # every window position the agent was shown
env.record.usage_totals      # token counts for the episode
```

Actions are `{"action": "move", "direction": ...}` or `{"action": "answer", "value": ...}`,
given as a dict or a JSON string. A malformed action ends the episode as
`invalid_action` rather than raising, so a bad model reply costs one episode, not the
whole run.

`reset(seed=...)` varies the start position. Without a seed the start comes from a
hash of the canvas, which is what the released runs used.

Wrappers do the bookkeeping. `TimeLimit` reports the step limit through `truncated`
without inventing an answer, and `TrajectoryRecorder` logs window positions, actions,
rewards, latency and token usage.

```python
from mnist_pro import ActiveGlimpseEnv, TimeLimit, TrajectoryRecorder

env = TrajectoryRecorder(TimeLimit(ActiveGlimpseEnv(images, label), max_steps=36))
```

Agents take a memory configuration rather than a subclass.

```python
from mnist_pro import AgentConfig, GlimpseAgent
from mnist_pro.backends import get_backend

agent = GlimpseAgent(get_backend("gemini-3.7-flash"),
                     AgentConfig(memory="textual_belief_state", digits=2))
action, raw_response, usage = agent.act(obs)
```

An MCP episode exposes the three tools and audits what it revealed.

```python
from mnist_pro.harness import MCPEpisode

with MCPEpisode(images, label="58", digits=2, arm="A2") as episode:
    episode.client.move("right")
    episode.client.submit("58")
    episode.audit()
```

## Coverage metrics

Every run reports stroke coverage two ways. `stroke_coverage` is the union of the full
glimpse windows. `stroke_coverage_readable` excludes the two-pixel ring that the window
outline is drawn over, which the agent could not actually read. The difference is about
0.017 at two digits and 0.022 at one, and is reported as `border_occlusion`.

The outline is on by default so figures stay comparable with earlier runs. `--no-border`
removes it, and then the two definitions coincide.

## Run outputs

```
results/<run-name>/
  run_config.json         the exact condition, so nothing is recovered from a path
  results_summary.json    metrics and one entry per episode
  episode_<i>/
    original.png          the unmasked canvas
    step_<n>.png          every observation the agent was shown
    trajectory.json       windows, actions, rewards, latency, usage, termination
    response_<n>.json     the raw model reply, saved before anything parses it
```

## The evaluation matrix

`configs/main_table.yaml` declares the conditions a study intends to cover, as a
cross-product. `mnist-pro matrix` reports which exist, which are missing, and which
axes the config claims to vary independently but does not.

```
declared cells: 216    runs found: 51
PRESENT (51)  MISSING (165)
```

Listing an axis pair under `expect_crossed:` means a warning only appears when a claim
is unsupported, rather than whenever an ablation is deliberately partial.

Directory names from earlier versions are parsed automatically, so existing logs can be
analysed without conversion.

## Tests

```bash
pytest
```

181 tests, and no provider is contacted. Model backends are faked, and the MCP server
is spawned as a real subprocess and driven over stdio JSON-RPC, so the harness tests
exercise the actual protocol.

The renderer is pinned byte-for-byte against real observations from released runs. When
MNIST and a log directory are present, canvas construction and episode sampling are
checked against them too, so a refactor cannot silently change what is measured.
`tests/make_golden.py` re-baselines the fixtures, which is a deliberate act rather than
a side effect.

## Layout

```
mnist_pro/
  env.py          ActiveGlimpseEnv, TerminationReason
  rendering.py    canvas construction and observation rendering
  wrappers.py     TimeLimit, TrajectoryRecorder
  metrics.py      coverage and exploration statistics
  agents/         one agent, four declarative memory configurations
  backends.py     Gemini, OpenAI, OpenRouter, Vertex, DeepSeek
  dataset.py      deterministic episode sampling
  matrix.py       the declarative run matrix
  runner.py       the evaluation driver
  analysis.py     result loading and tables
  harness/        MCP server, episode mailbox, and the tool-use drivers
configs/          declared evaluation matrices
docs/             harness and migration notes
tests/            181 tests and golden fixtures
```

Migrating from the older layout: [docs/migration.md](docs/migration.md).
