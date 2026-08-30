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

**Turn mode.** `--turn-mode natural` is the **default**: the real conversation
transcript is kept, with observations interleaved in their own turns and the model's
outputs between them. It implies `--horizon -1`, since the transcript is complete by
construction. `--turn-mode turn_based --horizon 1` re-renders a textual summary of
past actions each turn instead, and is what the earlier published runs used.

**Harness.** How the agent is driven. `mnist-pro harness` lists what can run here.

| `--harness` | family | memory controlled | needs |
|---|---|---|---|
| `natural` (default) | in-process | yes | nothing |
| `turn_based` | in-process | yes | nothing |
| `mcp` | tool use | no | nothing — built-in MCP client |
| `antigravity` | tool use | no | Antigravity CCPA/BYOK credentials |
| `claude_code` | tool use | no | `claude` on PATH |
| `deepseek` | tool use | no | `DEEPSEEK_API_KEY` |

In-process harnesses parse the model's JSON. Tool-use harnesses give it `move`,
`view_image` and `submit` over MCP, so there is no response parsing to fail — and no
controlled memory, which is why those cells are reported separately.

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

The agent calls tools instead of emitting JSON:

```python
from mnist_pro.harness import MCPEpisode

with MCPEpisode(images, label="58", digits=2, arm="A2") as episode:
    episode.client.move("right")
    episode.client.submit("58")
    print(episode.audit())      # what was exposed, what was delivered
```

An external runtime is pointed at the same server using the config the episode writes:

```bash
claude --mcp-config "$EPISODE/mcp.json"        --allowed-tools mcp__activeglimpse__move,mcp__activeglimpse__view_image,mcp__activeglimpse__submit
```

Arms `A0`, `A1` and `A2` vary what carries *between* episodes: nothing, a persistent
`NOTES.md`, or notes plus a correctness receipt.

**Isolation.** The server holds no label, schedule or coordinates. It reaches the
environment only through an HMAC-signed file mailbox, observations are exposed under
opaque `secrets.token_hex(16)` names so filenames reveal no step index or ordering,
`view_image` is confined to a workspace containing nothing but observations, and the
mailbox and auth token live outside it. Only arm A2 is told whether a submission was
correct. `tests/test_harness_isolation.py` attacks each of these from the agent's
side — forged signatures, path traversal, symlinks, feedback entitlement.

Harness results belong in their own table — a harness supplies its own context
management, so memory is no longer the controlled variable. See
[docs/harness.md](docs/harness.md).

## Tests

```bash
pytest
```

164 tests, no provider contacted. The renderer is pinned byte-for-byte against real
observations from released runs, and — when MNIST and a log directory are present —
canvas construction and episode sampling are checked against them too, so a refactor
cannot silently change what is being measured. The MCP server is spawned as a real
subprocess and driven over stdio JSON-RPC, so the harness tests exercise the actual
protocol rather than a stand-in. `tests/make_golden.py` re-baselines the fixtures;
that is a deliberate act, not a side effect.

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
  harness/
    ag_mcp_server.py   stdio MCP server: move / view_image / submit
    protocol.py        controller side of the HMAC-signed episode mailbox
    mcp_client.py      minimal MCP stdio client
    session.py         MCPEpisode: env + controller + server together
    tool_agent.py      tool-calling loop for OpenAI-compatible providers
    registry.py        harness descriptors and availability
    *_controller.py    vendored Antigravity controllers
configs/          the declared evaluation matrix
docs/             harness and migration notes
```

Migrating from the pre-`mnist_pro` layout: [docs/migration.md](docs/migration.md).
