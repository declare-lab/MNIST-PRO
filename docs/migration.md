# Migrating from the pre-`mnist_pro` layout

The framework is a restructuring of the original research code. Behaviour is
unchanged where it was correct; this note records what moved, what changed, and how
existing runs and logs are still usable.

## Existing logs still work

Run directories produced by the original drivers are parsed automatically, so
`mnist-pro analyse` and `mnist-pro matrix` read them without conversion:

```bash
mnist-pro analyse --results-dir main_table_logs
```

New runs additionally write `run_config.json`, which is preferred over the directory
name when present.

Episode ids are stable. `dataset.sample_balanced` reproduces the original sampling
exactly — verified against released runs at both levels, indices and labels — so
episode `i` still refers to the same MNIST images.

## Where things moved

| was | now |
|---|---|
| `src/environment.py` `MnistActiveVisionEnv`, `MultiDigitActiveVisionEnv` | `mnist_pro.env.ActiveGlimpseEnv`, with `digits` |
| `src/agent.py` 15 agent classes | `mnist_pro.agents.GlimpseAgent` + `MEMORY_SPECS` |
| `src/agent.py` backends | `mnist_pro.backends` |
| `scripts/evaluate.py`, `scripts/evaluate_multidigit.py` | `mnist_pro.runner` / `mnist-pro run` |
| `analysis/generate_table*.py` | `mnist_pro.analysis` / `mnist-pro analyse` |

Old class names resolve to configurations:

```python
from mnist_pro.agents import from_legacy_class
from_legacy_class("MultiDigitMemoryVisionAgent")
# AgentConfig(memory='textual_belief_state', digits=2, horizon=1, turn_mode='turn_based')
```

## Behaviour that deliberately changed

**Malformed actions are scored, not raised.** `step()` previously raised `ValueError`
on unparseable JSON, an unknown direction or an unknown action type, so one bad model
output aborted the process. The episode now terminates with
`termination_reason == "invalid_action"` and the sweep continues.

**The step limit no longer fabricates an answer.** The driver used to inject
`{"action": "answer", "value": -1}`, which was indistinguishable in the logs from the
agent's JSON-parse fallback (also `-1`) and from a genuine model answer of `-1`.
Truncation is now reported through `truncated` and carries no answer.

**Parse failure is explicit.** After `max_attempts`, the agent returns
`{"action": "invalid", "error": "JSONDecodeError"}` rather than an answer.

**A failed episode no longer kills the run.** Backends raise `BackendError` after
exhausting retries; the runner records the episode with its error and continues.

**Coverage is reported two ways.** The window outline is drawn over each glimpse and
hides its outer two pixels, so the original single figure counted pixels the agent
could not read. Both `stroke_coverage` and `stroke_coverage_readable` are now
recorded, along with their difference. The default still draws the outline, so
published figures remain comparable; `--no-border` removes the cause.

## What is now recorded that was not

* window positions per observation — previously recoverable only by replaying an
  md5-seeded RNG over the canvas bytes
* termination reason as an enum — previously recovered by matching the string
  `"Forced answer: maximum steps reached"`
* per-step latency and token usage
* the raw model response per step, written before anything parses it

## API differences to watch

* `step()` returns the Gymnasium five-tuple `(obs, reward, terminated, truncated,
  info)`; the original returned four, with `done` conflating both endings.
* `reset()` returns `(obs, info)`; the original returned only `obs`.
* `agent.act()` returns `(action_dict, raw_response, usage)`; the original returned
  `(action_json_string, raw_response)`.
* `predict_full_image()` returns `(raw, prediction, usage)`; the original returned
  `(raw, prediction)`.

## Study design

`configs/main_table.yaml` declares the intended conditions. Running `mnist-pro matrix`
against the released logs shows 51 of 168 declared cells present. Notably absent is
any `textual_belief_state` run at `horizon: -1` — unbounded horizon appears only with
`event_logging` — so in those logs a horizon comparison also changes the memory
configuration. Closing that gap requires new runs, not a code change.
