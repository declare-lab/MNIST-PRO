# Tool-use harnesses, and why their numbers live in their own table

The default evaluation is turn-based: the agent is shown an observation and emits a
JSON action, which the framework parses. `mnist_pro.harness` adds a second mode where
the agent calls tools instead — `move`, `view_image`, `submit` — over MCP.

## Running one

```python
from mnist_pro.harness import MCPEpisode

with MCPEpisode(images, label="58", digits=2, arm="A2") as episode:
    episode.client.move("right")
    episode.client.submit("58")
    print(episode.audit())
```

`mnist-pro harness` lists every harness and whether it can run on this machine.
`mcp` needs nothing external; `claude_code` needs `claude` on PATH; `deepseek` needs
`DEEPSEEK_API_KEY`; `antigravity` needs CCPA/BYOK credentials.

## What the MCP server does and does not know

`ag_mcp_server.py` is a stdio JSON-RPC server. It holds **no label, no schedule and no
coordinate state**. It relays opaque requests to the controller over an
HMAC-authenticated file mailbox and returns the resulting masked canvas as native MCP
`ImageContent`, plus an opaque absolute path to the same PNG. Correctness is never
derived server-side, so an agent cannot read the answer out of the harness.

Tools are annotated (`readOnlyHint`, `destructiveHint`, `idempotentHint`,
`openWorldHint`), the protocol version is negotiated, and image payloads are checked
for the PNG signature and capped at 4 MB.

## Isolation constraints

These are the constraints the original Codex/Antigravity setup enforced, and every
one is tested from the agent's side in `tests/test_harness_isolation.py`:

| constraint | why | enforced by |
|---|---|---|
| the server never receives the label, schedule or coordinates | the answer must only be obtainable from pixels | `EpisodeController` holds them; only four env vars cross the boundary |
| requests are HMAC-SHA256 signed over a canonical serialisation | a forged or replayed request must not reach the environment | `protocol.verify`, constant-time |
| observations use opaque `secrets.token_hex(16)` names | sequential names would reveal step index, count and ordering | `EpisodeController._write_observation` |
| `view_image` is confined to the workspace | no reading the unmasked canvas, the trajectory, or anything else | server-side path validation: absolute, in-tree, no traversal, `O_NOFOLLOW` |
| the workspace contains only observations | the mailbox and auth token must not be enumerable by the agent | `MCPEpisode` puts them beside the workspace, not inside |
| only arm A2 learns correctness | A0 and A1 must not distinguish a right answer from a wrong one | `EpisodeController._submit` |
| every exposure is recorded by digest | a run can be audited for what was actually made readable | `EpisodeController.audit()` |

`audit()` cross-checks what the controller exposed against what the server reported
delivering. Any delivery whose digest was never exposed is a protocol violation.

## How Gemini is driven

Gemini does not go through a generic tool-calling loop. It goes through the
**Antigravity managed agent**, which is how the released six-arm results were
produced:

```
controller  ──POST interaction──▶  Antigravity agent (gemini-3.7-flash)
            ◀── pending calls ───
            ── execute locally ──▶  MCP server ──▶ mailbox ──▶ environment
            ──POST results───────▶  next turn
```

The controller never runs the model. It posts an interaction, receives the function
calls the agent wants to make, executes them, and returns the results in the next
turn. The agent's environment has networking disabled, and for arms A1 and A2 the
persistent notes are injected as an inline source at `/workspace/NOTES.md`.

Executing those calls through the MCP server rather than against the environment
directly means the isolation guarantees below apply to this path too, not just to the
in-process MCP client.

`mnist_pro/harness/antigravity.py` carries the tool names, the per-level answer
pattern, the `path_or_id` argument and the interaction body over from the vendored
controller, so runs stay comparable with the released results.

## Arms

Arms are a **cross-episode** axis, orthogonal to the within-episode memory taxonomy:

| arm | carried between episodes |
|---|---|
| A0 | nothing |
| A1 | a persistent `NOTES.md` |
| A2 | `NOTES.md`, plus a correctness receipt on submission |

A2's receipt makes the suite a learning experiment rather than a static benchmark:
the agent is told whether each answer was right and may revise its notes.

## The reporting rule

**Do not merge harness results into the memory-taxonomy table.**

The contribution of that table is that *memory is controlled*: Visual Buffer, Event
Logging, Textual Belief State and Metric Grid Map differ in exactly what the agent is
allowed to carry between turns, and the horizon `H` bounds the image history
precisely. A tool-use harness supplies its own scaffolding — its own context
management, its own retry behaviour, its own notion of what to keep. What is held
constant is no longer the same thing.

So the two tracks measure different quantities:

* **turn-based / natural** — *model + declared memory configuration*
* **mcp / native** — *model + harness*, memory uncontrolled

Both are worth reporting. Reporting them in one table would silently void the axis
the first one is built on.

`mnist-pro matrix` treats `harness` and `arm` as declared axes, so a cell always
records which track produced it, and `Cell.key()` keeps the tracks separate by
construction.

## Reproducing the released suite

The frozen controller snapshot, the shuffled episode schedules and the notes
instructions used by the reported run are stored alongside the outputs, under
`controller_assets/`. `protocol_preflight.py` performs the isolation and protocol
checks that must pass before a suite launches.
