# Tool-use harnesses, and why their numbers live in their own table

The default evaluation is turn-based: the agent is shown an observation and emits a
JSON action, which the framework parses. `mnist_pro.harness` adds a second mode where
the agent calls tools instead — `move`, `view_image`, `submit` — over MCP.

## What the MCP server does and does not know

`ag_mcp_server.py` is a stdio JSON-RPC server. It holds **no label, no schedule and no
coordinate state**. It relays opaque requests to the controller over an
HMAC-authenticated file mailbox and returns the resulting masked canvas as native MCP
`ImageContent`, plus an opaque absolute path to the same PNG. Correctness is never
derived server-side, so an agent cannot read the answer out of the harness.

Tools are annotated (`readOnlyHint`, `destructiveHint`, `idempotentHint`,
`openWorldHint`), the protocol version is negotiated, and image payloads are checked
for the PNG signature and capped at 4 MB.

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
