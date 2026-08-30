"""Tool-use harnesses, where the agent calls tools instead of emitting JSON text.

These are vendored from the working Antigravity implementation rather than rewritten;
they produced the released six-arm results and are the reproducible artefact.

    ag_mcp_server.py        stdio MCP server exposing move / view_image / submit.
                            Holds no label, schedule or coordinate state: it relays
                            opaque requests over an HMAC-authenticated file mailbox,
                            so the agent cannot read the answer out of the harness.
    native_controller.py    local CCPA/BYOK episode controller (native delivery)
    six_arm_controller.py   the six-arm controller (3 arms x 2 levels)
    three_arm_controller.py the Terra three-arm controller
    launch_suite.py         suite launcher
    protocol_preflight.py   isolation and protocol checks, run before a suite

Arms are a *cross-episode* axis and are orthogonal to the within-episode memory
taxonomy in `mnist_pro.agents`:

    A0  nothing carried between episodes
    A1  a persistent NOTES.md is carried forward
    A2  A1, plus a correctness receipt on submission

Read `docs/harness.md` before merging harness numbers into the main table: a
tool-use harness supplies its own scaffolding, so what is held constant differs from
the turn-based arm.
"""

HARNESS_TOOLS = ("move", "view_image", "submit")
ARMS = ("A0", "A1", "A2")
