"""Which harness drives an episode, and what each one needs to run.

Two families:

* **in-process** -- `turn_based` and `natural`. The framework holds the conversation
  and parses the model's JSON. Memory is fully controlled: the memory configuration
  and the horizon say exactly what the model may carry between turns.

* **tool-use** -- `mcp`, `antigravity`, `claude_code`, `deepseek`. The agent calls
  `move` / `view_image` / `submit` against the MCP server. `mcp` is driven by this
  package's own client; the rest are external runtimes that connect to the same
  server using the config `MCPEpisode` writes.

An external runtime brings its own context management, so what is held constant is
"model + harness", not "model + declared memory". `mnist_pro.matrix` keeps the two
families in separate cells; see docs/harness.md for why they are reported separately.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HarnessSpec:
    name: str
    family: str                     # "inprocess" | "tool_use"
    description: str
    controls_memory: bool           # is the memory axis meaningful for this harness?
    supports_arms: bool             # can it carry state between episodes?
    runtime: str | None = None      # executable an external runtime needs
    install_hint: str | None = None
    env_vars: tuple = ()

    def availability(self) -> dict:
        """Whether this harness can run here, and what is missing if not."""
        missing = []
        if self.runtime and shutil.which(self.runtime) is None:
            missing.append(f"executable {self.runtime!r} not on PATH")
        import os
        for var in self.env_vars:
            if not os.environ.get(var):
                missing.append(f"{var} is not set")
        return {"name": self.name, "available": not missing, "missing": missing,
                "install_hint": self.install_hint}


TURN_BASED = HarnessSpec(
    name="turn_based", family="inprocess", controls_memory=True, supports_arms=False,
    description="Framework-held loop; past actions re-rendered as text each turn. "
                "What the earlier published runs used.")

NATURAL = HarnessSpec(
    name="natural", family="inprocess", controls_memory=True, supports_arms=False,
    description="Framework-held multi-turn conversation; observations stay in their "
                "own turns and model outputs stay between them. The default.")

MCP = HarnessSpec(
    name="mcp", family="tool_use", controls_memory=False, supports_arms=True,
    description="This package's MCP client drives move/view_image/submit against "
                "ag_mcp_server.py. Runs and is tested without any external runtime.")

ANTIGRAVITY = HarnessSpec(
    name="antigravity", family="tool_use", controls_memory=False, supports_arms=True,
    runtime=None,
    description="The vendored Antigravity controllers, which produced the released "
                "six-arm results. Driven by mnist_pro/harness/native_controller.py.",
    install_hint="Needs Antigravity CCPA/BYOK credentials; see docs/harness.md.")

CLAUDE_CODE = HarnessSpec(
    name="claude_code", family="tool_use", controls_memory=False, supports_arms=True,
    runtime="claude",
    description="Claude Code as the agent runtime, pointed at the episode's MCP "
                "server with --mcp-config.",
    install_hint="npm install -g @anthropic-ai/claude-code")

DEEPSEEK = HarnessSpec(
    name="deepseek", family="tool_use", controls_memory=False, supports_arms=True,
    env_vars=("DEEPSEEK_API_KEY",),
    description="DeepSeek models driven through the built-in MCP client using "
                "OpenAI-compatible tool calling.",
    install_hint="export DEEPSEEK_API_KEY=...")

HARNESSES = {h.name: h for h in
             (TURN_BASED, NATURAL, MCP, ANTIGRAVITY, CLAUDE_CODE, DEEPSEEK)}

TOOL_USE = tuple(h.name for h in HARNESSES.values() if h.family == "tool_use")
IN_PROCESS = tuple(h.name for h in HARNESSES.values() if h.family == "inprocess")


def get(name: str) -> HarnessSpec:
    if name not in HARNESSES:
        raise ValueError(f"unknown harness {name!r}; expected one of "
                         f"{sorted(HARNESSES)}")
    return HARNESSES[name]


def availability_report() -> list[dict]:
    return [h.availability() for h in HARNESSES.values()]


def external_command(name: str, config_path, prompt: str) -> list[str]:
    """The command line that hands an episode to an external runtime.

    Returned rather than executed, so a caller can log it, dry-run it, or refuse.
    """
    spec = get(name)
    if spec.family != "tool_use":
        raise ValueError(f"{name} is in-process; it needs no external command")
    if name == "claude_code":
        return ["claude", "--mcp-config", str(config_path),
                "--allowed-tools", "mcp__activeglimpse__move,"
                                   "mcp__activeglimpse__view_image,"
                                   "mcp__activeglimpse__submit",
                "--print", prompt]
    raise ValueError(f"{name} has no generic external command; it is driven by its "
                     f"own controller (see mnist_pro/harness/)")
