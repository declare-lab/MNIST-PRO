"""An MCP episode: environment, mailbox controller and server subprocess together.

    with MCPEpisode(images, label="58", digits=2) as episode:
        episode.client.move("right")
        episode.client.submit("58")
        print(episode.result())

The agent -- built in, or an external runtime pointed at `episode.config_path` --
only ever sees the three tools. The label lives in the controller, one process away.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..rendering import CanvasSpec
from ..wrappers import make_env
from .mcp_client import SERVER_PATH, MCPStdioClient
from .protocol import EpisodeController, Mailbox


def write_mcp_config(path: Path, env: dict, server_path: Path = SERVER_PATH,
                     python: str | None = None) -> Path:
    """Emit the `.mcp.json` an external agent runtime consumes.

    Claude Code reads this with `--mcp-config`; Antigravity and other MCP clients
    take the same shape. The env block carries the mailbox location and token, so the
    server can reach this episode's controller and no other.
    """
    config = {"mcpServers": {"activeglimpse": {
        "command": python or sys.executable,
        "args": [str(server_path)],
        "env": dict(env)}}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path


@dataclass
class EpisodeResult:
    answer: object
    success: bool | None
    moves: int
    steps: int
    termination_reason: str
    rejected_requests: int
    windows: list
    observations: list


class MCPEpisode:
    """One episode exposed over MCP."""

    def __init__(self, images, label, digits: int = 1, spec: CanvasSpec | None = None,
                 max_steps: int | None = None, arm: str = "A0",
                 workdir: str | Path | None = None, draw_border: bool = True,
                 start_client: bool = True):
        self.spec = spec or CanvasSpec(digits=digits)
        self.max_steps = max_steps
        self.arm = arm
        self.start_client = start_client
        self._tmp = None
        if workdir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="mnist-pro-mcp-")
            workdir = self._tmp.name
        # Resolved because the server compares observation paths against the store
        # root textually. On macOS /var is a symlink to /private/var, so an
        # unresolved root and a resolved path would never share a common prefix.
        self.workdir = Path(workdir).resolve()
        # Only `workspace/` is reachable by the agent. The mailbox and the config
        # holding the auth token live beside it, not inside it, so nothing the agent
        # can enumerate exposes the channel it is talking over.
        self.workspace = self.workdir / "workspace"
        self.observation_dir = self.workspace / "observations"
        self.observation_dir.mkdir(parents=True, exist_ok=True)

        self.env = make_env(images, label, spec=self.spec, max_steps=max_steps,
                            draw_border=draw_border)
        self.mailbox = Mailbox(self.workdir / "mailbox")
        self.controller = EpisodeController(self.env, self.mailbox,
                                            self.observation_dir, arm=arm)
        self.server_env = self.mailbox.env(self.observation_dir, self.workspace)
        self.config_path = write_mcp_config(self.workdir / "mcp.json", self.server_env)
        self.client: MCPStdioClient | None = None
        self.initial_observation: Path | None = None

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> "MCPEpisode":
        obs, _ = self.env.reset()
        self.controller.start()
        self.initial_observation = self.controller.write_initial_observation(obs)
        if self.start_client:
            self.client = MCPStdioClient(env=self.server_env).start()
        return self

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
        self.controller.stop()
        if self._tmp:
            self._tmp.cleanup()
            self._tmp = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False

    # --- results --------------------------------------------------------------

    def result(self) -> EpisodeResult:
        record = self.env.record
        return EpisodeResult(
            answer=self.controller.answer, success=self.controller.success,
            moves=self.controller.moves, steps=self.env.steps,
            termination_reason=record.termination_reason,
            rejected_requests=self.controller.rejected,
            windows=list(record.windows),
            observations=sorted(str(p) for p in self.observation_dir.glob("*.png")))

    def audit(self) -> dict:
        return self.controller.audit()

    def to_dict(self) -> dict:
        r = self.result()
        return {"answer": r.answer, "success": r.success, "moves": r.moves,
                "steps": r.steps, "termination_reason": r.termination_reason,
                "rejected_requests": r.rejected_requests,
                "n_observations": len(r.observations),
                "windows": [list(w) for w in r.windows], "arm": self.arm,
                "deliveries": list(self.controller.deliveries),
                "audit": self.controller.audit()}
