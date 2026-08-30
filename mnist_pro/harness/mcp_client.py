"""A minimal MCP stdio client.

Enough of the protocol to drive `ag_mcp_server.py`: initialize, ping, tools/list and
tools/call over newline-delimited JSON-RPC on stdin/stdout. It exists so the MCP
harness can be exercised in-process -- by a test, or by the built-in tool-calling
agent -- without requiring an external agent runtime to be installed.

External runtimes (Claude Code, Antigravity) speak to the same server themselves;
`registry.write_mcp_config` emits the config they consume.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

PROTOCOL_VERSION = "2025-06-18"
SERVER_PATH = Path(__file__).with_name("ag_mcp_server.py")


class MCPError(RuntimeError):
    pass


@dataclass
class ToolResult:
    content: list
    is_error: bool = False

    def text(self) -> str:
        return "\n".join(part.get("text", "") for part in self.content
                         if part.get("type") == "text")

    def images(self) -> list[dict]:
        return [p for p in self.content if p.get("type") == "image"]


class MCPStdioClient:
    """Speaks JSON-RPC to a server subprocess over stdio."""

    def __init__(self, command: list[str] | None = None, env: dict | None = None,
                 cwd: str | None = None, timeout: float = 130.0):
        self.command = command or [sys.executable, str(SERVER_PATH)]
        self.env = {**os.environ, **(env or {})}
        self.cwd = cwd
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self._id = 0
        self._lock = threading.Lock()
        self.server_info: dict = {}
        self.tools: list[dict] = []

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> "MCPStdioClient":
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=self.env, cwd=self.cwd, text=True, bufsize=1)
        result = self.request("initialize",
                              {"protocolVersion": PROTOCOL_VERSION,
                               "capabilities": {},
                               "clientInfo": {"name": "mnist-pro", "version": "0.1.0"}})
        self.server_info = result.get("serverInfo", {})
        self.tools = self.request("tools/list").get("tools", [])
        return self

    def close(self):
        if self.process is None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        finally:
            self.process = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False

    # --- rpc ------------------------------------------------------------------

    def request(self, method: str, params: dict | None = None) -> dict:
        if self.process is None or self.process.stdin is None:
            raise MCPError("client is not started")
        with self._lock:
            self._id += 1
            message = {"jsonrpc": "2.0", "id": self._id, "method": method}
            if params is not None:
                message["params"] = params
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise MCPError(f"server closed the connection. stderr: {stderr[:500]}")
        response = json.loads(line)
        if "error" in response:
            raise MCPError(f"{method}: {response['error'].get('message')}")
        return response.get("result", {})

    def call_tool(self, name: str, arguments: dict | None = None) -> ToolResult:
        result = self.request("tools/call",
                              {"name": name, "arguments": arguments or {}})
        return ToolResult(content=result.get("content", []),
                          is_error=bool(result.get("isError")))

    # --- convenience ----------------------------------------------------------

    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools]

    def move(self, direction: str) -> ToolResult:
        return self.call_tool("move", {"direction": direction})

    def submit(self, value) -> ToolResult:
        return self.call_tool("submit", {"value": value})

    def view_image(self, path: str) -> ToolResult:
        return self.call_tool("view_image", {"path": path})
