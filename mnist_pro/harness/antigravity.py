"""The Antigravity managed-agent driver.

This is how Gemini was actually driven, and it is preserved rather than replaced. The
controller does not run the model itself: it posts an interaction to the Antigravity
managed-agent API, the agent comes back with pending function calls, the controller
executes those locally against the episode, and posts the results into the next turn.

    controller  ──POST interaction──▶  Antigravity agent (gemini-3.7-flash)
                ◀── pending calls ───
                ── execute locally ──▶  MCP server ──▶ episode mailbox ──▶ env
                ──POST results───────▶  next turn

Executing the calls through the MCP server rather than against the environment
directly keeps every isolation guarantee in one place: opaque observation names, the
workspace confinement, and the arm's feedback entitlement all still apply.

Shapes here -- tool names, the `path_or_id` argument, the per-level answer pattern,
the `environment` block with networking disabled -- are taken from the vendored
controller so runs stay comparable with the released six-arm results.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field

AGENT = "antigravity-preview-05-2026"
DEFAULT_MODEL = "gemini-3.7-flash"
MAX_TOTAL_TOKENS = 16_000
BASE_URL = "https://generativelanguage.googleapis.com"

TOOL_PREFIX = "activeglimpse_"
TOOL_NAMES = {f"{TOOL_PREFIX}move", f"{TOOL_PREFIX}view_image", f"{TOOL_PREFIX}submit"}


def tool_definitions(digits: int) -> list[dict]:
    """Verbatim from the released controller, with the answer pattern per level."""
    pattern = "^[0-9]$" if digits == 1 else "^[0-9]{%d}$" % digits
    return [
        {"type": "code_execution"},
        {"type": "function", "name": f"{TOOL_PREFIX}move",
         "description": "Move the reveal window once and return the resulting PNG natively.",
         "parameters": {"type": "object",
                        "properties": {"direction": {"type": "string",
                                                     "enum": ["up", "down", "left", "right"]}},
                        "required": ["direction"], "additionalProperties": False}},
        {"type": "function", "name": f"{TOOL_PREFIX}view_image",
         "description": "Reopen a previously exposed PNG by its opaque path or image id.",
         "parameters": {"type": "object",
                        "properties": {"path_or_id": {"type": "string"}},
                        "required": ["path_or_id"], "additionalProperties": False}},
        {"type": "function", "name": f"{TOOL_PREFIX}submit",
         "description": "Submit the final digit answer exactly once.",
         "parameters": {"type": "object",
                        "properties": {"value": {"type": "string", "pattern": pattern}},
                        "required": ["value"], "additionalProperties": False}},
    ]


def pending_calls(response: dict) -> list[dict]:
    """Function calls the agent has requested and that have no result yet."""
    steps = response.get("steps", [])
    if not isinstance(steps, list):
        return []
    done = {str(s.get("call_id")) for s in steps
            if isinstance(s, dict) and s.get("type") == "function_result"}
    return [s for s in steps
            if isinstance(s, dict) and s.get("type") == "function_call"
            and str(s.get("id")) not in done and s.get("name") in TOOL_NAMES]


def initial_body(prompt: str, initial_png: bytes, digits: int,
                 notes: bytes | None = None, model: str = DEFAULT_MODEL) -> dict:
    """The first interaction. Networking is disabled for the agent's environment."""
    sources = []
    if notes is not None:
        sources.append({"type": "inline", "target": "/workspace/NOTES.md",
                        "content": notes.decode("utf-8", errors="replace")})
    return {
        "agent": AGENT,
        "agent_config": {"type": "antigravity", "model": model,
                         "max_total_tokens": MAX_TOTAL_TOKENS},
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "mime_type": "image/png",
             "data": base64.b64encode(initial_png).decode("ascii")},
        ],
        "environment": {"type": "remote", "network": "disabled", "sources": sources},
        "tools": tool_definitions(digits),
        "store": True,
    }


class AntigravityClient:
    """Thin HTTP client for the managed-agent endpoint.

    Injectable: tests and dry runs pass a stand-in with the same `interaction`
    method, so the driver can be exercised without a credential or a socket.
    """

    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL,
                 timeout: float = 600.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Antigravity needs a Gemini credential; set GEMINI_API_KEY or pass "
                "api_key=. The released runs resolved it from the login keychain.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def interaction(self, body: dict) -> dict:
        import urllib.request
        request = urllib.request.Request(
            f"{self.base_url}/v1beta/interactions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": self.api_key})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())


PROMPT = {
    1: ("Level 1. You are identifying a single handwritten MNIST digit through a "
        "small reveal window on a mostly masked canvas. The digit is black on white; "
        "unrevealed area is dark gray. Use activeglimpse_move to reveal more, "
        "activeglimpse_view_image to reopen a PNG you were given, and "
        "activeglimpse_submit exactly once when you are confident."),
    2: ("Level 2. You are identifying two horizontally concatenated handwritten MNIST "
        "digits through a small reveal window on a mostly masked canvas. The digits "
        "are black on white; unrevealed area is dark gray. Use activeglimpse_move to "
        "reveal more, activeglimpse_view_image to reopen a PNG you were given, and "
        "activeglimpse_submit exactly once, giving both digits left to right."),
}


@dataclass
class AntigravityResult:
    submitted: bool = False
    turns: int = 0
    tool_calls: int = 0
    stop_reason: str = "unfinished"
    transcript: list = field(default_factory=list)


class AntigravityDriver:
    """Runs one MCPEpisode through the Antigravity managed agent."""

    def __init__(self, client=None, model: str = DEFAULT_MODEL, digits: int = 1,
                 max_turns: int = 60, notes: bytes | None = None):
        self.client = client or AntigravityClient()
        self.model = model
        self.digits = digits
        self.max_turns = max_turns
        self.notes = notes

    def drive(self, episode) -> AntigravityResult:
        if episode.client is None:
            raise RuntimeError("episode was started without an MCP client")
        initial = open(episode.initial_observation, "rb").read()
        body = initial_body(PROMPT[min(self.digits, 2)], initial, self.digits,
                            notes=self.notes, model=self.model)

        result = AntigravityResult()
        for _ in range(self.max_turns):
            result.turns += 1
            response = self.client.interaction(body)
            calls = pending_calls(response)
            if not calls:
                result.stop_reason = "agent stopped calling tools"
                return result

            results = []
            for call in calls:
                outcome, submitted = self._execute(episode, call)
                result.tool_calls += 1
                result.transcript.append({"name": call.get("name"),
                                          "arguments": call.get("arguments"),
                                          "is_error": outcome.get("is_error", False)})
                results.append({"type": "function_result",
                                "call_id": call.get("id"),
                                "output": outcome["output"]})
                if submitted:
                    result.submitted = True
                    result.stop_reason = "submitted"
                    return result

            body = {"interaction_id": response.get("id"), "input": results,
                    "store": True}
            if episode.controller.finished:
                result.stop_reason = episode.env.record.termination_reason
                return result
        result.stop_reason = "turn limit"
        return result

    def _execute(self, episode, call) -> tuple[dict, bool]:
        """Route one agent tool call through the MCP server."""
        name = call.get("name", "")
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if name == f"{TOOL_PREFIX}move":
            outcome = episode.client.move(arguments.get("direction"))
        elif name == f"{TOOL_PREFIX}view_image":
            outcome = episode.client.view_image(arguments.get("path_or_id", ""))
        elif name == f"{TOOL_PREFIX}submit":
            outcome = episode.client.submit(arguments.get("value"))
        else:
            return {"output": [{"type": "text", "text": "unknown tool"}],
                    "is_error": True}, False

        output = [{"type": "text", "text": outcome.text()}]
        for image in outcome.images():
            output.append({"type": "image",
                           "mime_type": image.get("mimeType", "image/png"),
                           "data": image["data"]})
        submitted = name.endswith("submit") and not outcome.is_error
        return {"output": output, "is_error": outcome.is_error}, submitted
