"""A tool-calling agent that drives an MCP episode.

The agent never emits JSON for the framework to parse. It calls `move`, `view_image`
and `submit` as tools, and the images come back as native MCP image content. That
removes response parsing as a source of failure -- which the in-process harnesses
have to handle explicitly -- and is why `mcp` cells are scored separately.

Tool schemas are taken from the server's own `tools/list`, so the model is offered
exactly what the server implements and the two cannot drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

SYSTEM = {
    1: ("You are an active vision agent identifying an MNIST digit. You see a small "
        "window onto a mostly masked canvas: the digit is drawn in black on white, "
        "and everything unrevealed is dark gray. Call `move` to shift the window and "
        "reveal more. When you are confident, call `submit` with the digit 0-9. You "
        "get one submission, so be confident before you use it."),
    2: ("You are an active vision agent identifying a sequence of two horizontally "
        "concatenated MNIST digits. You see a small window onto a mostly masked "
        "canvas: the digits are drawn in black on white, and everything unrevealed "
        "is dark gray. Call `move` to shift the window and reveal more. When you are "
        "confident, call `submit` with the two digits left to right, as a string "
        "such as \"58\". You get one submission, so be confident before you use it."),
}

FIRST_TURN = ("Here is your starting view. Explore until you can identify the "
              "digits, then submit.")


def to_openai_tools(mcp_tools: list[dict]) -> list[dict]:
    """Translate MCP tool descriptors into OpenAI-compatible function schemas."""
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t.get("description", ""),
                          "parameters": t.get("inputSchema",
                                              {"type": "object", "properties": {}})}}
            for t in mcp_tools]


@dataclass
class ToolEpisodeResult:
    submitted: bool = False
    tool_calls: int = 0
    turns: int = 0
    stop_reason: str = "unfinished"
    transcript: list = field(default_factory=list)
    usage_totals: dict = field(default_factory=dict)


class OpenAIToolAgent:
    """Drives an MCPEpisode using an OpenAI-compatible chat client.

    Works for any provider exposing the standard `tools=` / `tool_calls` shape --
    DeepSeek, OpenRouter and OpenAI itself. Other providers need their own adapter;
    `drive` raises rather than silently degrading to text parsing.
    """

    def __init__(self, client, model: str, digits: int = 1, max_turns: int = 60):
        self.client = client
        self.model = model
        self.digits = digits
        self.max_turns = max_turns

    def drive(self, episode) -> ToolEpisodeResult:
        mcp = episode.client
        if mcp is None:
            raise RuntimeError("episode was started without an MCP client")
        tools = to_openai_tools(mcp.tools)
        messages = [{"role": "system", "content": SYSTEM[min(self.digits, 2)]},
                    {"role": "user", "content": [
                        {"type": "text", "text": FIRST_TURN},
                        _image_part(episode.initial_observation)]}]

        result = ToolEpisodeResult()
        for _ in range(self.max_turns):
            result.turns += 1
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools)
            message = response.choices[0].message
            _accumulate(result.usage_totals, getattr(response, "usage", None))
            calls = getattr(message, "tool_calls", None) or []
            messages.append(_assistant_message(message, calls))
            if not calls:
                result.stop_reason = "model stopped calling tools"
                return result

            for call in calls:
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                outcome = mcp.call_tool(name, arguments)
                result.tool_calls += 1
                result.transcript.append({"tool": name, "arguments": arguments,
                                          "is_error": outcome.is_error})
                messages.append(_tool_message(call.id, name, outcome))
                if name == "submit" and not outcome.is_error:
                    result.submitted = True
                    result.stop_reason = "submitted"
                    return result
            if episode.controller.finished:
                result.stop_reason = episode.env.record.termination_reason
                return result
        result.stop_reason = "turn limit"
        return result


def _image_part(path):
    import base64
    with open(path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"}}


def _assistant_message(message, calls):
    out = {"role": "assistant", "content": getattr(message, "content", None) or ""}
    if calls:
        out["tool_calls"] = [{"id": c.id, "type": "function",
                              "function": {"name": c.function.name,
                                           "arguments": c.function.arguments}}
                             for c in calls]
    return out


def _tool_message(call_id, name, outcome):
    """Tool results carry the image back as a data URL where the provider allows it."""
    content = [{"type": "text", "text": outcome.text() or f"{name} ok"}]
    for image in outcome.images():
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{image.get('mimeType', 'image/png')};base64,{image['data']}"}})
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _accumulate(totals, usage):
    if usage is None:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0) + value
