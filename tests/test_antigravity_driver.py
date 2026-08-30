"""The Antigravity managed-agent driver, exercised without a credential or a socket.

This is the path the released six-arm Gemini results came through, so its shapes are
pinned: tool names, the per-level answer pattern, the `path_or_id` argument, the
interaction body, and the rule for which function calls are still pending.

Tool calls are executed through the MCP server, so the isolation guarantees tested in
`test_harness_isolation.py` apply to this driver too.
"""

import base64
import json

import numpy as np
import pytest
from PIL import Image

from mnist_pro.harness.antigravity import (AGENT, MAX_TOTAL_TOKENS, AntigravityClient,
                                           AntigravityDriver, initial_body,
                                           pending_calls, tool_definitions)
from mnist_pro.harness.session import MCPEpisode


def digit_image():
    a = np.zeros((28, 28), dtype=np.uint8)
    a[6:22, 10:14] = 255
    return Image.fromarray(a, mode="L")


def episode(digits=1, **kw):
    label = "7" if digits == 1 else "73"
    return MCPEpisode([digit_image() for _ in range(digits)], label, digits=digits, **kw)


class FakeAntigravity:
    """Replays scripted function calls in the managed-agent response shape."""

    def __init__(self, script):
        self.script = list(script)
        self.bodies = []
        self._n = 0

    def interaction(self, body):
        self.bodies.append(body)
        if not self.script:
            return {"id": f"int_{len(self.bodies)}", "steps": []}
        name, arguments = self.script.pop(0)
        self._n += 1
        return {"id": f"int_{len(self.bodies)}",
                "steps": [{"type": "function_call", "id": f"call_{self._n}",
                           "name": name, "arguments": arguments}]}


# --- shapes -------------------------------------------------------------------

def test_tool_definitions_match_the_released_controller():
    tools = tool_definitions(1)
    names = [t.get("name") for t in tools if t.get("type") == "function"]
    assert names == ["activeglimpse_move", "activeglimpse_view_image",
                     "activeglimpse_submit"]
    assert {"type": "code_execution"} in tools
    view = next(t for t in tools if t.get("name") == "activeglimpse_view_image")
    assert list(view["parameters"]["properties"]) == ["path_or_id"]


@pytest.mark.parametrize("digits,pattern", [(1, "^[0-9]$"), (2, "^[0-9]{2}$")])
def test_answer_pattern_is_per_level(digits, pattern):
    submit = next(t for t in tool_definitions(digits)
                  if t.get("name") == "activeglimpse_submit")
    assert submit["parameters"]["properties"]["value"]["pattern"] == pattern


def test_initial_body_matches_the_released_agent_config():
    body = initial_body("prompt", b"\x89PNG\r\n\x1a\nrest", digits=1)
    assert body["agent"] == AGENT
    assert body["agent_config"] == {"type": "antigravity", "model": "gemini-3.7-flash",
                                    "max_total_tokens": MAX_TOTAL_TOKENS}
    assert body["environment"]["network"] == "disabled"
    assert body["store"] is True
    image = body["input"][1]
    assert image["mime_type"] == "image/png"
    assert base64.b64decode(image["data"]).startswith(b"\x89PNG")


def test_notes_are_injected_only_when_an_arm_carries_them():
    without = initial_body("p", b"png", 1, notes=None)
    assert without["environment"]["sources"] == []
    with_notes = initial_body("p", b"png", 1, notes=b"remember this")
    source = with_notes["environment"]["sources"][0]
    assert source["target"] == "/workspace/NOTES.md"
    assert source["content"] == "remember this"


def test_pending_calls_skips_calls_that_already_have_results():
    response = {"steps": [
        {"type": "function_call", "id": "a", "name": "activeglimpse_move"},
        {"type": "function_result", "call_id": "a"},
        {"type": "function_call", "id": "b", "name": "activeglimpse_submit"},
        {"type": "function_call", "id": "c", "name": "something_else"},
    ]}
    assert [c["id"] for c in pending_calls(response)] == ["b"]


def test_pending_calls_tolerates_a_malformed_response():
    assert pending_calls({}) == []
    assert pending_calls({"steps": "not a list"}) == []


# --- driving ------------------------------------------------------------------

def test_driver_moves_then_submits_through_the_mcp_server():
    api = FakeAntigravity([("activeglimpse_move", {"direction": "right"}),
                           ("activeglimpse_move", {"direction": "down"}),
                           ("activeglimpse_submit", {"value": "7"})])
    with episode() as ep:
        result = AntigravityDriver(client=api, digits=1).drive(ep)
        assert result.submitted is True
        assert result.stop_reason == "submitted"
        assert result.tool_calls == 3
        assert ep.controller.moves == 2
        assert ep.controller.success is True
        assert ep.audit()["consistent"] is True


def test_driver_returns_images_to_the_agent_as_function_results():
    api = FakeAntigravity([("activeglimpse_move", {"direction": "right"}),
                           ("activeglimpse_submit", {"value": "7"})])
    with episode() as ep:
        AntigravityDriver(client=api, digits=1).drive(ep)
    follow_up = api.bodies[1]["input"][0]
    assert follow_up["type"] == "function_result"
    assert any(part["type"] == "image" for part in follow_up["output"])


def test_driver_accepts_arguments_as_a_json_string():
    api = FakeAntigravity([("activeglimpse_submit", json.dumps({"value": "7"}))])
    with episode() as ep:
        result = AntigravityDriver(client=api, digits=1).drive(ep)
        assert result.submitted is True
        assert ep.controller.success is True


def test_driver_reports_an_agent_that_stops_calling_tools():
    with episode() as ep:
        result = AntigravityDriver(client=FakeAntigravity([]), digits=1).drive(ep)
    assert result.submitted is False
    assert result.stop_reason == "agent stopped calling tools"


def test_driver_respects_the_turn_limit():
    api = FakeAntigravity([("activeglimpse_move", {"direction": "right"})] * 40)
    with episode(max_steps=100) as ep:
        result = AntigravityDriver(client=api, digits=1, max_turns=3).drive(ep)
    assert result.stop_reason == "turn limit"


def test_driver_carries_a_wrong_answer_through_correctly():
    api = FakeAntigravity([("activeglimpse_submit", {"value": "1"})])
    with episode(arm="A2") as ep:
        AntigravityDriver(client=api, digits=1).drive(ep)
        assert ep.controller.success is False


def test_two_digit_episode_uses_the_two_digit_prompt_and_pattern():
    api = FakeAntigravity([("activeglimpse_submit", {"value": "73"})])
    with episode(digits=2) as ep:
        AntigravityDriver(client=api, digits=2).drive(ep)
        assert ep.controller.success is True
    prompt = api.bodies[0]["input"][0]["text"]
    assert "Level 2" in prompt
    submit = next(t for t in api.bodies[0]["tools"] if t.get("name") == "activeglimpse_submit")
    assert submit["parameters"]["properties"]["value"]["pattern"] == "^[0-9]{2}$"


def test_unknown_tool_is_reported_not_executed():
    api = FakeAntigravity([("activeglimpse_move", {"direction": "right"})])
    with episode() as ep:
        driver = AntigravityDriver(client=api, digits=1)
        outcome, submitted = driver._execute(ep, {"name": "rm_rf", "arguments": {}})
        assert outcome["is_error"] is True
        assert submitted is False
        assert ep.controller.moves == 0


def test_client_requires_a_credential(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        AntigravityClient()
