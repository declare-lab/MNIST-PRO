"""Harness registry, tool-calling agent, and the runner's MCP path.

No provider is contacted: the tool agent is driven by a fake OpenAI-shaped client,
and the runner is given a scripted driver. The MCP server itself is a real
subprocess throughout.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mnist_pro import runner
from mnist_pro.harness import registry
from mnist_pro.harness.tool_agent import OpenAIToolAgent, to_openai_tools
from mnist_pro.harness.session import MCPEpisode
from mnist_pro.matrix import Cell
from mnist_pro.dataset import EpisodeSpec
from mnist_pro.rendering import CanvasSpec


def digit_image():
    a = np.zeros((28, 28), dtype=np.uint8)
    a[6:22, 10:14] = 255
    return Image.fromarray(a, mode="L")


class FakeDataset:
    def __init__(self, n=8):
        self.items = [(digit_image(), i % 10) for i in range(n)]

    def __getitem__(self, i):
        return self.items[i]


# --- registry -----------------------------------------------------------------

def test_every_harness_declares_whether_memory_is_controlled():
    for spec in registry.HARNESSES.values():
        assert spec.family in {"inprocess", "tool_use"}
        assert isinstance(spec.controls_memory, bool)
        # Only the framework-held loops control memory; a tool-use runtime brings
        # its own context management, which is why those cells report separately.
        assert spec.controls_memory == (spec.family == "inprocess")


def test_named_runtimes_are_registered():
    assert {"mcp", "antigravity", "claude_code", "deepseek"} <= set(registry.HARNESSES)
    assert set(registry.IN_PROCESS) == {"turn_based", "natural"}


def test_availability_names_what_is_missing():
    report = {r["name"]: r for r in registry.availability_report()}
    assert report["mcp"]["available"] is True          # needs nothing external
    claude = report["claude_code"]
    if not claude["available"]:
        assert any("claude" in m for m in claude["missing"])
        assert claude["install_hint"]


def test_unknown_harness_is_rejected():
    with pytest.raises(ValueError, match="unknown harness"):
        registry.get("telepathy")


def test_claude_code_command_points_at_the_episode_config(tmp_path):
    cmd = registry.external_command("claude_code", tmp_path / "mcp.json", "go")
    assert cmd[0] == "claude"
    assert "--mcp-config" in cmd
    assert str(tmp_path / "mcp.json") in cmd
    allowed = cmd[cmd.index("--allowed-tools") + 1]
    assert "mcp__activeglimpse__move" in allowed
    assert "mcp__activeglimpse__submit" in allowed


def test_in_process_harness_has_no_external_command():
    with pytest.raises(ValueError, match="in-process"):
        registry.external_command("natural", "x", "go")


# --- tool schema translation ---------------------------------------------------

def test_mcp_tools_translate_to_openai_function_schemas():
    with MCPEpisode([digit_image()], "0", digits=1) as ep:
        tools = to_openai_tools(ep.client.tools)
    names = {t["function"]["name"] for t in tools}
    assert names == {"move", "view_image", "submit"}
    move = next(t for t in tools if t["function"]["name"] == "move")
    assert move["type"] == "function"
    assert move["function"]["parameters"]["properties"]["direction"]["enum"] == [
        "up", "down", "left", "right"]


# --- tool-calling agent --------------------------------------------------------

class FakeOpenAIClient:
    """Replays a script of tool calls through the OpenAI response shape."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, tools=None, **kw):
        self.calls.append({"messages": messages, "tools": tools})
        if not self.script:
            message = SimpleNamespace(content="done", tool_calls=None)
        else:
            name, arguments = self.script.pop(0)
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                id=f"call_{len(self.calls)}",
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2,
                                  total_tokens=12))


def test_tool_agent_explores_then_submits():
    client = FakeOpenAIClient([("move", {"direction": "right"}),
                              ("move", {"direction": "down"}),
                              ("submit", {"value": "0"})])
    with MCPEpisode([digit_image()], "0", digits=1) as ep:
        result = OpenAIToolAgent(client, "fake-model", digits=1).drive(ep)
        assert result.submitted is True
        assert result.stop_reason == "submitted"
        assert result.tool_calls == 3
        assert ep.controller.moves == 2
        assert ep.controller.success is True
    assert result.usage_totals["total_tokens"] == 36


def test_tool_agent_sends_the_starting_view_and_the_tool_schemas():
    client = FakeOpenAIClient([("submit", {"value": "0"})])
    with MCPEpisode([digit_image()], "0", digits=1) as ep:
        OpenAIToolAgent(client, "fake-model", digits=1).drive(ep)
    first = client.calls[0]
    assert {t["function"]["name"] for t in first["tools"]} == {"move", "view_image",
                                                              "submit"}
    user = first["messages"][1]["content"]
    assert any(p["type"] == "image_url" for p in user)


def test_tool_agent_feeds_returned_images_back_to_the_model():
    client = FakeOpenAIClient([("move", {"direction": "right"}),
                              ("submit", {"value": "0"})])
    with MCPEpisode([digit_image()], "0", digits=1) as ep:
        OpenAIToolAgent(client, "fake-model", digits=1).drive(ep)
    tool_messages = [m for call in client.calls for m in call["messages"]
                     if m.get("role") == "tool"]
    assert tool_messages
    assert any(part["type"] == "image_url"
               for m in tool_messages for part in m["content"])


def test_tool_agent_stops_when_the_model_stops_calling_tools():
    client = FakeOpenAIClient([])
    with MCPEpisode([digit_image()], "0", digits=1) as ep:
        result = OpenAIToolAgent(client, "fake-model", digits=1).drive(ep)
    assert result.submitted is False
    assert result.stop_reason == "model stopped calling tools"


def test_tool_agent_respects_the_turn_limit():
    client = FakeOpenAIClient([("move", {"direction": "right"})] * 50)
    with MCPEpisode([digit_image()], "0", digits=1, max_steps=100) as ep:
        result = OpenAIToolAgent(client, "fake", digits=1, max_turns=3).drive(ep)
    assert result.stop_reason == "turn limit"


# --- runner integration --------------------------------------------------------

def test_runner_executes_an_mcp_episode(tmp_path):
    cell = Cell(model="fake", digits=1, memory="image_only_baseline", horizon=-1,
                turn_mode="natural", harness="mcp", arm="A2")

    def scripted(episode):
        episode.client.move("right")
        episode.client.submit("0")

    out = runner.run_mcp_episode(EpisodeSpec(0, (0,), (0,)), cell, FakeDataset(),
                                 str(tmp_path), max_steps=10, driver=scripted)
    assert out["harness"] == "mcp"
    assert out["arm"] == "A2"
    assert out["success"] is True
    assert out["termination_reason"] == "answered"
    assert out["n_observations"] >= 2
    assert out["deliveries"], "the server should report what it delivered"
    assert out["deliveries"][0]["delivery_format"] == "mcp_image_content"
    assert "stroke_coverage" in out
    assert (tmp_path / "episode_0" / "original.png").exists()
    assert (tmp_path / "episode_0" / "trajectory.json").exists()


def test_runner_records_a_driver_failure_without_aborting(tmp_path):
    cell = Cell(model="fake", digits=1, memory="image_only_baseline", horizon=-1,
                turn_mode="natural", harness="mcp")

    def broken(episode):
        raise RuntimeError("runtime went away")

    out = runner.run_mcp_episode(EpisodeSpec(1, (1,), (1,)), cell, FakeDataset(),
                                 str(tmp_path), max_steps=5, driver=broken)
    assert "runtime went away" in out["error"]
    assert (tmp_path / "episode_1" / "trajectory.json").exists()


def test_gemini_routes_to_the_antigravity_managed_agent(monkeypatch):
    """Gemini is driven the way the released runs drove it, not by the OpenAI loop."""
    seen = {}

    class FakeDriver:
        def __init__(self, **kw):
            seen.update(kw)

        def drive(self, episode):
            return "antigravity"

    import mnist_pro.harness.antigravity as ag
    monkeypatch.setattr(ag, "AntigravityDriver", FakeDriver)
    cell = Cell(model="gemini-3.7-flash", harness="mcp", horizon=-1, digits=2)
    with MCPEpisode([digit_image()] * 2, "00", digits=2) as ep:
        assert runner._default_mcp_driver(cell)(ep) == "antigravity"
    assert seen["model"] == "gemini-3.7-flash"
    assert seen["digits"] == 2


def test_default_mcp_driver_refuses_a_provider_it_cannot_drive(monkeypatch):
    """Rather than silently falling back to text parsing, it says why it cannot run."""
    cell = Cell(model="some-other-model", harness="mcp", horizon=-1)
    monkeypatch.setattr(runner, "get_backend",
                        lambda *a, **k: SimpleNamespace(client=None))
    driver = runner._default_mcp_driver(cell)
    with MCPEpisode([digit_image()], "0", digits=1) as ep:
        with pytest.raises(NotImplementedError, match="no MCP driver"):
            driver(ep)
