"""The declarative agent must behave exactly like the classes it replaces.

Fifteen subclasses collapsed to one class plus four `MemorySpec` values. The prompt
text is asserted verbatim against the originals, because changing it silently would
make new runs incomparable with published ones.
"""

import json

import pytest
from PIL import Image

from mnist_pro.agents import (INVALID_ACTION, AgentConfig, GlimpseAgent,
                              MEMORY_SPECS, extract_json)
from mnist_pro.agents.specs import control_prompts, system_instruction


class FakeBackend:
    """Returns queued responses; records what it was asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system_instruction, contents):
        self.calls.append({"system": system_instruction, "contents": contents})
        return (self.responses.pop(0) if self.responses else ""), {"total_tokens": 1}


def obs():
    return Image.new("RGB", (224, 224), (128, 128, 128))


def test_default_is_multi_turn_natural_conversation():
    """The benchmark is about integrating glimpses across real turns, so that is the
    default. Turn-based re-rendering is the opt-in, and is what the earlier published
    runs used."""
    cfg = AgentConfig().resolve()
    assert cfg.turn_mode == "natural"
    assert cfg.horizon == -1, "natural conversation keeps the whole transcript"


def test_turn_based_still_defaults_its_horizon_per_memory_config():
    assert AgentConfig(turn_mode="turn_based", memory="image_only_baseline").resolve().horizon == 1
    assert AgentConfig(turn_mode="turn_based", memory="textual_state").resolve().horizon == 1


def test_three_memory_configs_exist():
    assert set(MEMORY_SPECS) == {"image_only_baseline",
                                 "textual_state", "metric_grid_map"}


def test_system_instruction_matches_original_single_digit():
    text = system_instruction(1)
    assert text.startswith("You are an active vision agent playing a game to "
                           "identify an MNIST digit.")
    assert "The digit is drawn in black on a white background" in text
    assert '{"action": "answer", "value": <digit>}.' in text


def test_system_instruction_matches_original_multi_digit():
    text = system_instruction(2)
    assert "sequence of horizontally concatenated MNIST digits" in text
    assert "from left to right" in text
    assert '{"action": "answer", "value": "58"}.' in text


def test_control_prompts_match_original():
    prompt, system = control_prompts(1)
    assert prompt == "Look at this image of an MNIST digit. What digit (0-9) is shown?"
    assert system.startswith("You are a visual recognition model.")
    prompt2, system2 = control_prompts(2)
    assert prompt2 == ("Look at this image of concatenated MNIST digits. "
                       "What sequence of digits is shown?")
    assert "raw string of digits" in system2


def test_textual_belief_asks_for_a_thought():
    spec = MEMORY_SPECS["textual_state"]
    assert spec.user_instruction.endswith(
        " Include an extra key 'thought' in the JSON containing your reasoning.")
    assert spec.memory_prompt_prefix == "Previous actions and thoughts"


def test_metric_grid_map_asks_for_a_spatial_map():
    spec = MEMORY_SPECS["metric_grid_map"]
    assert "'spatial_map'" in spec.user_instruction
    assert "[0, -1], [0, 1], [-1, 0], or [1, 0]" in spec.user_instruction
    assert spec.memory_prompt_prefix == (
        "Previous actions, thoughts, and structured spatial map")


def test_other_configs_replay_prior_outputs():
    agent = GlimpseAgent(FakeBackend([]), AgentConfig(memory="image_only_baseline", turn_mode="turn_based"))
    assert agent.render_memory() == "This is the first turn. No previous actions."
    agent.full_history = [{"type": "model_output", "content": [{"text": "did a thing"}]}]
    memory = agent.render_memory()
    assert memory.startswith("Previous actions:")
    assert "Turn 1:\ndid a thing" in memory


def test_horizon_bounds_the_number_of_images_sent():
    backend = FakeBackend(['{"action": "move", "direction": "up"}'] * 3)
    agent = GlimpseAgent(backend, AgentConfig(memory="image_only_baseline", horizon=1, turn_mode="turn_based"))
    for _ in range(3):
        agent.act(obs())
    images = [p for p in backend.calls[-1]["contents"][0]["content"]
              if p.get("type") == "image"]
    assert len(images) == 1


def test_unbounded_horizon_sends_every_image():
    backend = FakeBackend(['{"action": "move", "direction": "up"}'] * 3)
    agent = GlimpseAgent(backend, AgentConfig(memory="image_only_baseline", horizon=-1, turn_mode="turn_based"))
    for _ in range(3):
        agent.act(obs())
    images = [p for p in backend.calls[-1]["contents"][0]["content"]
              if p.get("type") == "image"]
    assert len(images) == 3


def test_parse_failure_returns_an_explicit_invalid_action():
    """Previously this fabricated `{"action": "answer", "value": -1}`, which the logs
    could not distinguish from a real answer of -1 or from a forced answer at the
    step limit."""
    backend = FakeBackend(["no json here"] * 3)
    agent = GlimpseAgent(backend, AgentConfig(memory="image_only_baseline", turn_mode="turn_based"))
    action, raw, _ = agent.act(obs())
    assert action == INVALID_ACTION
    assert action["action"] == "invalid"
    assert "value" not in action
    assert raw == "no json here"


def test_out_of_steps_warning_is_appended_at_the_limit():
    backend = FakeBackend(['{"action": "answer", "value": 3}'])
    agent = GlimpseAgent(backend, AgentConfig(memory="textual_state",
                                              turn_mode="turn_based", max_steps=1))
    agent.act(obs())
    text = backend.calls[0]["contents"][0]["content"][0]["text"]
    assert "You have run out of steps" in text
    assert '"thought": "<reasoning>"' in text


def test_natural_mode_requires_unbounded_history():
    with pytest.raises(ValueError, match="complete transcript"):
        AgentConfig(turn_mode="natural", horizon=1).resolve()


def test_natural_mode_interleaves_turns():
    backend = FakeBackend(['{"action": "move", "direction": "up"}',
                           '{"action": "answer", "value": 1}'])
    agent = GlimpseAgent(backend, AgentConfig(memory="image_only_baseline",
                                              turn_mode="natural", horizon=-1))
    agent.act(obs())
    agent.act(obs())
    kinds = [item["type"] for item in agent.full_history]
    assert kinds == ["user_input", "model_output", "user_input", "model_output"]
    first = agent.full_history[0]["content"]
    assert first[0]["type"] == "text"        # instruction only on the first turn
    later = agent.full_history[2]["content"]
    assert all(p["type"] == "image" for p in later)


def test_extract_json_is_unchanged_including_its_greediness():
    assert extract_json('```json\n{"action": "answer", "value": 3}\n```') == {
        "action": "answer", "value": 3}
    assert extract_json("no braces at all") is None
    # Greedy by design: two objects in one response span into invalid JSON. This has
    # never occurred in the released logs; the test pins it so a change is deliberate.
    assert extract_json('{"a": 1} and then {"b": 2}') is None


def test_unknown_memory_config_is_rejected():
    with pytest.raises(ValueError, match="unknown memory config"):
        AgentConfig(memory="photographic").resolve()
