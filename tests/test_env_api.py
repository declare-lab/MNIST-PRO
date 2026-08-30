"""Environment contract: the Gymnasium five-tuple, and failures that are scored
rather than raised.

The original `step()` raised `ValueError` on unparseable JSON, an unknown direction
or an unknown action type. A model that emitted a bad direction therefore killed the
process instead of losing the episode.
"""

import json

import numpy as np
import pytest
from PIL import Image

from mnist_pro.env import ActiveGlimpseEnv, TerminationReason
from mnist_pro.rendering import CanvasSpec
from mnist_pro.wrappers import TimeLimit, TrajectoryRecorder, make_env


def fake_digit(value=0):
    """A 28x28 MNIST-like image: white strokes on black, as torchvision supplies."""
    a = np.zeros((28, 28), dtype=np.uint8)
    a[6:22, 10:14] = 255
    a[6:10, 10:20] = 255
    return Image.fromarray(a, mode="L")


def make(digits=1, **kw):
    spec = CanvasSpec(digits=digits, image_size=224, box_size=64, step_size=32)
    label = 0 if digits == 1 else "00"
    return ActiveGlimpseEnv([fake_digit() for _ in range(digits)], label, spec=spec, **kw)


def test_reset_returns_obs_and_info():
    env = make()
    obs, info = env.reset()
    assert isinstance(obs, Image.Image)
    assert info["termination_reason"] == TerminationReason.RUNNING.value
    assert info["window"] == env.window


def test_step_returns_five_tuple():
    env = make()
    env.reset()
    result = env.step('{"action": "move", "direction": "right"}')
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert not terminated and not truncated
    assert reward == pytest.approx(-0.1)


def test_move_is_clamped_at_the_canvas_edge():
    env = make()
    env.reset()
    for _ in range(50):
        env.step({"action": "move", "direction": "left"})
    assert env.window[0] == 0
    for _ in range(50):
        env.step({"action": "move", "direction": "right"})
    assert env.window[0] == env.spec.width - env.spec.box_size


@pytest.mark.parametrize("bad", [
    "not json at all",
    '{"action": "move", "direction": "diagonally"}',
    '{"action": "teleport"}',
    '["not", "an", "object"]',
])
def test_malformed_actions_are_scored_not_raised(bad):
    env = make()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(bad)
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == TerminationReason.INVALID_ACTION.value
    assert info["success"] is False
    assert info["error"]


def test_correct_answer_terminates_with_reward():
    env = make()
    env.reset()
    _, reward, terminated, truncated, info = env.step({"action": "answer", "value": 0})
    assert terminated and not truncated
    assert reward == pytest.approx(10.0)
    assert info["success"] is True
    assert info["termination_reason"] == TerminationReason.ANSWERED.value


def test_step_limit_truncates_without_inventing_an_answer():
    """The old driver injected `{"action": "answer", "value": -1}` at the limit,
    which was indistinguishable from a genuine answer of -1 and from the agent's
    JSON-parse fallback. Truncation now carries no answer at all."""
    env = TimeLimit(make(), max_steps=3)
    env.reset()
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(
            {"action": "move", "direction": "right"})
    assert truncated is True
    assert terminated is False
    assert info["termination_reason"] == TerminationReason.STEP_LIMIT.value
    assert info["answer"] is None


def test_multi_digit_uses_string_comparison():
    env = make(digits=2)
    env.reset()
    _, _, _, _, info = env.step({"action": "answer", "value": "00"})
    assert info["success"] is True


def test_seeded_reset_can_move_the_start():
    """The original start was fixed per image by a content hash, so start-position
    sensitivity could not be measured at all."""
    env = make()
    default, _ = env.reset()
    starts = {env.window}
    for seed in range(25):
        env.reset(seed=seed)
        starts.add(env.window)
    assert len(starts) > 1


def test_recorder_captures_windows_actions_and_usage():
    env = make_env([fake_digit()], 0, spec=CanvasSpec(), max_steps=10)
    env.reset()
    env.step({"action": "move", "direction": "right"}, raw_response="r",
             usage={"total_tokens": 11})
    env.step({"action": "answer", "value": 0}, raw_response="a",
             usage={"total_tokens": 7})
    rec = env.record
    assert len(rec.windows) == 2           # start, then one move; answer does not move
    assert rec.n_moves == 1
    assert rec.success is True
    assert rec.termination_reason == TerminationReason.ANSWERED.value
    assert rec.usage_totals["total_tokens"] == 18
    assert rec.steps[0].latency_s is not None
    payload = rec.to_dict()
    assert payload["n_unique_windows"] == len(set(map(tuple, payload["unique_windows"])))
    json.dumps(payload)                     # must be serialisable as written


def test_single_env_class_handles_both_levels():
    """The two 92%-identical env classes are gone; digits is a parameter."""
    assert make(digits=1).canvas.shape == (224, 224)
    assert make(digits=2).canvas.shape == (224, 448)
