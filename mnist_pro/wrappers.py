"""Environment wrappers.

Provides structural components for managing evaluation:
* `TimeLimit`: Enforces the episode step limit.
* `TrajectoryRecorder`: Records window positions, actions, rewards, latencies, and token usage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .env import ActiveGlimpseEnv, TerminationReason


class Wrapper:
    """Minimal pass-through wrapper. Avoids a hard gymnasium dependency."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class TimeLimit(Wrapper):
    """Truncate after `max_steps` actions.

    Truncation is reported through `truncated=True` and
    `info["termination_reason"] == "step_limit"`. No answer is fabricated.
    """

    def __init__(self, env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated and self.env.steps >= self.max_steps:
            truncated = True
            self.env.mark_truncated()
            info = dict(info)
            info["termination_reason"] = TerminationReason.STEP_LIMIT.value
            info["answer"] = None
            info["success"] = False
        return obs, reward, terminated, truncated, info

    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.env.steps)


@dataclass
class StepRecord:
    step: int
    window_before: tuple
    window_after: tuple
    action: Any
    raw_response: str | None
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    latency_s: float | None = None
    usage: dict | None = None


@dataclass
class EpisodeRecord:
    label: str
    windows: list = field(default_factory=list)  # one per observation shown
    steps: list = field(default_factory=list)
    total_reward: float = 0.0
    termination_reason: str = TerminationReason.RUNNING.value
    answer: Any = None
    success: bool = False
    usage_totals: dict = field(default_factory=dict)

    @property
    def n_moves(self) -> int:
        return sum(
            1
            for s in self.steps
            if isinstance(s.action, dict) and s.action.get("action") == "move"
        )

    @property
    def unique_windows(self) -> list:
        return sorted(set(self.windows))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "success": self.success,
            "answer": self.answer,
            "termination_reason": self.termination_reason,
            "total_reward": round(self.total_reward, 6),
            "n_steps": len(self.steps),
            "n_moves": self.n_moves,
            "n_observations": len(self.windows),
            "n_unique_windows": len(self.unique_windows),
            "windows": [list(w) for w in self.windows],
            "unique_windows": [list(w) for w in self.unique_windows],
            "usage_totals": self.usage_totals,
            "trajectory": [
                {
                    "step": s.step,
                    "window_before": list(s.window_before),
                    "window_after": list(s.window_after),
                    "action": s.action,
                    "raw_response": s.raw_response,
                    "reward": s.reward,
                    "terminated": s.terminated,
                    "truncated": s.truncated,
                    "latency_s": s.latency_s,
                    "usage": s.usage,
                    "info": {k: v for k, v in s.info.items() if k != "window"},
                }
                for s in self.steps
            ],
        }


class TrajectoryRecorder(Wrapper):
    """Record window positions, actions, rewards, timing and token usage.

    `record.windows` holds the position of every observation shown to the agent.
    """

    def __init__(self, env):
        super().__init__(env)
        self.record: EpisodeRecord | None = None
        self._t0 = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.record = EpisodeRecord(label=self.env.label)
        self.record.windows.append(self.env.window)
        self._t0 = time.monotonic()
        return obs, info

    def step(self, action, raw_response: str | None = None, usage: dict | None = None):
        before = self.env.window
        t0 = time.monotonic()
        obs, reward, terminated, truncated, info = self.env.step(action)
        latency = time.monotonic() - t0

        parsed = action if isinstance(action, dict) else _safe_json(action)
        if isinstance(parsed, dict) and parsed.get("action") == "move":
            self.record.windows.append(self.env.window)

        self.record.steps.append(
            StepRecord(
                step=self.env.steps,
                window_before=before,
                window_after=self.env.window,
                action=parsed,
                raw_response=raw_response,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                latency_s=round(latency, 6),
                usage=usage,
            )
        )
        self.record.total_reward += reward
        if usage:
            for k, v in usage.items():
                if isinstance(v, (int, float)):
                    self.record.usage_totals[k] = self.record.usage_totals.get(k, 0) + v
        if terminated or truncated:
            self.record.termination_reason = info.get("termination_reason")
            self.record.answer = info.get("answer")
            self.record.success = bool(info.get("success", False))
        return obs, reward, terminated, truncated, info


def _safe_json(action):
    import json

    try:
        return json.loads(action)
    except Exception:
        return {"action": "<unparseable>", "raw": action}


def make_env(
    images,
    label,
    spec=None,
    max_steps: int | None = None,
    record: bool = True,
    **kwargs,
):
    """Build the standard evaluation stack: env -> TimeLimit -> TrajectoryRecorder."""
    env = ActiveGlimpseEnv(images, label, spec=spec, **kwargs)
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    if record:
        env = TrajectoryRecorder(env)
    return env
