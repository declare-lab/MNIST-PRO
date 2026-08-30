"""The active-glimpse environment, unified across digit counts.

One class replaces the previous `MnistActiveVisionEnv` / `MultiDigitActiveVisionEnv`
pair, which were 92% textually identical and had to be edited in parallel for every
change. `digits=1` and `digits=2` are now the same code path.

The API is Gymnasium's:

    obs, info                                = env.reset(seed=None)
    obs, reward, terminated, truncated, info = env.step(action)

Two consequences matter for evaluation:

* `truncated` marks the step limit, distinct from `terminated`, so "ran out of steps"
  is no longer conflated with "answered wrong". The old code signalled both by
  injecting an answer of -1, which was also the agent's JSON-parse fallback value --
  three different failures sharing one sentinel.
* A malformed action **terminates the episode and is scored**, rather than raising.
  The old `step()` raised `ValueError` on an unknown direction or unparseable JSON,
  which killed the whole run instead of recording the failure.

`info["termination_reason"]` is always one of `TerminationReason`, recorded
explicitly rather than recovered later by matching a string in a log field.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image

try:  # Gymnasium supplies the spaces and the base class when installed.
    import gymnasium as gym
    from gymnasium import spaces
    _BASE = gym.Env
except ImportError:  # pragma: no cover - the env is usable without the dependency
    spaces = None
    _BASE = object

from .rendering import (CanvasSpec, build_canvas, deterministic_start,
                        render_observation, seeded_start)

DIRECTIONS = ("up", "down", "left", "right")


class TerminationReason(str, Enum):
    """Why an episode ended. Recorded in `info`, never inferred from a log string."""

    ANSWERED = "answered"                    # the agent committed to an answer
    STEP_LIMIT = "step_limit"                # truncated by TimeLimit
    INVALID_ACTION = "invalid_action"        # unparseable or unknown action
    RUNNING = "running"                      # not finished


class ActiveGlimpseEnv(_BASE):
    """A masked canvas with one movable glimpse window.

    Args:
        images: `digits` PIL images, 28x28 greyscale MNIST.
        label: ground truth. An int for one digit, a string such as "58" for more.
        spec: canvas geometry. Defaults to the published 224px / 64px / 32px setup.
        draw_border: keep the cyan outline on observations. True reproduces the
            published runs; False removes the outline that occludes the outer two
            pixels of every glimpse.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, images, label, spec: CanvasSpec | None = None,
                 step_cost: float = -0.1, answer_reward: float = 10.0,
                 wrong_penalty: float = -10.0, draw_border: bool = True,
                 digits: int | None = None, image_size: int | None = None,
                 box_size: int | None = None, step_size: int | None = None):
        if spec is None:
            spec = CanvasSpec(
                digits=digits if digits is not None else len(images),
                image_size=image_size if image_size is not None else 224,
                box_size=box_size if box_size is not None else 64,
                step_size=step_size if step_size is not None else 32,
            )
        self.spec = spec
        self.canvas = build_canvas(list(images), spec)
        self.label = self._normalise_label(label, spec.digits)
        self.step_cost = step_cost
        self.answer_reward = answer_reward
        self.wrong_penalty = wrong_penalty
        self.draw_border = draw_border

        if spaces is not None:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(spec.height, spec.width, 3), dtype=np.uint8)
            # Actions arrive as JSON text from a language model. The semantic action
            # set is {move(direction), answer(value)}; see `DIRECTIONS`.
            self.action_space = spaces.Text(max_length=4096)

        self.x = self.y = 0
        self.steps = 0
        self.done = False
        self._reason = TerminationReason.RUNNING
        self.reset()

    # --- labels ---------------------------------------------------------------

    @staticmethod
    def _normalise_label(label, digits):
        return str(int(label)) if digits == 1 else str(label).strip()

    def _normalise_answer(self, value):
        """Match the original comparison semantics exactly.

        One digit compared as an int (an unparseable value became -1); more than one
        compared as a stripped string.
        """
        if self.spec.digits == 1:
            try:
                return str(int(value))
            except (TypeError, ValueError):
                return "-1"
        return str(value).strip()

    # --- gymnasium API --------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Place the window and clear episode state.

        With no seed the start is the original content-hashed position, so runs
        reproduce. With a seed the start is drawn from it instead, which is what
        makes start-position sensitivity measurable at all.
        """
        if seed is None:
            self.x, self.y = deterministic_start(self.canvas, self.spec)
        else:
            if hasattr(super(), "reset"):
                try:
                    super().reset(seed=seed)
                except TypeError:
                    pass
            self.x, self.y = seeded_start(self.canvas, self.spec, seed)
        self.steps = 0
        self.done = False
        self._reason = TerminationReason.RUNNING
        return self._observation(), self._info()

    def step(self, action: Any):
        """Apply one action. Never raises on malformed input -- it scores it."""
        if self.done:
            return (self._observation(), 0.0, True, False,
                    self._info(error="episode already finished"))

        self.steps += 1
        parsed, parse_error = self._parse_action(action)
        if parse_error is not None:
            self.done = True
            self._reason = TerminationReason.INVALID_ACTION
            return (self._observation(), self.wrong_penalty, True, False,
                    self._info(error=parse_error, answer=None, success=False))

        kind = parsed.get("action")
        if kind == "move":
            direction = parsed.get("direction")
            if direction not in DIRECTIONS:
                self.done = True
                self._reason = TerminationReason.INVALID_ACTION
                return (self._observation(), self.wrong_penalty, True, False,
                        self._info(error=f"unknown direction: {direction!r}",
                                   answer=None, success=False))
            self._move(direction)
            return self._observation(), self.step_cost, False, False, self._info()

        if kind == "answer":
            answer = self._normalise_answer(parsed.get("value"))
            success = answer == self.label
            self.done = True
            self._reason = TerminationReason.ANSWERED
            reward = self.answer_reward if success else self.wrong_penalty
            return (self._observation(), reward, True, False,
                    self._info(answer=answer, success=success))

        self.done = True
        self._reason = TerminationReason.INVALID_ACTION
        return (self._observation(), self.wrong_penalty, True, False,
                self._info(error=f"unknown action type: {kind!r}",
                           answer=None, success=False))

    def render(self):
        return np.array(self._observation())

    # --- internals ------------------------------------------------------------

    @staticmethod
    def _parse_action(action):
        if isinstance(action, dict):
            return action, None
        if isinstance(action, str):
            try:
                parsed = json.loads(action)
            except json.JSONDecodeError:
                return None, f"action is not valid JSON: {action!r}"
            if not isinstance(parsed, dict):
                return None, f"action JSON is not an object: {action!r}"
            return parsed, None
        return None, f"unsupported action type: {type(action).__name__}"

    def _move(self, direction):
        box, step = self.spec.box_size, self.spec.step_size
        if direction == "up":
            self.y = max(0, self.y - step)
        elif direction == "down":
            self.y = min(self.spec.height - box, self.y + step)
        elif direction == "left":
            self.x = max(0, self.x - step)
        elif direction == "right":
            self.x = min(self.spec.width - box, self.x + step)

    def _observation(self) -> Image.Image:
        return render_observation(self.canvas, self.x, self.y, self.spec,
                                  draw_border=self.draw_border)

    def _info(self, **extra):
        info = {
            "window": (self.x, self.y),
            "steps": self.steps,
            "termination_reason": self._reason.value,
            "label": self.label,
        }
        info.update(extra)
        return info

    # --- convenience ----------------------------------------------------------

    @property
    def window(self) -> tuple[int, int]:
        return (self.x, self.y)

    def mark_truncated(self):
        """Used by TimeLimit to record why an episode stopped."""
        self.done = True
        self._reason = TerminationReason.STEP_LIMIT
