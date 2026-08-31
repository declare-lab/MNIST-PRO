"""One agent, parameterised by memory configuration, digit count and turn mode.

When the model cannot produce parseable JSON after `max_attempts`, this returns an
explicit invalid action rather than fabricating an answer of -1.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .specs import (
    MEMORY_SPECS,
    WARNING_TEMPLATE,
    MemorySpec,
    control_prompts,
    system_instruction,
)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
TURN_MODES = ("turn_based", "natural")

INVALID_ACTION = {"action": "invalid", "error": "JSONDecodeError"}


def extract_json(raw_response: str | None):
    """Extract and parse the first JSON object found in the raw response.

    The pattern is greedy, so a response containing a JSON example followed by an
    answer would match across both and fail.
    """
    if not raw_response:
        return None
    match = JSON_RE.search(raw_response)
    text = match.group(0) if match else raw_response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def encode_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class AgentConfig:
    """Defaults to multi-turn natural conversation.

    In natural conversation mode, observations stay in their own turns and the
    model's own outputs stay between them, which is how an agent actually experiences
    a sequence of glimpses. `turn_mode="turn_based"` re-renders a textual summary of
    past actions each turn instead.
    """

    memory: str = "textual_state"
    digits: int = 1
    horizon: int | None = None  # images retained; -1 or None = unbounded
    turn_mode: str = "natural"
    max_steps: int = 36
    max_attempts: int = 3

    def resolve(self) -> "AgentConfig":
        if self.memory not in MEMORY_SPECS:
            raise ValueError(
                f"unknown memory config {self.memory!r}; "
                f"expected one of {sorted(MEMORY_SPECS)}"
            )
        if self.turn_mode not in TURN_MODES:
            raise ValueError(f"unknown turn mode {self.turn_mode!r}")
        if self.horizon is None:
            # Natural conversation keeps the whole transcript by construction.
            self.horizon = (
                -1
                if self.turn_mode == "natural"
                else MEMORY_SPECS[self.memory].default_horizon
            )
        if self.turn_mode == "natural" and self.horizon != -1:
            raise ValueError(
                "natural conversation preserves the complete transcript; "
                "use horizon=-1"
            )
        return self

    @property
    def spec(self) -> MemorySpec:
        return MEMORY_SPECS[self.memory]

    @property
    def max_image_history(self) -> int | None:
        return None if self.horizon in (-1, None) else self.horizon


class GlimpseAgent:
    """The single agent implementation."""

    def __init__(self, backend, config: AgentConfig | None = None, **kwargs):
        self.backend = backend
        self.config = (config or AgentConfig(**kwargs)).resolve()
        self.system_instruction = system_instruction(self.config.digits)
        self.user_instruction = self.config.spec.user_instruction
        self.steps = 0
        self.image_history: list[str] = []
        self.full_history: list[dict] = []

    # --- state ---------------------------------------------------------------

    def reset(self):
        self.steps = 0
        self.image_history = []
        self.full_history = []

    def render_memory(self) -> str:
        """Replay the model's own prior outputs as text.

        Visual-buffer configurations return "" -- they rely purely on the image
        history and are shown no textual record of what they did.
        """
        if not self.config.spec.include_text_memory:
            return ""
        outputs = []
        for item in self.full_history:
            if item.get("type") != "model_output":
                continue
            content = item["content"]
            outputs.append(content[0]["text"] if isinstance(content, list) else content)
        if not outputs:
            return "This is the first turn. No previous actions."
        text = f"{self.config.spec.memory_prompt_prefix}:\n"
        for i, out in enumerate(outputs):
            text += f"Turn {i + 1}:\n{out}\n\n"
        return text.strip()

    def _image_parts(self, current_b64: str) -> list[dict]:
        self.image_history.append(current_b64)
        keep = self.config.max_image_history
        chosen = self.image_history if keep is None else self.image_history[-keep:]
        return [{"type": "image", "data": b, "mime_type": "image/png"} for b in chosen]

    # --- acting --------------------------------------------------------------

    def act(self, observation: Image.Image) -> tuple[dict, str, dict]:
        """Return `(action, raw_response, usage)`.

        The action is a dict, not a JSON string: the environment accepts either, and
        keeping it structured means the recorder logs a real object.
        """
        if self.config.turn_mode == "natural":
            return self._act_natural(observation)
        return self._act_turn_based(observation)

    def _act_turn_based(self, observation):
        self.steps += 1
        img_b64 = encode_png(observation)

        prompt_text = self.render_memory()
        if self.user_instruction:
            prompt_text = (
                f"{self.user_instruction}\n\n{prompt_text}"
                if prompt_text
                else self.user_instruction
            )
        if self.steps >= self.config.max_steps:
            prompt_text = f"{prompt_text}\n\n" + WARNING_TEMPLATE.format(
                self.config.spec.warning_example(self.config.digits)
            )

        contents = [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": prompt_text}]
                + self._image_parts(img_b64),
            }
        ]

        raw_response, usage = "", {}
        for _ in range(self.config.max_attempts):
            raw_response, usage = self._generate(contents)
            parsed = extract_json(raw_response)
            if parsed is not None:
                self._remember(prompt_text, raw_response)
                return parsed, raw_response, usage
        self._remember(prompt_text, raw_response)
        return dict(INVALID_ACTION), raw_response, usage

    def _act_natural(self, observation):
        """Tool-free interleaved conversation: observations stay in their own turns.

        The first user turn carries the instruction and one image; later turns carry
        only the new image. Model outputs remain in place between them.
        """
        self.steps += 1
        img_b64 = encode_png(observation)
        content = []
        if not self.full_history:
            content.append({"type": "text", "text": self.user_instruction})
        content.append({"type": "image", "data": img_b64, "mime_type": "image/png"})
        turn = {"type": "user_input", "content": content}
        request = self.full_history + [turn]

        raw_response, usage, steps = "", {}, []
        for _ in range(self.config.max_attempts):
            raw_response, usage, steps = self._generate_turn(request)
            parsed = extract_json(raw_response)
            if parsed is not None:
                self.full_history.append(turn)
                self.full_history.extend(steps)
                return parsed, raw_response, usage
        self.full_history.append(turn)
        self.full_history.extend(steps or [self._model_step(raw_response)])
        return dict(INVALID_ACTION), raw_response, usage

    # --- backend plumbing ----------------------------------------------------

    def _generate(self, contents):
        result = self.backend.generate(
            system_instruction=self.system_instruction, contents=contents
        )
        return _split_result(result)

    def _generate_turn(self, contents):
        generate_turn = getattr(self.backend, "generate_turn", None)
        if callable(generate_turn):
            raw, steps = generate_turn(
                system_instruction=self.system_instruction, contents=contents
            )
            return raw, {}, (steps or [self._model_step(raw)])
        raw, usage = self._generate(contents)
        return raw, usage, [self._model_step(raw)]

    @staticmethod
    def _model_step(raw_response):
        return {
            "type": "model_output",
            "content": [{"type": "text", "text": raw_response}],
        }

    def _remember(self, prompt_text, raw_response):
        self.full_history.append(
            {"type": "user_input", "content": [{"type": "text", "text": prompt_text}]}
        )
        self.full_history.append(self._model_step(raw_response))

    # --- control condition ---------------------------------------------------

    def predict_full_image(self, image: Image.Image) -> tuple[str, str, dict]:
        """Classify the fully unmasked canvas.

        Lives on the agent rather than in the eval driver, so the control and the
        main path cannot drift apart.
        """
        prompt_text, system = control_prompts(self.config.digits)
        contents = [
            {
                "type": "user_input",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image",
                        "data": encode_png(image),
                        "mime_type": "image/png",
                    },
                ],
            }
        ]
        try:
            raw, usage = _split_result(
                self.backend.generate(system_instruction=system, contents=contents)
            )
        except Exception as exc:  # recorded, never fatal
            return f"Error: {exc}", "-1", {}
        parsed = extract_json(raw)
        value = "-1"
        if parsed is not None:
            value = str(parsed.get("value", "-1")).strip()
            if self.config.digits == 1:
                try:
                    value = str(int(parsed.get("value", -1)))
                except (TypeError, ValueError):
                    value = "-1"
        return raw, value, usage


def _split_result(result):
    """Backends may return text, or (text, usage)."""
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], (result[1] or {})
    return result, {}
