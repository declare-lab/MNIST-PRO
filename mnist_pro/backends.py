"""Model backends.

Ported from the original `src/agent.py` with three changes:

* `generate` returns `(text, usage)` so token counts are recorded per step instead of
  being dug out of a stored response blob afterwards;
* exhausting the retries raises `BackendError`, which the runner records as a failed
  episode rather than letting it abort a whole sweep;
* the API key is read from the environment and never written to any artefact.

Routing is unchanged, so `--model` values from existing scripts still resolve to the
same provider.
"""

from __future__ import annotations

import base64
import os
import time

DEFAULT_MAX_RETRIES = 8
DEFAULT_BACKOFF = 2
DEFAULT_INITIAL_DELAY = 2
MAX_DELAY = 60


class BackendError(RuntimeError):
    """All retries exhausted. Recorded per episode; never aborts a sweep."""


def _retry(fn, label, max_retries=DEFAULT_MAX_RETRIES):
    delay = DEFAULT_INITIAL_DELAY
    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            last = exc
            print(f"[{label}] attempt {attempt + 1}/{max_retries} failed: {exc}. "
                  f"Retrying in {min(delay, MAX_DELAY)}s...", flush=True)
            time.sleep(min(delay, MAX_DELAY))
            delay *= DEFAULT_BACKOFF
    raise BackendError(f"[{label}] all {max_retries} attempts failed: {last}")


def _usage_from(obj) -> dict:
    usage = getattr(obj, "usage", None)
    if usage is None:
        return {}
    for attr in ("model_dump",):
        if hasattr(usage, attr):
            try:
                return {k: v for k, v in getattr(usage, attr)().items()
                        if isinstance(v, (int, float))}
            except Exception:
                pass
    return {k: getattr(usage, k) for k in dir(usage)
            if not k.startswith("_") and isinstance(getattr(usage, k, None), (int, float))}


class GeminiBackend:
    """Google Gemini via `google-genai`. No generation_config is sent, matching the
    published runs: temperature and thinking level are provider defaults."""

    def __init__(self, model_name="gemini-3.6-flash", generation_config=None):
        from google import genai
        if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
            raise BackendError("GEMINI_API_KEY is not set in the environment")
        self.model_name = model_name
        self.generation_config = generation_config
        self.client = genai.Client()

    def generate(self, system_instruction, contents):
        def call():
            kwargs = {"model": self.model_name, "input": contents,
                      "system_instruction": system_instruction}
            if self.generation_config:
                kwargs["generation_config"] = self.generation_config
            interaction = self.client.interactions.create(**kwargs)
            return interaction.output_text.strip(), _usage_from(interaction)
        return _retry(call, "GeminiBackend")

    def generate_turn(self, system_instruction, contents):
        """Return the model's own steps so a natural conversation can replay them."""
        def call():
            kwargs = {"model": self.model_name, "input": contents,
                      "system_instruction": system_instruction}
            if self.generation_config:
                kwargs["generation_config"] = self.generation_config
            interaction = self.client.interactions.create(**kwargs)
            steps = [s.model_dump() for s in (interaction.steps or [])]
            return interaction.output_text.strip(), steps
        return _retry(call, "GeminiBackend.generate_turn")


class OpenRouterBackend:
    def __init__(self, model_name="qwen/qwen3.8-27b"):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ.get("OPENROUTER_API_KEY"))

    def generate(self, system_instruction, contents):
        messages = to_openai_messages(system_instruction, contents)

        def call():
            resp = self.client.chat.completions.create(model=self.model_name,
                                                       messages=messages)
            text = ""
            if resp.choices:
                text = (resp.choices[0].message.content or "").strip()
            return text, _usage_from(resp)
        return _retry(call, "OpenRouterBackend")


class GCPBackend:
    def __init__(self, model_name="claude-sonnet-5",
                 project_id="gcp-eee-declare-48e6", region="global"):
        from anthropic import AnthropicVertex
        self.model_name = model_name
        self.client = AnthropicVertex(project_id=project_id, region=region)

    def generate(self, system_instruction, contents):
        messages = to_anthropic_messages(contents)

        def call():
            resp = self.client.messages.create(
                model=self.model_name, max_tokens=4096,
                system=system_instruction, messages=messages)
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return text.strip(), _usage_from(resp)
        return _retry(call, "GCPBackend")


class OpenAIBackend:
    def __init__(self, model_name="gpt-5.6"):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, system_instruction, contents):
        messages = to_openai_messages(system_instruction, contents)

        def call():
            resp = self.client.chat.completions.create(model=self.model_name,
                                                       messages=messages)
            text = ""
            if resp.choices:
                text = (resp.choices[0].message.content or "").strip()
            return text, _usage_from(resp)
        return _retry(call, "OpenAIBackend")


# --- content translation ------------------------------------------------------

def to_openai_messages(system_instruction, contents):
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    for item in contents:
        role = "assistant" if item.get("type") == "model_output" else "user"
        value = item.get("content")
        if isinstance(value, str):
            messages.append({"role": role, "content": value})
            continue
        parts = []
        for part in value or []:
            if isinstance(part, str):
                parts.append({"type": "text", "text": part})
            elif part.get("type") == "image":
                data = part.get("data")
                b64 = base64.b64encode(data).decode() if isinstance(data, bytes) else data
                mime = part.get("mime_type", "image/png")
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{mime};base64,{b64}"}})
            else:
                parts.append({"type": "text", "text": part.get("text", "")})
        messages.append({"role": role, "content": parts})
    return messages


def to_anthropic_messages(contents):
    messages = []
    for item in contents:
        role = "assistant" if item.get("type") == "model_output" else "user"
        value = item.get("content")
        if isinstance(value, str):
            messages.append({"role": role, "content": value})
            continue
        parts = []
        for part in value or []:
            if isinstance(part, str):
                parts.append({"type": "text", "text": part})
            elif part.get("type") == "image":
                data = part.get("data")
                b64 = base64.b64encode(data).decode() if isinstance(data, bytes) else data
                parts.append({"type": "image", "source": {
                    "type": "base64", "media_type": part.get("mime_type", "image/png"),
                    "data": b64}})
            else:
                parts.append({"type": "text", "text": part.get("text", "")})
        messages.append({"role": role, "content": parts})
    return messages


def get_backend(model_name: str | None, **kwargs):
    """Unchanged routing, so existing --model values resolve identically."""
    if not model_name:
        return GeminiBackend(**kwargs)
    lowered = model_name.lower()
    if "claude" in lowered or "vertex" in lowered or "gcp" in lowered:
        return GCPBackend(model_name=model_name)
    if model_name == "gpt-5.6":
        return OpenAIBackend(model_name="gpt-5.6-terra-fast-test")
    if model_name == "glm-4.6v":
        return OpenRouterBackend(model_name="z-ai/glm-4.6v")
    if model_name == "qwen3.8-27b":
        return OpenRouterBackend(model_name="qwen/qwen3.8-27b")
    if "qwen" in lowered or "openrouter" in lowered or "/" in model_name:
        return OpenRouterBackend(model_name=model_name)
    return GeminiBackend(model_name=model_name, **kwargs)
