"""Provider settings that must remain compatible with the existing experiments."""

import sys
from types import SimpleNamespace
from types import ModuleType

from mnist_pro import backends


class _Interactions:
    def __init__(self):
        self.kwargs = None
        self.dump_modes = []

    def create(self, **kwargs):
        self.kwargs = kwargs
        owner = self

        class Step:
            def model_dump(self, *, mode=None):
                owner.dump_modes.append(mode)
                return {"type": "model_output", "content": "ok"}

        return SimpleNamespace(output_text=" done ", steps=[Step()])


def test_gemini_natural_turn_is_stateless_and_json_serialisable():
    interactions = _Interactions()
    backend = backends.GeminiBackend.__new__(backends.GeminiBackend)
    backend.model_name = "gemini-3.7-flash"
    backend.generation_config = None
    backend.client = SimpleNamespace(interactions=interactions)

    text, steps = backend.generate_turn("system", [{"type": "user_input"}])

    assert text == "done"
    assert steps == [{"type": "model_output", "content": "ok"}]
    assert interactions.kwargs["store"] is False
    assert interactions.dump_modes == ["json"]


def test_claude_uses_existing_experiment_output_budget():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=" done ")], usage=None
        )

    backend = backends.GCPBackend.__new__(backends.GCPBackend)
    backend.model_name = "claude-sonnet-5"
    backend.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    text, usage = backend.generate("system", [])

    assert text == "done"
    assert usage == {}
    assert seen["max_tokens"] == 8192


def test_openai_backend_uses_the_standard_openai_client(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend = backends.OpenAIBackend(model_name="gpt-5.6")

    assert isinstance(backend.client, Client)
    assert seen == {"api_key": "test-key"}


def test_gpt_5_6_routes_to_the_openai_client(monkeypatch):
    seen = {}

    class Backend:
        def __init__(self, model_name):
            seen["model_name"] = model_name

    monkeypatch.setattr(backends, "OpenAIBackend", Backend)

    backend = backends.get_backend("gpt-5.6")

    assert isinstance(backend, Backend)
    assert seen["model_name"] == "gpt-5.6"
