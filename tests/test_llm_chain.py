"""LLM fallback chain: local Ollama first, then hosted providers. Offline, with fake providers."""
from __future__ import annotations

import pytest

from mqm import llm as L


class Fake:
    """A provider that replays a script: a dict answer, or an exception to raise."""

    def __init__(self, name, *script):
        self.name, self.script, self.calls = name, list(script), 0
        self.wait_on_rate_limit = True

    def json(self, purpose, system, user, schema, effort, max_tokens=16000):
        self.calls += 1
        step = self.script.pop(0) if self.script else {"by": self.name}
        if isinstance(step, Exception):
            raise step
        return step, None


class FakeOllama(L.OllamaLLM, Fake):
    def __init__(self, up=True, *script):
        Fake.__init__(self, "ollama", *script)
        self.up, self.model, self.base_url, self.probes = up, "qwen3:8b", "http://127.0.0.1:11434", 0

    def available(self):
        self.probes += 1
        return (True, "") if self.up else (False, "Ollama not reachable")

    json = Fake.json


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(L, "log", lambda *a, **k: None)


def chain(monkeypatch, **providers):
    """Build a FallbackLLM whose providers are the given fakes (or exceptions = 'missing key')."""
    def factory(v):
        def make():
            if isinstance(v, Exception):
                raise v
            return v
        return make
    monkeypatch.setattr(L, "PROVIDERS", {k: factory(v) for k, v in providers.items()})
    return L.FallbackLLM(",".join(providers))


def ask(c):
    return c.json("quote_selection", "sys", "user", {}, "low")[0]


def test_default_order_is_ollama_first():
    assert L.DEFAULT_CHAIN.split(",")[0] == "ollama"
    assert L.DEFAULT_CHAIN == "ollama,groq,huggingface,anthropic"


def test_ollama_answers_first_when_running(monkeypatch):
    o, g = FakeOllama(True), Fake("groq")
    c = chain(monkeypatch, ollama=o, groq=g)
    assert ask(c) == {"by": "ollama"}
    assert g.calls == 0


def test_ollama_not_running_falls_back_and_is_not_probed_again_at_once(monkeypatch):
    o, g = FakeOllama(False), Fake("groq")
    c = chain(monkeypatch, ollama=o, groq=g)
    assert ask(c) == {"by": "groq"}
    assert ask(c) == {"by": "groq"}
    assert o.calls == 0 and o.probes == 1          # recheck only after OLLAMA_RECHECK_S


def test_failed_call_is_retried_on_next_provider(monkeypatch):
    o, g = FakeOllama(True, L.LLMError("no valid JSON after 3 attempts")), Fake("groq")
    c = chain(monkeypatch, ollama=o, groq=g)
    assert ask(c) == {"by": "groq"}                 # same call answered by the fallback
    assert ask(c) == {"by": "ollama"}               # a one-off failure does not disable Ollama


def test_rate_limited_provider_is_paused_not_waited_for(monkeypatch):
    g, h = Fake("groq", L.LLMRateLimited("Groq: rate limited", 60)), Fake("huggingface")
    c = chain(monkeypatch, groq=g, huggingface=h)
    assert ask(c) == {"by": "huggingface"}
    assert ask(c) == {"by": "huggingface"} and g.calls == 1   # still cooling down


def test_bad_key_or_no_credits_disables_provider_for_session(monkeypatch):
    g = Fake("groq", L.LLMError("Groq authentication failed: check GROQ_API_KEY"))
    h = Fake("huggingface", L.LLMError("Hugging Face: inference credits exhausted"))
    a = Fake("anthropic")
    c = chain(monkeypatch, groq=g, huggingface=h, anthropic=a)
    assert ask(c) == {"by": "anthropic"}
    assert ask(c) == {"by": "anthropic"}
    assert g.calls == 1 and h.calls == 1


def test_providers_without_keys_are_skipped(monkeypatch):
    c = chain(monkeypatch, ollama=FakeOllama(True), groq=L.LLMError("GROQ_API_KEY is not set"),
              anthropic=L.LLMError("ANTHROPIC_API_KEY is not set"))
    assert c.providers == ["ollama"]


def test_chain_puts_providers_in_no_wait_mode(monkeypatch):
    g = Fake("groq")
    chain(monkeypatch, groq=g)
    assert g.wait_on_rate_limit is False


def test_error_when_nothing_is_available(monkeypatch):
    with pytest.raises(L.LLMError, match="no LLM provider available"):
        chain(monkeypatch, groq=L.LLMError("GROQ_API_KEY is not set"))
    c = chain(monkeypatch, ollama=FakeOllama(False))
    with pytest.raises(L.LLMError, match="all LLM providers failed or unavailable"):
        ask(c)


def test_open_model_hands_over_on_429_inside_a_chain():
    class Resp:
        status_code, headers = 429, {"retry-after": "12"}

    class Boom(Exception):
        response = Resp()

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise Boom("429 Too Many Requests")

    m = L.OpenModelLLM.__new__(L.OpenModelLLM)
    m.model, m.client, m.wait_on_rate_limit = "x", Client(), False
    import mqm.llm
    mqm.llm._record = lambda *a, **k: None
    with pytest.raises(L.LLMRateLimited) as e:
        m.json("p", "s", "u", {}, "low")
    assert e.value.wait == 12.0


@pytest.mark.parametrize("tags,ok,why", [
    ({"models": [{"name": "qwen3:8b"}]}, True, ""),
    ({"models": [{"name": "llama3:8b"}]}, False, "ollama pull qwen3:8b"),
    (None, False, "not reachable"),
])
def test_ollama_availability_probe(monkeypatch, tags, ok, why):
    import httpx

    def fake_get(url, timeout):
        assert url == "http://127.0.0.1:11434/api/tags" and timeout == 2
        if tags is None:
            raise httpx.ConnectError("refused")
        return type("R", (), {"json": lambda self: tags})()

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.delenv("MQM_OLLAMA_URL", raising=False)
    monkeypatch.delenv("MQM_OLLAMA_MODEL", raising=False)
    got_ok, got_why = L.OllamaLLM().available()
    assert got_ok is ok and why in got_why


def test_default_llm_is_the_chain_and_single_provider_can_be_forced(monkeypatch):
    monkeypatch.setattr(L, "_default", None)
    monkeypatch.delenv("MQM_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(L, "FallbackLLM", lambda: "chain")
    assert L.default_llm() == "chain"
    monkeypatch.setattr(L, "_default", None)
    monkeypatch.setenv("MQM_LLM_PROVIDER", "ollama")
    monkeypatch.setattr(L, "PROVIDERS", {"ollama": lambda: "ollama-only"})
    assert L.default_llm() == "ollama-only"
