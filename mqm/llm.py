"""Thin LLM layer: every call is structured (JSON schema), logged, costed and stored.

Only three steps are probabilistic, and each returns ids/labels rather than free text that
could leak into the verbatim data:
  1. quote span selection + section labelling   (quotes.py)
  2. obsolescence / supersession judgement       (report.py)
  3. 1-2 sentence section summaries              (report.py)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Protocol

import anthropic

from .config import SETTINGS
from .db import dumps, session
from .logs import current_run_id, log, now_iso

# USD per million tokens: (input, output, cache read)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


class LLMError(Exception):
    pass


class LLMRateLimited(LLMError):
    """Provider asked us to slow down; `wait` is its Retry-After in seconds."""

    def __init__(self, message: str, wait: float):
        super().__init__(message)
        self.wait = wait


class LLM(Protocol):
    def json(self, purpose: str, system: str, user: str, schema: dict, effort: str,
             max_tokens: int = 16000) -> tuple[dict, int | None]: ...


def _record(purpose: str, model: str, effort: str, req_sha: str, usage: Any = None,
            stop: str | None = None, response: Any = None, error: str | None = None) -> int:
    # Anthropic usage: input_tokens/output_tokens; OpenAI-style (HF): prompt_tokens/completion_tokens
    inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    pin, pout, pcache = PRICES.get(model, (0, 0, 0))
    cost = (inp * pin + out * pout + cached * pcache) / 1e6
    with session() as c:
        cur = c.execute(
            "INSERT INTO llm_calls(run_id,purpose,model,effort,input_tokens,output_tokens,cache_read_tokens,"
            "cost_usd,stop_reason,request_sha256,response_json,error,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (current_run_id(), purpose, model, effort, inp, out, cached, round(cost, 6), stop, req_sha,
             dumps(response) if response is not None else None, error, now_iso()),
        )
        call_id = cur.lastrowid
    log("llm.call", purpose=purpose, model=model, effort=effort, input_tokens=inp, output_tokens=out,
        cache_read=cached, cost_usd=round(cost, 4), stop_reason=stop, call_id=call_id, error=error)
    return call_id


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set: add it to .env (console.anthropic.com)")
        self.model = model or SETTINGS.llm.model
        self.client = anthropic.Anthropic(max_retries=4, timeout=600)

    def json(self, purpose: str, system: str, user: str, schema: dict, effort: str,
             max_tokens: int = 16000) -> tuple[dict, int | None]:
        req_sha = hashlib.sha256((self.model + system + user + json.dumps(schema)).encode()).hexdigest()
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                # The system prompt is identical for every call of a given purpose -> cache it.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort,
                               "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            _record(purpose, self.model, effort, req_sha, error=f"auth: {e}")
            raise LLMError("Anthropic authentication failed: set ANTHROPIC_API_KEY in .env") from e
        except anthropic.BadRequestError as e:
            _record(purpose, self.model, effort, req_sha, error=f"bad_request: {e}")
            raise LLMError(f"bad request: {e}") from e
        except anthropic.RateLimitError as e:
            _record(purpose, self.model, effort, req_sha, error=f"rate_limited: {e}")
            raise LLMError("rate limited after retries") from e
        except anthropic.APIStatusError as e:
            _record(purpose, self.model, effort, req_sha, error=f"status {e.status_code}: {e}")
            raise LLMError(f"API error {e.status_code}") from e
        except anthropic.APIConnectionError as e:
            _record(purpose, self.model, effort, req_sha, error=f"connection: {e}")
            raise LLMError("cannot reach the API") from e

        if resp.stop_reason == "refusal":
            _record(purpose, self.model, effort, req_sha, resp.usage, "refusal", error="refused")
            raise LLMError("model declined the request")
        if resp.stop_reason == "max_tokens":
            _record(purpose, self.model, effort, req_sha, resp.usage, "max_tokens", error="truncated")
            raise LLMError("response truncated (max_tokens)")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            _record(purpose, self.model, effort, req_sha, resp.usage, resp.stop_reason, text, "invalid json")
            raise LLMError("model returned invalid JSON") from e
        call_id = _record(purpose, self.model, effort, req_sha, resp.usage, resp.stop_reason, data)
        return data, call_id


def _strip_to_json(text: str) -> str:
    """Open models sometimes wrap JSON in <think> blocks or ``` fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    return text[start:]


def schema_errors(data: Any, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-schema check (type, required, enum, additionalProperties, items) so that
    providers without enforced structured output are still validated deterministically."""
    errs: list[str] = []
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}
    if t in types and (not isinstance(data, types[t]) or (t == "integer" and isinstance(data, bool))):
        return [f"{path}: expected {t}"]
    if "enum" in schema and data not in schema["enum"]:
        errs.append(f"{path}: {data!r} not in {schema['enum']}")
    if t == "object":
        for k in schema.get("required", []):
            if k not in data:
                errs.append(f"{path}.{k}: missing")
        props = schema.get("properties", {})
        for k, v in data.items():
            if k in props:
                errs += schema_errors(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}.{k}: unexpected")
    if t == "array" and "items" in schema:
        for i, v in enumerate(data):
            errs += schema_errors(v, schema["items"], f"{path}[{i}]")
    return errs


class OpenModelLLM:
    """Open-weight models through an OpenAI-compatible chat API (Hugging Face router or Groq).
    Output is always re-validated against the schema in code, with a self-repair retry."""

    label = "open model"
    key_env = ""
    wait_on_rate_limit = True     # False inside a fallback chain: hand over to the next provider instead

    def __init__(self, model: str, client):
        self.model, self.client = model, client
        # Optional USD per 1M tokens for cost tracking (provider prices vary).
        PRICES.setdefault(self.model, (float(os.environ.get("MQM_PRICE_IN", 0)),
                                       float(os.environ.get("MQM_PRICE_OUT", 0)), 0.0))

    def json(self, purpose: str, system: str, user: str, schema: dict, effort: str,
             max_tokens: int = 16000) -> tuple[dict, int | None]:
        req_sha = hashlib.sha256((self.model + system + user + json.dumps(schema)).encode()).hexdigest()
        sys_prompt = (f"{system}\n\nReply with ONE JSON object only (no prose, no code fences) that "
                      f"matches this JSON schema:\n{json.dumps(schema)}")
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
        last_err = ""
        attempt = rate_waits = 0
        while attempt < 3:
            attempt += 1
            kwargs = {"model": self.model, "messages": messages, "max_tokens": min(max_tokens, 8192),
                      "temperature": 0}
            if attempt == 1:
                kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": purpose, "schema": schema, "strict": True}}
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 - provider SDK raises various HTTP errors
                last_err = f"{type(e).__name__}: {e}"
                status = getattr(getattr(e, "response", None), "status_code", None)
                _record(purpose, self.model, effort, req_sha, error=f"attempt {attempt}: {last_err[:500]}")
                if status in (401, 403):
                    raise LLMError(f"{self.label} authentication failed: check {self.key_env} ({last_err})") from e
                if status == 402:
                    raise LLMError(f"{self.label}: inference credits exhausted") from e
                if status == 429 and not self.wait_on_rate_limit:
                    hdr = getattr(getattr(e, "response", None), "headers", {}) or {}
                    wait = str(hdr.get("retry-after", ""))
                    raise LLMRateLimited(f"{self.label}: rate limited",
                                         float(wait) if wait.replace(".", "").isdigit() else 30.0) from e
                if status == 429 and rate_waits < 8:
                    # free tiers limit tokens per minute: wait and retry without using an attempt
                    hdr = getattr(getattr(e, "response", None), "headers", {}) or {}
                    wait = hdr.get("retry-after", "")
                    rate_waits += 1
                    attempt -= 1
                    time.sleep(min(float(wait) if wait.replace(".", "").isdigit() else 20, 90))
                    continue
                if status == 400 and "response_format" in kwargs:
                    continue  # provider/model lacks json_schema mode: next attempt uses prompt-only JSON
                time.sleep(2 * attempt)
                continue
            text = resp.choices[0].message.content or ""
            finish = resp.choices[0].finish_reason
            try:
                data = json.loads(_strip_to_json(text))
                problems = schema_errors(data, schema)
            except json.JSONDecodeError as e:
                data, problems = None, [f"invalid JSON: {e}"]
            if not problems:
                call_id = _record(purpose, self.model, effort, req_sha, resp.usage, finish, data)
                return data, call_id
            last_err = "; ".join(problems[:5])
            _record(purpose, self.model, effort, req_sha, resp.usage, finish, text[:2000],
                    error=f"attempt {attempt}: {last_err}")
            # self-repair: show the model exactly what failed validation
            messages = messages[:2] + [{"role": "assistant", "content": text[:6000]},
                                       {"role": "user", "content": f"Your JSON failed validation: {last_err}. "
                                                                   "Return the corrected JSON object only."}]
        raise LLMError(f"no valid JSON after 3 attempts: {last_err}")


class HuggingFaceLLM(OpenModelLLM):
    label, key_env = "Hugging Face", "HF_TOKEN"

    def __init__(self, model: str | None = None, provider: str | None = None):
        from huggingface_hub import InferenceClient
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise LLMError("HF_TOKEN is not set: add it to .env (huggingface.co/settings/tokens)")
        client = InferenceClient(provider=provider or os.environ.get("MQM_HF_PROVIDER", "auto"),
                                 api_key=token, timeout=300)
        super().__init__(model or os.environ.get("MQM_HF_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"), client)


class GroqLLM(OpenModelLLM):
    label, key_env = "Groq", "GROQ_API_KEY"

    def __init__(self, model: str | None = None):
        from huggingface_hub import InferenceClient   # OpenAI-compatible client, pointed at Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY is not set: add it to .env (console.groq.com/keys)")
        client = InferenceClient(base_url="https://api.groq.com/openai/v1", api_key=key, timeout=300)
        super().__init__(model or os.environ.get("MQM_GROQ_MODEL", "openai/gpt-oss-120b"), client)


class OllamaLLM(OpenModelLLM):
    """A local open-weight model served by Ollama: no API key, no rate limit, no cost."""

    label, key_env = "Ollama (local)", "MQM_OLLAMA_URL"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        from huggingface_hub import InferenceClient   # OpenAI-compatible client, pointed at Ollama
        self.base_url = (base_url or os.environ.get("MQM_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        client = InferenceClient(base_url=f"{self.base_url}/v1", api_key="ollama",
                                 timeout=float(os.environ.get("MQM_OLLAMA_TIMEOUT", "900")))
        super().__init__(model or os.environ.get("MQM_OLLAMA_MODEL", "qwen3:8b"), client)

    def available(self) -> tuple[bool, str]:
        """Quick check (2 s) that the Ollama server is up and the model is pulled."""
        import httpx
        try:
            tags = httpx.get(f"{self.base_url}/api/tags", timeout=2).json()
        except Exception as e:  # noqa: BLE001 - not running / not reachable
            return False, f"Ollama not reachable at {self.base_url} ({type(e).__name__})"
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            return False, f"model {self.model} is not pulled (run: ollama pull {self.model})"
        return True, ""


PROVIDERS = {"ollama": OllamaLLM, "groq": GroqLLM, "huggingface": HuggingFaceLLM, "anthropic": AnthropicLLM}
DEFAULT_CHAIN = "ollama,groq,huggingface,anthropic"


class FallbackLLM:
    """Tries providers in order (default: local Ollama first, then Groq, Hugging Face, Claude).

    - A provider without a key, or an Ollama server that is not running, is skipped.
    - Rate limited (429): the provider is paused for its Retry-After and the next one is used at once.
    - Bad key / no credits: the provider is switched off for the rest of the session.
    - Any other failure on a call (e.g. no valid JSON): the next provider gets the same call.
    Every attempt is still recorded in `llm_calls`, so the dashboard shows which model answered."""

    OLLAMA_RECHECK_S = 300

    def __init__(self, chain: str | None = None):
        self.entries: list[dict] = []
        for name in (chain or os.environ.get("MQM_LLM_CHAIN", DEFAULT_CHAIN)).split(","):
            name = name.strip().lower()
            if name not in PROVIDERS:
                log("llm.chain_unknown_provider", "warning", provider=name)
                continue
            try:
                inst = PROVIDERS[name]()
            except Exception as e:  # noqa: BLE001 - missing key / SDK error: skip this provider
                log("llm.chain_skip", "info", provider=name, reason=str(e)[:200])
                continue
            if hasattr(inst, "wait_on_rate_limit"):
                inst.wait_on_rate_limit = False
            self.entries.append({"name": name, "llm": inst, "until": 0.0})
        if not self.entries:
            raise LLMError("no LLM provider available: start Ollama, or set GROQ_API_KEY / HF_TOKEN / "
                           "ANTHROPIC_API_KEY in .env")
        log("llm.chain", providers=[e["name"] for e in self.entries])

    @property
    def providers(self) -> list[str]:
        return [e["name"] for e in self.entries]

    def _ready(self, entry: dict) -> bool:
        if time.time() < entry["until"]:
            return False
        # local providers expose a cheap availability probe (Ollama: is the server up, is the model pulled)
        if hasattr(entry["llm"], "available") and time.time() - entry.get("checked", 0.0) > 60:
            ok, why = entry["llm"].available()
            if not ok:
                entry["until"] = time.time() + self.OLLAMA_RECHECK_S
                log("llm.provider_unavailable", "warning", provider=entry["name"], reason=why)
                return False
            entry["checked"] = time.time()
        return True

    def json(self, purpose: str, system: str, user: str, schema: dict, effort: str,
             max_tokens: int = 16000) -> tuple[dict, int | None]:
        errors: list[str] = []
        for entry in self.entries:
            if not self._ready(entry):
                continue
            try:
                return entry["llm"].json(purpose, system, user, schema, effort, max_tokens)
            except LLMRateLimited as e:
                entry["until"] = time.time() + e.wait
                log("llm.rate_limited", "info", provider=entry["name"], cooldown_s=round(e.wait, 1), purpose=purpose)
                errors.append(f"{entry['name']}: rate limited")
            except LLMError as e:
                msg = str(e)
                if "authentication" in msg or "credits" in msg:
                    entry["until"] = float("inf")          # will not fix itself this session
                log("llm.provider_failed", "warning", provider=entry["name"], error=msg[:300], purpose=purpose)
                errors.append(f"{entry['name']}: {msg[:120]}")
        raise LLMError("all LLM providers failed or unavailable: " + ("; ".join(errors) or "none ready"))


_default: LLM | None = None


def default_llm() -> LLM:
    """MQM_LLM_PROVIDER=auto (default) uses the fallback chain; a provider name uses that one only."""
    global _default
    if _default is None:
        provider = os.environ.get("MQM_LLM_PROVIDER", "auto").lower()
        _default = PROVIDERS[provider]() if provider in PROVIDERS else FallbackLLM()
    return _default


def set_default_llm(llm: LLM | None) -> None:
    """Used by tests to inject a scripted LLM."""
    global _default
    _default = llm
