"""Deterministic article relevance rules (no LLM). Same input -> same decision, with the
matched rule recorded on the article row for audit."""
from __future__ import annotations

import re
from functools import lru_cache

from .config import domain_config


@lru_cache(maxsize=1)
def _patterns() -> dict[str, list[re.Pattern]]:
    rel = domain_config()["relevance"]
    return {k: [re.compile(p, re.I) for p in v] for k, v in rel.items()}


def _hits(group: str, text: str) -> list[str]:
    return [p.pattern for p in _patterns()[group] if p.search(text)]


def slug_text(url: str) -> str:
    path = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[-_]+", " ", path)


def classify(title: str, standfirst: str = "", url: str = "", tagged: bool = False) -> str | None:
    """Return the name of the rule that made the article relevant, or None."""
    text = " ".join([title or "", standfirst or "", slug_text(url) if url else ""])
    if _hits("exclude", text):
        return None
    if tagged:
        return "topic_tag"
    core = _hits("core", text)
    if core:
        return f"core:{core[0]}"
    ru, ctx = _hits("russia", text), _hits("war_context", text)
    if ru and ctx:
        return f"russia+context:{ru[0]}&{ctx[0]}"
    return None
