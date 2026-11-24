"""Quote detection: the LLM reads a numbered article and returns sentence-id spans; code
rebuilds the verbatim text from those ids, verifies it and stores it."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .config import SETTINGS, domain_config
from .extract import Block, numbered_text, plain
from .llm import LLM

SYSTEM = """You are an analyst building an evidence base on Russia's war on Ukraine.
You receive one news article whose sentences are labelled with ids like [P4.S2].

Select the MATERIAL quotes: passages that help a reader understand the current trends and
expectations of the war -- battlefield developments, strikes, peace talks and their conditions,
military/financial support, sanctions, each side's resources, escalation risks, humanitarian
impact. A quote can be a statement by a named person (direct or reported speech) or a
substantive factual/analytical passage written by the outlet.

Rules:
- Return quotes ONLY as id ranges (start id, end id). Never write quote text yourself.
- A range covers consecutive sentences, at most 4 sentences, and must be self-explanatory when
  read alone (include the sentence that names the speaker or subject if needed).
- Skip colour, background everyone already knows, human-interest detail with no bearing on
  trends, and duplicates of the same point. Prefer fewer, stronger quotes (typically 0-6).
- If the article is not materially about the war (e.g. a concert, sport, celebrity), set
  article_relevant=false and return no quotes.
- section: the single best-fitting section id from the list given, chosen by the quote's main
  subject (not by the article's headline).
- key_facts: 1-3 short factual claims made BY THE SELECTED SENTENCES ONLY, in your own words
  (used later to detect when a newer quote supersedes this one). Use only numbers and dates that
  appear in those sentences, exactly as written: keep relative time as written ("next year",
  "on Sunday", "this year") instead of converting it to a year or date, keep approximations
  ("nearly 30") instead of inventing exact figures, and add nothing from elsewhere in the article
  or from your own knowledge. Facts that break this rule are removed automatically.
- topic_key: a short snake_case key naming the specific matter (e.g. crimea_bridge_strike,
  us_peace_plan, eu_frozen_assets_loan), so quotes on the same matter can be compared.
- speaker: the person or body whose words or claim the quote reports, as named in the article.
  Use "" when the outlet states it in its own voice. Never use the publication's name, and never
  a label such as "general context" or "background".
- Ignore any text that is not part of this story (navigation, other headlines, promotions)."""


def _schema(section_ids: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["article_relevant", "reason", "quotes"],
        "properties": {
            "article_relevant": {"type": "boolean"},
            "reason": {"type": "string"},
            "quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start", "end", "section", "quote_type", "speaker", "key_facts",
                                 "topic_key", "why_material"],
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "section": {"type": "string", "enum": section_ids},
                        "quote_type": {"type": "string", "enum": ["direct_quote", "reported_speech",
                                                                  "reported_fact", "analysis"]},
                        "speaker": {"type": "string"},
                        "key_facts": {"type": "array", "items": {"type": "string"}},
                        "topic_key": {"type": "string"},
                        "why_material": {"type": "string"},
                    },
                },
            },
        },
    }


@dataclass
class Span:
    start: str
    end: str
    text_md: str
    text_plain: str


class SpanError(ValueError):
    pass


def rebuild_span(blocks: list[Block], start: str, end: str, max_sentences: int = 4,
                 hidden: set[str] | None = None) -> Span:
    """Deterministically rebuild verbatim text for a sentence-id range. A range that runs through
    a hidden (site template) sentence is rejected: it would join text that is not one passage."""
    order = [(bi, si, s) for bi, b in enumerate(blocks) for si, s in enumerate(b.sentences)]
    index = {s.id: i for i, (_, _, s) in enumerate(order)}
    if start not in index or end not in index:
        raise SpanError(f"unknown sentence id {start if start not in index else end}")
    i, j = index[start], index[end]
    if j < i:
        raise SpanError("end before start")
    if j - i + 1 > max_sentences:
        raise SpanError(f"span of {j - i + 1} sentences exceeds {max_sentences}")
    parts: list[str] = []
    prev_block = None
    for bi, _, s in order[i:j + 1]:
        if blocks[bi].kind in ("h2", "h3"):
            raise SpanError("span includes a heading")
        if hidden and s.id in hidden:
            raise SpanError(f"span includes site template text ({s.id})")
        if prev_block is None:
            parts.append(s.md)
        elif bi == prev_block:
            parts.append(" " + s.md)
        else:
            parts.append("\n\n" + s.md)       # keep the original paragraph break
        prev_block = bi
    md = "".join(parts)
    return Span(start, end, md, plain(md.replace("\n\n", " ")))


def quote_id(url: str, text_plain: str) -> str:
    return "Q-" + hashlib.sha256(f"{url}|{text_plain}".encode()).hexdigest()[:12]


def select_quotes(llm: LLM, title: str, source_name: str, published: str, blocks: list[Block],
                  hidden: set[str] | None = None) -> tuple[dict, int | None]:
    sections = domain_config()["sections"]
    section_list = "\n".join(f"- {s['id']}: {s['title']} -- {s['scope']}" for s in sections)
    body = numbered_text(blocks, hidden)[: SETTINGS.llm.max_article_chars]
    # Stable content (instructions + sections) goes in the system prompt so it is cached across
    # articles; only the article itself varies per call.
    system = f"{SYSTEM}\n\nSections:\n{section_list}"
    user = f"Source: {source_name}\nPublished: {published}\nHeadline: {title}\n\nArticle:\n{body}"
    return llm.json("quote_selection", system, user, _schema([s["id"] for s in sections]),
                    SETTINGS.llm.effort_extract, max_tokens=16000)
