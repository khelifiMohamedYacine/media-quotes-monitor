"""Deterministic grounding of the fields the LLM writes itself (speaker, key facts) and detection
of site template text (navigation, feeds, newsletter boxes) repeated across a source's pages.

The quote text is already guaranteed verbatim by verify.py. These rules cover what verbatim
matching cannot: an LLM-written field that is not supported by the quote, and text that is on the
page but does not belong to the story. Everything here is rule-based, so an auditor can re-run it
and get the same result; every change is recorded on the quote with the reason.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from .extract import Block, norm_for_match

# A sentence that appears on at least this many pages of the same source is part of the site's
# template (feeds, promos, boilerplate background), not of a particular story. Measured on real
# data: legitimate cross-story reuse is rare at 3+ pages, while feeds and promos repeat on most pages.
TEMPLATE_MIN_PAGES = 3
MIN_TEMPLATE_CHARS = 25

_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "a dozen": "12", "dozen": "12", "half": "50", "a third": "33",
    "two-and-a-half": "2.5", "two and a half": "2.5", "hour-and-a-half": "1.5", "an hour and a half": "1.5",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
}
# Words that are also ordinary English ("an exemplary one", "second-largest", "half the city"):
# they count as numbers when reading the quote, but are not treated as figures inside a key fact.
_AMBIGUOUS_WORDS = {"one", "first", "second", "half", "third"}
# Labels an LLM sometimes puts in the speaker field when nobody is speaking (whole label only).
_NON_SPEAKERS = re.compile(r"(the\s+)?(general(\s+(context|background|information))?|background(\s+information)?|"
                           r"context|n/?a|none|unknown|narrator|author|article|outlet|editorial|reporter|report)",
                           re.I)
_SPEAKER_STOP = {"the", "a", "an", "of", "and", "for", "in", "on", "at", "to", "s", "mr", "mrs", "ms", "dr",
                 "said", "says", "official", "officials", "spokesperson", "spokesman", "spokeswoman"}


# ----------------------------------------------------------------------------- numbers
def numbers_in(text: str, include_ambiguous: bool = True) -> set[str]:
    """Numbers written in digits or words, normalised: '1,600' -> '1600', 'nine' -> '9', '4.5%' -> '4.5'."""
    t = norm_for_match(text)
    # thousands separators only in groups of three ("1,600"), so "2, 3, 4" stays three numbers
    out = {m.replace(",", "") for m in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", t)}
    for word, digit in _NUMBER_WORDS.items():
        if not include_ambiguous and word in _AMBIGUOUS_WORDS:
            continue
        if re.search(rf"\b{re.escape(word)}\b", t):
            out.add(digit)
    # 'Nov. 24' style ordinals and '24th' -> 24 are already covered by the digit pattern
    return {n for n in out if n}


def ground_key_facts(facts: list[str], quote_plain: str) -> tuple[list[str], list[dict]]:
    """Keep a key fact only if every number in it (counts, amounts, dates, years) is in the quote.

    Key facts are the model's paraphrase of the quote and are later used to compare quotes, so a
    number the quote does not contain - a converted 'next year', a rounded figure, a figure from
    elsewhere on the page or from the model's own knowledge - must not be stored as a fact."""
    q_nums = numbers_in(quote_plain)
    kept, removed = [], []
    for f in facts:
        f = (f or "").strip()
        if not f:
            continue
        missing = sorted(numbers_in(f, include_ambiguous=False) - q_nums)
        if missing:
            removed.append({"field": "key_fact", "value": f,
                            "reason": f"number(s) {', '.join(missing)} not in the quote"})
        else:
            kept.append(f)
    return kept, removed


# ----------------------------------------------------------------------------- speaker
def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", norm_for_match(s)) if t not in _SPEAKER_STOP]


def ground_speaker(speaker: str, quote_plain: str, article_text: str, source: dict) -> tuple[str, list[dict]]:
    """Blank a speaker that is the publication itself, a placeholder label, or a name/role that
    appears nowhere in the article. The outlet's own reporting has no speaker."""
    sp = (speaker or "").strip()
    if not sp:
        return "", []
    outlet = {norm_for_match(source.get("name", "")),
              norm_for_match(re.sub(r"^the\s+", "", source.get("name", ""), flags=re.I)),
              norm_for_match(source.get("domain", "").split(".")[0])}
    sp_norm = norm_for_match(re.sub(r"^the\s+", "", sp, flags=re.I))
    if sp_norm in outlet or norm_for_match(sp) in outlet:
        return "", [{"field": "speaker", "value": sp, "reason": "the publication itself is not a speaker"}]
    if _NON_SPEAKERS.fullmatch(sp.strip(" .")):
        return "", [{"field": "speaker", "value": sp, "reason": "placeholder label, not a speaker"}]
    hay = norm_for_match(quote_plain + " " + article_text)
    toks = _tokens(sp)
    if toks and not any(re.search(rf"\b{re.escape(t)}", hay) for t in toks):
        return "", [{"field": "speaker", "value": sp, "reason": "name/role not found in the article"}]
    return sp, []


def ground_fields(q: dict, article_text: str, source: dict) -> list[dict]:
    """Apply all field rules to a quote dict in place; return the list of changes made."""
    facts = q.get("key_facts")
    if facts is None:
        facts = json.loads(q.get("key_facts_json") or "[]")
    kept, changes = ground_key_facts(facts, q.get("quote_plain", ""))
    q["key_facts_json"] = json.dumps(kept, ensure_ascii=False)
    q.pop("key_facts", None)
    q["speaker"], sp_changes = ground_speaker(q.get("speaker", ""), q.get("quote_plain", ""), article_text, source)
    return changes + sp_changes


# ----------------------------------------------------------------------------- site template
class SiteTemplate:
    """Which sentences of a source recur across its pages (i.e. belong to the site, not a story)."""

    def __init__(self, pages: dict[int, list[Block]], min_pages: int = TEMPLATE_MIN_PAGES):
        self.min_pages = min_pages
        self._pages: dict[str, set[int]] = defaultdict(set)
        for article_id, blocks in pages.items():
            for b in blocks:
                for s in b.sentences:
                    t = norm_for_match(s.md)
                    if len(t) >= MIN_TEMPLATE_CHARS:
                        self._pages[t].add(article_id)

    def add(self, article_id: int, blocks: list[Block]) -> None:
        for b in blocks:
            for s in b.sentences:
                t = norm_for_match(s.md)
                if len(t) >= MIN_TEMPLATE_CHARS:
                    self._pages[t].add(article_id)

    def is_template(self, sentence_md: str, article_id: int | None = None) -> bool:
        t = norm_for_match(sentence_md)
        if len(t) < MIN_TEMPLATE_CHARS:
            return False
        pages = self._pages.get(t, set()) | ({article_id} if article_id is not None else set())
        return len(pages) >= self.min_pages

    def template_ids(self, blocks: list[Block], article_id: int | None = None) -> set[str]:
        """Sentence ids of an article that are site template text."""
        return {s.id for b in blocks for s in b.sentences if self.is_template(s.md, article_id)}

    def contains_template(self, quote_plain: str, article_id: int | None = None) -> list[str]:
        """Template sentences found inside a quote (empty list = the quote belongs to the story)."""
        qn = norm_for_match(quote_plain)
        return [t for t, pages in self._pages.items()
                if len(pages | ({article_id} if article_id is not None else set())) >= self.min_pages and t in qn]
