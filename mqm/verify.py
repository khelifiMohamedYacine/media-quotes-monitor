"""Deterministic verification of every quote before it is trusted.

A quote is stored as verified only if ALL checks pass. The full check results are kept on the
quote row (verification_json) so an auditor can see exactly why it was accepted or rejected.
"""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urlparse

from .extract import NOISE, Block, norm_for_match, plain

MIN_CHARS, MAX_CHARS = 40, 1500


def _iso_ok(v: str | None) -> bool:
    if not v:
        return False
    try:
        datetime.fromisoformat(v)
        return True
    except ValueError:
        return False


def article_plain(blocks: list[Block]) -> str:
    return " ".join(plain(b.md) for b in blocks)


def verify_quote(q: dict, blocks: list[Block], source: dict, period_start: str | None = None,
                 snapshot_blocks: list[Block] | None = None) -> dict:
    checks: dict[str, bool] = {}
    text = q.get("quote_plain", "")
    host = urlparse(q.get("article_url", "")).netloc.lower()

    checks["verbatim_in_article"] = bool(text) and norm_for_match(text) in norm_for_match(article_plain(blocks))
    if snapshot_blocks is not None:
        # re-extracted independently from the stored raw HTML: must give the same text
        checks["verbatim_in_snapshot"] = norm_for_match(text) in norm_for_match(article_plain(snapshot_blocks))
    checks["length_ok"] = MIN_CHARS <= len(text) <= MAX_CHARS
    checks["no_boilerplate"] = not NOISE.match(text) and "http" not in text and text.count("|") < 2
    checks["complete_sentence_end"] = bool(re.search(r"[.!?\"”’')\]]$", text))
    checks["metadata_complete"] = all(q.get(k) for k in (
        "quote_id", "source_id", "source_name", "publication_date", "article_name", "article_url",
        "extraction_date", "section_id"))
    checks["dates_valid"] = _iso_ok(q.get("publication_date")) and _iso_ok(q.get("extraction_date")) \
        and q.get("publication_date", "") <= q.get("extraction_date", "")
    checks["url_matches_source"] = q.get("article_url", "").startswith("https://") and \
        (host == source["domain"] or host.endswith("." + source["domain"]))
    if period_start:
        checks["in_period"] = q.get("publication_date", "") >= period_start
    return {"passed": all(checks.values()), "checks": checks}
