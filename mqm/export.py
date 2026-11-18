"""Exports for LLM contextualisation and audit.

- quotes.jsonl         one verified quote per line, all metadata (load into any vector store / RAG)
- context_pack.md      compact, token-efficient Markdown grouped by section, with ids for citation
- audit integrity      re-verifies every stored quote against its gzipped raw HTML snapshot
"""
from __future__ import annotations

import json

from .config import EXPORT_DIR, domain_config, get_source
from .db import rows, session
from .extract import extract
from .fetch import load_snapshot
from .verify import verify_quote

FIELDS = ["quote_id", "quote_text", "source_id", "source_name", "publication_date", "article_name",
          "article_url", "extraction_date", "speaker", "quote_type", "section_id", "topic_key",
          "key_facts", "span_start", "span_end", "text_sha256"]


def export_all(since: str | None = None) -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with session() as c:
        qs = rows(c, "SELECT * FROM quotes WHERE verified=1 AND publication_date>=? "
                     "ORDER BY publication_date DESC, quote_id", since or "")
    with (EXPORT_DIR / "quotes.jsonl").open("w", encoding="utf-8") as f:
        for q in qs:
            q["key_facts"] = json.loads(q["key_facts_json"])
            f.write(json.dumps({k: q.get(k) for k in FIELDS}, ensure_ascii=False) + "\n")
    titles = {s["id"]: s["title"] for s in domain_config()["sections"]}
    lines = ["# Context pack: media quotes on Russia's war on Ukraine",
             "Cite quotes by their [id]. Text is verbatim from the linked article.", ""]
    for sid, title in titles.items():
        sec = [q for q in qs if q["section_id"] == sid]
        if not sec:
            continue
        lines += [f"## {title}"]
        lines += [f"- [{q['quote_id']}] {q['publication_date'][:10]} {q['source_name']}: "
                  f"\"{q['quote_plain']}\" <{q['article_url']}>" for q in sec]
        lines.append("")
    (EXPORT_DIR / "context_pack.md").write_text("\n".join(lines), encoding="utf-8")
    return {"quotes": len(qs), "files": [str(EXPORT_DIR / "quotes.jsonl"), str(EXPORT_DIR / "context_pack.md")]}


def audit_integrity() -> dict:
    """Re-extract each article from its stored snapshot and re-run every verification check."""
    with session() as c:
        qs = rows(c, "SELECT q.*, a.snapshot_sha256 FROM quotes q JOIN articles a ON a.id=q.article_id "
                     "WHERE q.verified=1")
    ok, bad = 0, []
    cache: dict[str, list] = {}
    for q in qs:
        src = get_source(q["source_id"])
        if q["snapshot_sha256"] not in cache:
            cache[q["snapshot_sha256"]] = extract(load_snapshot(q["snapshot_sha256"]), q["article_url"],
                                                  src.get("body_xpath")).blocks
        v = verify_quote(q, cache[q["snapshot_sha256"]], src)
        if v["passed"]:
            ok += 1
        else:
            bad.append({"quote_id": q["quote_id"], "failed": [k for k, x in v["checks"].items() if not x]})
    return {"checked": len(qs), "passed": ok, "failed": bad}
