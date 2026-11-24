"""Exports for LLM contextualisation and audit.

- quotes.jsonl         one verified quote per line, all metadata (load into any vector store / RAG)
- context_pack.md      compact, token-efficient Markdown grouped by section, with ids for citation
- audit integrity      re-verifies every stored quote against its gzipped raw HTML snapshot
"""
from __future__ import annotations

import json
from urllib.parse import urlparse

from .config import EXPORT_DIR, domain_config, get_source
from .db import rows, session
from .extract import extract
from .fetch import load_snapshot
from .grounding import SiteTemplate, ground_fields
from .logs import log, now_iso
from .verify import article_plain, verify_quote

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


def audit_integrity(apply: bool = False) -> dict:
    """Re-check every stored quote from scratch against its raw HTML snapshot.

    Each article is re-extracted with the current extractor, the site's template text is
    recomputed from all of the source's snapshots, and every verification check and field rule is
    run again. Quotes that no longer pass, and LLM-written fields that are not supported by the
    quote, are reported; with apply=True the database is updated (a quote that fails is marked
    unverified, never deleted, and the reasons are stored on it)."""
    with session() as c:
        qs = rows(c, "SELECT q.*, a.snapshot_sha256 FROM quotes q JOIN articles a ON a.id=q.article_id "
                     "ORDER BY q.source_id")
        snaps = rows(c, "SELECT id, source_id, url, snapshot_sha256 FROM articles WHERE snapshot_sha256 IS NOT NULL")
    by_source: dict[str, dict[int, tuple[str, str]]] = {}
    for a in snaps:
        by_source.setdefault(a["source_id"], {})[a["id"]] = (a["snapshot_sha256"], a["url"])

    blocks_cache: dict[str, list] = {}
    templates: dict[str, SiteTemplate] = {}

    def blocks_for(sha: str, url: str, src: dict) -> list:
        if sha not in blocks_cache:
            try:
                blocks_cache[sha] = extract(load_snapshot(sha), url, src.get("body_xpath")).blocks
            except (FileNotFoundError, OSError):
                blocks_cache[sha] = []
        return blocks_cache[sha]

    report: dict = {"checked": 0, "passed": 0, "failed": [], "newly_failed": [], "fields_adjusted": []}
    updates = []
    for q in qs:
        try:
            src = get_source(q["source_id"])
        except KeyError:
            src = {"id": q["source_id"], "name": q["source_name"], "domain": urlparse(q["article_url"]).netloc}
        if q["source_id"] not in templates:
            pages = {aid: blocks_for(sha, url, src) for aid, (sha, url) in by_source.get(q["source_id"], {}).items()}
            templates[q["source_id"]] = SiteTemplate(pages)
        blocks = blocks_for(q["snapshot_sha256"], q["article_url"], src) if q["snapshot_sha256"] else []
        fields = {"quote_plain": q["quote_plain"], "key_facts_json": q["key_facts_json"], "speaker": q["speaker"]}
        adjustments = ground_fields(fields, article_plain(blocks), src)
        v = verify_quote(q, blocks, src, None, blocks, templates[q["source_id"]])
        previous = json.loads(q["verification_json"] or "{}")
        old_adj = previous.get("adjustments", [])
        if adjustments or old_adj:
            v["adjustments"] = old_adj + [a for a in adjustments if a not in old_adj]
        v["reverified_at"] = now_iso()
        report["checked"] += 1
        failed = [k for k, x in v["checks"].items() if not x]
        if v["passed"]:
            report["passed"] += 1
        else:
            report["failed"].append({"quote_id": q["quote_id"], "failed": failed})
            if q["verified"]:
                report["newly_failed"].append({"quote_id": q["quote_id"], "failed": failed})
        if adjustments:
            report["fields_adjusted"].append({"quote_id": q["quote_id"], "changes": adjustments})
        updates.append((int(v["passed"]), json.dumps(v, ensure_ascii=False), fields["key_facts_json"],
                        fields["speaker"], q["quote_id"]))
    if apply:
        with session() as c:
            c.executemany("UPDATE quotes SET verified=?, verification_json=?, key_facts_json=?, speaker=? "
                          "WHERE quote_id=?", updates)
        log("audit.applied", checked=report["checked"], passed=report["passed"],
            newly_failed=len(report["newly_failed"]), fields_adjusted=len(report["fields_adjusted"]))
    report["applied"] = apply
    return report
