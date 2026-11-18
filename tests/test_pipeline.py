"""Offline tests: extraction, span rebuilding, verification, relevance rules and the report,
using a scripted LLM so no network or API key is needed."""
from __future__ import annotations

import importlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "bbc_article.html"
BBC_XPATH = "//article//div[@data-component='text-block' or @data-testid='rich-text']"
URL = "https://www.bbc.co.uk/news/articles/c3zxjx28n15yo"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MQM_DATA", str(tmp_path))
    import mqm.config as config
    importlib.reload(config)
    for mod in ("mqm.logs", "mqm.db", "mqm.fetch", "mqm.llm", "mqm.verify", "mqm.quotes",
                "mqm.pipeline", "mqm.report", "mqm.export"):
        importlib.reload(importlib.import_module(mod))
    from mqm import pipeline
    pipeline.sync_sources()
    return tmp_path


def article():
    from mqm.extract import extract
    return extract(FIXTURE.read_text(encoding="utf-8"), URL, BBC_XPATH)


# ------------------------------------------------------------------ extraction
def test_extracts_metadata_and_clean_blocks():
    a = article()
    assert a.title == "Charity gig planned to support Ukrainian families"
    assert a.published_at == "2026-11-10T10:56:24+00:00"
    assert a.url == URL
    assert len(a.blocks) >= 10
    text = " ".join(b.md for b in a.blocks)
    for noise in ("Related topics", "Image source", "Getty Images", "Read more", "Copyright", "Skip to"):
        assert noise not in text


def test_sentence_splitter_keeps_abbreviations_and_quotes():
    from mqm.extract import split_sentences
    s = split_sentences('Mr. Putin met Gen. Smith in the U.S. on Monday. "We will win," he said. Talks resume.')
    assert s == ['Mr. Putin met Gen. Smith in the U.S. on Monday.', '"We will win," he said.', "Talks resume."]


# ------------------------------------------------------------------ span rebuild = no frankenquotes
def test_rebuild_span_is_verbatim_and_rejects_bad_ranges():
    from mqm.quotes import SpanError, rebuild_span
    a = article()
    span = rebuild_span(a.blocks, "P9.S1", "P10.S1")
    assert span.text_md.startswith("The promoter, Mr. Grant, said he was")
    assert "\n\n" in span.text_md                    # original paragraph break preserved
    with pytest.raises(SpanError):
        rebuild_span(a.blocks, "P10.S1", "P9.S1")    # reversed
    with pytest.raises(SpanError):
        rebuild_span(a.blocks, "P99.S1", "P99.S1")   # invented id
    with pytest.raises(SpanError):
        rebuild_span(a.blocks, "P1.S1", "P8.S1")     # too long


def test_verification_catches_altered_text_and_bad_metadata():
    from mqm.config import get_source
    from mqm.quotes import rebuild_span
    from mqm.verify import verify_quote
    a = article()
    span = rebuild_span(a.blocks, "P10.S1", "P10.S1")
    q = {"quote_id": "Q-x", "source_id": "bbc", "source_name": "BBC News", "publication_date": a.published_at,
         "article_name": a.title, "article_url": a.url, "extraction_date": "2026-11-18T00:00:00+00:00",
         "section_id": "peace_initiatives", "quote_plain": span.text_plain}
    src = {"id": "bbc", "domain": "bbc.co.uk"}
    assert verify_quote(q, a.blocks, src)["passed"]
    forged = dict(q, quote_plain=span.text_plain.replace("stop this war", "end this war now"))
    assert not verify_quote(forged, a.blocks, src)["checks"]["verbatim_in_article"]
    assert not verify_quote(dict(q, article_name=""), a.blocks, src)["checks"]["metadata_complete"]
    assert not verify_quote(dict(q, article_url="https://evil.example/x"), a.blocks, src)["checks"]["url_matches_source"]
    assert not verify_quote(q, a.blocks, src, period_start="2026-11-11")["checks"]["in_period"]


# ------------------------------------------------------------------ deterministic relevance
@pytest.mark.parametrize("title,expected", [
    ("Landmark Crimea bridge hit as Russia reports biggest Ukrainian drone attack", True),
    ("Estonia to close Russia border crossings after attempted drone factory arson attack", True),
    ("Football Union of Russia v Venezuela", False),
    ("Bank of England holds interest rates", False),
])
def test_relevance_rules(title, expected):
    from mqm.relevance import classify
    assert (classify(title) is not None) is expected


# ------------------------------------------------------------------ end to end with a scripted LLM
class ScriptedLLM:
    def __init__(self):
        self.calls = []

    def json(self, purpose, system, user, schema, effort, max_tokens=16000):
        self.calls.append(purpose)
        if purpose == "quote_selection":
            return {"article_relevant": True, "reason": "aid", "quotes": [
                {"start": "P10.S1", "end": "P10.S1", "section": "international_support", "quote_type": "direct_quote",
                 "speaker": "Mr. Grant", "key_facts": ["calls to stop the war"], "topic_key": "aid_gig",
                 "why_material": "x"},
                {"start": "P1.S1", "end": "P1.S1", "section": "international_support", "quote_type": "reported_fact",
                 "speaker": "", "key_facts": ["band organises aid gig"], "topic_key": "aid_gig", "why_material": "x"},
                {"start": "P77.S1", "end": "P77.S1", "section": "sanctions", "quote_type": "analysis",
                 "speaker": "", "key_facts": [], "topic_key": "made_up", "why_material": "hallucinated"},
            ]}, None
        if purpose == "obsolescence":
            ids = [q["id"] for q in json.loads(user)]
            assert len(ids) == 2                    # both quotes are in the same section -> one request
            return {"assessments": [
                {"quote_id": ids[0], "status": "superseded", "superseded_by": ids[1], "materiality": 4, "reason": "r"},
                {"quote_id": ids[1], "status": "current", "superseded_by": "", "materiality": 5, "reason": "r"},
            ]}, None
        if purpose == "section_summaries":
            return {"summaries": [{"section_id": k, "summary": "Reports indicate continued support."}
                                  for k in json.loads(user)]}, None
        raise AssertionError(purpose)


def test_end_to_end_offline(env, monkeypatch):
    from mqm import pipeline, report
    from mqm.config import get_source
    from mqm.db import rows, session
    from mqm.logs import new_run_id
    with session() as c:
        c.execute("INSERT INTO articles(source_id,url,title,discovered_at,discovery_method,relevance_rule,status) "
                  "VALUES ('bbc',?,?,?,'topic_pages','topic_tag','detected')",
                  (URL, "x", datetime.now(timezone.utc).isoformat()))
        art = rows(c, "SELECT * FROM articles")[0]
    llm = ScriptedLLM()
    status = pipeline.process_article(art, get_source("bbc"), llm, "2026-10-01", new_run_id(),
                                      page_html=FIXTURE.read_text(encoding="utf-8"))
    assert status == "extracted"
    with session() as c:
        qs = rows(c, "SELECT * FROM quotes")
    assert len(qs) == 2                                   # the hallucinated span was rejected
    assert all(q["verified"] == 1 for q in qs)

    out = report.build_report(date(2026, 11, 18), llm)
    md = Path(out["path"]).read_text(encoding="utf-8")
    assert md.startswith("# Russia's War on Ukraine — Material Media Quotes, 5 November 2026 – 18 November 2026")
    assert out["shown"] == 1 and out["withheld"] == 1     # superseded quote withheld
    audit = json.loads(Path(out["path"].replace(".md", ".audit.json")).read_text(encoding="utf-8"))
    kept_id = next(i for v in audit["shown"].values() for i in v)
    dropped_id = next(q["quote_id"] for q in qs if q["quote_id"] != kept_id)
    kept = next(q for q in qs if q["quote_id"] == kept_id)
    assert kept["quote_text"] in md                       # verbatim copy
    assert dropped_id not in md                           # obsolete quote not shown
    assert "**Inference:** Reports indicate continued support." in md
    assert f"[{kept['article_name']}]({kept['article_url']})" in md

    from mqm.export import audit_integrity, export_all
    assert audit_integrity()["passed"] == 2
    assert export_all()["quotes"] == 2


# ------------------------------------------------------------------ open-model output hardening
def test_json_cleanup_and_schema_validation():
    from mqm.llm import _strip_to_json, schema_errors
    from mqm.quotes import _schema
    raw = '<think>hmm</think>```json\n{"article_relevant": true, "reason": "r", "quotes": []}\n```'
    data = json.loads(_strip_to_json(raw))
    schema = _schema(["war_intensity"])
    assert schema_errors(data, schema) == []
    bad = {"article_relevant": "yes", "quotes": [{"start": "P1.S1", "section": "made_up"}]}
    errs = schema_errors(bad, schema)
    assert any("expected boolean" in e for e in errs)
    assert any("reason: missing" in e for e in errs)
    assert any("not in" in e for e in errs)
