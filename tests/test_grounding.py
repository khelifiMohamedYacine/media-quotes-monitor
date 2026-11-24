"""Trust rules beyond verbatim matching: the story's own text only (no feeds or site template),
and LLM-written fields (key facts, speaker) supported by the quote. All offline."""
from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone

import pytest

from mqm.extract import Block, Sentence, extract, numbered_text
from mqm.grounding import SiteTemplate, ground_key_facts, ground_speaker, numbers_in


GUARDIAN = {"id": "guardian", "name": "The Guardian", "domain": "theguardian.com"}
BBC = {"id": "bbc", "name": "BBC News", "domain": "bbc.co.uk"}


# ------------------------------------------------------------------ numbers
@pytest.mark.parametrize("text,expected", [
    ("Yalta 2, 3, 4", {"2", "3", "4"}),            # a list, not the number 234
    ("1,600 new entries", {"1600"}),
    ("support at just 18%", {"18"}),
    ("nine people were hurt", {"9"}),
    ("on Nov. 24", {"24"}),
    ("$174bn next year", {"174"}),
])
def test_numbers_are_read_from_digits_and_words(text, expected):
    assert numbers_in(text) == expected


# ------------------------------------------------------------------ key facts (real cases from the audit)
@pytest.mark.parametrize("quote,fact,kept", [
    # invented statistic
    ("Polls show a majority want an end to military aid. One put support for arms at just 18%.",
     "62% of Italians oppose continued military aid", False),
    # relative time converted to a (wrong) year
    ("The war will cost EUR155bn next year, with at least EUR70bn needed from the EU.",
     "Ukraine needs EUR70bn from the EU in 2026", False),
    # approximation turned into a falsely exact figure
    ("At least eight people were killed and nearly 30 injured.", "8 killed, 29 injured", False),
    # weekday turned into a date
    ("A Russian warship was sunk by Ukrainian drones on Sunday.", "Warship sunk on 2026-11-22", False),
    # faithful paraphrases are kept, including numbers written as words in the quote
    ("Nine people are reported to have been hurt, including one child.", "9 people injured, including a child", True),
    ("At least eight people were killed and nearly 30 injured.", "At least 8 killed and nearly 30 injured", True),
    ("Zelensky called it an exemplary operation.", "Zelensky described the operation as an exemplary one", True),
    ("Russia controls approximately a fifth of Ukrainian territory.", "Russia controls roughly one-fifth of Ukraine", True),
    ("About an hour later, the alert was ended.", "Alert ended after one hour", True),
])
def test_key_facts_must_be_supported_by_the_quote(quote, fact, kept):
    facts, removed = ground_key_facts([fact], quote)
    assert (facts == [fact]) is kept
    if not kept:
        assert removed[0]["field"] == "key_fact" and "not in the quote" in removed[0]["reason"]


# ------------------------------------------------------------------ speaker
@pytest.mark.parametrize("speaker,source,expected,reason", [
    ("The Guardian", GUARDIAN, "", "publication"),
    ("Guardian", GUARDIAN, "", "publication"),
    ("BBC", BBC, "", "publication"),
    ("General context", BBC, "", "placeholder"),
    ("Background", BBC, "", "placeholder"),
    ("General Staff of the Armed Forces of Ukraine", BBC, "General Staff of the Armed Forces of Ukraine", None),
    ("Prime Minister Serhii Koretskyi", BBC, "Prime Minister Serhii Koretskyi", None),
    ("Dmitry Medvedev", BBC, "", "not found"),        # nobody of that name in the article
])
def test_speaker_must_be_a_real_speaker_named_in_the_article(speaker, source, expected, reason):
    article = ("Ukraine's General Staff said losses rose. \"Pressure on the aggressor's finances must grow,\" "
               "said Prime Minister Serhii Koretskyi.")
    got, changes = ground_speaker(speaker, "Losses rose sharply, officials said.", article, source)
    assert got == expected
    assert (reason is None and not changes) or (reason and reason in changes[0]["reason"])


# ------------------------------------------------------------------ extraction: the story, not the feed
PAGE_WITH_FEED = """<html><head><title>Ukraine strikes Russian ship in Libya</title>
<meta property="article:published_time" content="2026-11-24T10:00:00Z"></head><body>
<article class="story"><h1>Ukraine strikes Russian ship in Libya</h1>
<p>Ukraine says its naval drones struck a Russian corvette in the port of Benghazi on Sunday.</p>
<p>The foreign ministry called the operation lawful self-defence under international law.</p>
<p>Russia has not commented on the damage to the vessel or on any casualties among the crew.</p>
<article class="embedded-card"><p>Read also: an unrelated story embedded inside the article body.</p></article>
</article>
<section class="news-feed">
<article><p>In Zaporizhzhia Oblast, an attack killed two people and injured 48 others.</p></article>
<article><p>A startup raised five million dollars to expand drone production lines.</p></article>
<article><p>South Korea issued its first preliminary air raid alert over Russian warplanes.</p></article>
</section></body></html>"""


def test_extraction_keeps_only_the_main_story_when_a_page_has_a_news_feed():
    a = extract(PAGE_WITH_FEED, "https://example.org/strike")
    text = " ".join(b.md for b in a.blocks)
    assert "naval drones struck a Russian corvette" in text
    assert "Zaporizhzhia" not in text and "startup" not in text and "South Korea" not in text
    assert "unrelated story embedded" not in text          # card nested inside the story


# ------------------------------------------------------------------ site template
def _blocks(*sentences):
    return [Block(f"P{i + 1}", "p", s, [Sentence(f"P{i + 1}.S1", s)]) for i, s in enumerate(sentences)]


FEED = "Swedish Gripen pilot unable to shoot down suspected missile over Poland, commander says."
BACKGROUND = "Russia launched its full-scale invasion of Ukraine in February 2022, officials recalled."


def test_text_repeated_on_three_pages_of_a_site_is_template_not_story():
    t = SiteTemplate({1: _blocks(FEED, "Story one about bridges."), 2: _blocks(FEED, BACKGROUND),
                      3: _blocks(FEED, BACKGROUND, "Story three about drones in the Black Sea region.")})
    assert t.is_template(FEED)                      # on 3 pages
    assert not t.is_template(BACKGROUND)            # on 2 pages: legitimate reuse (e.g. a daily briefing)
    assert t.template_ids(_blocks(FEED, "A new story sentence that is long enough."), 4) == {"P1.S1"}
    assert t.contains_template(f"Lead sentence. {FEED}", 9)
    assert not t.contains_template("Story three about drones in the Black Sea region.", 3)


def test_template_sentences_are_hidden_from_the_llm_and_cannot_be_quoted():
    from mqm.quotes import SpanError, rebuild_span   # imported here: other tests reload mqm.quotes
    blocks = _blocks("Real story sentence number one is here.", FEED, "Real story sentence number two is here.")
    view = numbered_text(blocks, hidden={"P2.S1"})
    assert "Gripen" not in view and "[P1.S1]" in view and "[P3.S1]" in view
    with pytest.raises(SpanError, match="site template"):
        rebuild_span(blocks, "P1.S1", "P3.S1", hidden={"P2.S1"})   # would stitch across the feed item
    assert rebuild_span(blocks, "P3.S1", "P3.S1", hidden={"P2.S1"}).text_plain.startswith("Real story")


# ------------------------------------------------------------------ end to end: store + audit
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


class OneQuoteLLM:
    def __init__(self, picks):
        self.picks = picks

    def json(self, purpose, system, user, schema, effort, max_tokens=16000):
        assert "Zaporizhzhia" not in user            # feed text never reaches the model
        return {"article_relevant": True, "reason": "r", "quotes": self.picks}, None


def _article(c, url):
    c.execute("INSERT INTO articles(source_id,url,title,discovered_at,discovery_method,relevance_rule,status) "
              "VALUES ('guardian',?,?,?,'rss','core:ukrain','detected')",
              (url, "Strike on ship", datetime.now(timezone.utc).isoformat()))


def test_stored_quote_records_every_correction_and_audit_reapplies_rules(env):
    from mqm import export, pipeline
    from mqm.config import get_source
    from mqm.db import one, rows, session
    from mqm.logs import new_run_id
    url = "https://www.theguardian.com/world/2026/nov/24/strike-on-ship"
    with session() as c:
        _article(c, url)
        art = rows(c, "SELECT * FROM articles")[0]
    llm = OneQuoteLLM([{"start": "P1.S1", "end": "P2.S1", "section": "escalation", "quote_type": "reported_fact",
                        "speaker": "The Guardian", "topic_key": "libya_strike", "why_material": "x",
                        "key_facts": ["Ukrainian drones struck a Russian corvette in Benghazi",
                                      "Corvette struck on 2026-11-22"]}])
    page = PAGE_WITH_FEED.replace("https://example.org", "https://www.theguardian.com")
    status = pipeline.process_article(art, get_source("guardian"), llm, "2026-10-01", new_run_id(), page_html=page)
    assert status == "extracted"
    with session() as c:
        q = one(c, "SELECT * FROM quotes")
    v = json.loads(q["verification_json"])
    assert q["verified"] == 1 and v["checks"]["belongs_to_story"] is True
    assert q["speaker"] == "" and json.loads(q["key_facts_json"]) == ["Ukrainian drones struck a Russian corvette in Benghazi"]
    assert {a["field"] for a in v["adjustments"]} == {"speaker", "key_fact"}     # every change is on record

    # simulate a quote stored before these rules existed, then re-audit from the raw snapshot
    with session() as c:
        c.execute("UPDATE quotes SET speaker='The Guardian', key_facts_json=? WHERE quote_id=?",
                  (json.dumps(["62% oppose aid"]), q["quote_id"]))
    dry = export.audit_integrity()
    assert dry["applied"] is False and dry["fields_adjusted"]
    with session() as c:
        assert one(c, "SELECT speaker FROM quotes")["speaker"] == "The Guardian"   # dry run changes nothing
    export.audit_integrity(apply=True)
    with session() as c:
        q2 = one(c, "SELECT * FROM quotes")
    assert q2["speaker"] == "" and json.loads(q2["key_facts_json"]) == [] and q2["verified"] == 1
    assert "reverified_at" in json.loads(q2["verification_json"])


def test_audit_marks_quotes_from_other_stories_unverified_without_deleting_them(env):
    from mqm import export
    from mqm.db import one, session
    from mqm.fetch import save_snapshot
    import hashlib
    url = "https://www.theguardian.com/world/2026/nov/24/strike-on-ship"
    raw = PAGE_WITH_FEED.encode()
    sha = hashlib.sha256(raw).hexdigest()
    save_snapshot(sha, raw)
    with session() as c:
        _article(c, url)
        c.execute("UPDATE articles SET snapshot_sha256=? WHERE url=?", (sha, url))
        aid = one(c, "SELECT id FROM articles")["id"]
        # a quote taken from the news feed by an older extractor
        c.execute("INSERT INTO quotes(quote_id,article_id,source_id,source_name,publication_date,article_name,"
                  "article_url,extraction_date,quote_text,quote_plain,span_start,span_end,speaker,quote_type,"
                  "section_id,key_facts_json,topic_key,relevance_rationale,text_sha256,verification_json,verified)"
                  " VALUES ('Q-feed',?,'guardian','The Guardian','2026-11-24T10:00:00+00:00','Strike on ship',?,"
                  "'2026-11-24T11:00:00+00:00',?,?,'P5.S1','P5.S1','','reported_fact','humanitarian','[]','k','r',"
                  "'x','{}',1)", (aid, url, *(["In Zaporizhzhia Oblast, an attack killed two people and injured 48 others."] * 2)))
    r = export.audit_integrity(apply=True)
    assert r["newly_failed"] == [{"quote_id": "Q-feed", "failed": ["verbatim_in_article", "verbatim_in_snapshot"]}]
    with session() as c:
        q = one(c, "SELECT verified, verification_json FROM quotes WHERE quote_id='Q-feed'")
    assert q["verified"] == 0 and json.loads(q["verification_json"])["checks"]["verbatim_in_snapshot"] is False


def test_quote_page_shows_new_check_and_corrections(env):
    from fastapi.testclient import TestClient
    from mqm import pipeline
    from mqm.config import get_source
    from mqm.db import one, rows, session
    from mqm.logs import new_run_id
    import mqm.web.app as webapp
    importlib.reload(webapp)
    url = "https://www.theguardian.com/world/2026/nov/24/strike-on-ship"
    with session() as c:
        _article(c, url)
        art = rows(c, "SELECT * FROM articles")[0]
    llm = OneQuoteLLM([{"start": "P1.S1", "end": "P1.S1", "section": "escalation", "quote_type": "reported_fact",
                        "speaker": "The Guardian", "topic_key": "k", "why_material": "x",
                        "key_facts": ["Strike on 2026-11-22"]}])
    page = PAGE_WITH_FEED.replace("https://example.org", "https://www.theguardian.com")
    pipeline.process_article(art, get_source("guardian"), llm, "2026-10-01", new_run_id(), page_html=page)
    with session() as c:
        qid = one(c, "SELECT quote_id FROM quotes")["quote_id"]
    with TestClient(webapp.app) as client:
        html = client.get(f"/quotes/{qid}").text
        assert "Belongs to story" in html and "not a feed, promotion or summary" in html
        assert "<h2>Corrections</h2>" in html
        assert "<s class=\"muted\">Strike on 2026-11-22</s>" in html and "Number(s) 11, 2026, 22 not in the quote" in html
        assert "<s class=\"muted\">The Guardian</s>" in html and "The publication itself is not a speaker" in html
        r = client.post("/run/audit", follow_redirects=False)
        assert r.status_code == 303 and "Re-checking" in r.headers["location"].replace("+", " ").replace("%20", " ")
