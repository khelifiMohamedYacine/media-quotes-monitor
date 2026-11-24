"""Tests for "Add a media / think-tank source" (pipeline.register_source / add_source and the
dashboard form at POST /run/add-source).

Everything runs offline: HTTP goes through an httpx.MockTransport that serves a fake website, and
the config directory is a temporary copy, so the real config/sources.yaml is never modified."""
from __future__ import annotations

import importlib
import shutil
import threading
from pathlib import Path

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

ROBOTS_OK = """User-agent: *
Disallow: /wp-admin/
Sitemap: https://www.example.org/news-sitemap.xml
Sitemap: https://www.example.org/sitemap_index.xml
"""
ROBOTS_BLOCK_ALL = "User-agent: *\nDisallow: /\n"
HOME_WITH_FEEDS = """<html><head>
<link rel="alternate" type="application/rss+xml" title="All" href="/feed/">
<link rel="alternate" type="application/atom+xml" href="https://www.example.org/atom.xml">
<link rel="alternate" type="application/rss+xml" href="/feed/">
<link rel="alternate" type="application/rss+xml">
<link rel="alternate" hreflang="fr" href="/fr/">
</head><body>hello</body></html>"""
HOME_NO_FEEDS = """<html><head><title>x</title></head><body>
<a href="/news/ukraine-ceasefire-talks-resume">Ukraine ceasefire talks resume in Geneva</a>
<a href="/news/bank-holds-rates">Bank of England holds interest rates</a>
<a href="https://elsewhere.com/x">Other site</a></body></html>"""
TOPIC_PAGE = """<html><body><a href="/blogs/kyiv-under-fire/">Kyiv under fire again</a>
<a href="/blogs/new-analysis/">A new analysis</a></body></html>"""
RSS_EMPTY = '<?xml version="1.0"?><rss version="2.0"><channel><title>x</title></channel></rss>'


class FakeWeb:
    """Maps URL -> (status, body). Unknown URLs are 404; hosts in `offline` raise a network error."""

    def __init__(self):
        self.pages: dict[str, tuple[int, str]] = {}
        self.offline: set[str] = set()
        self.requests: list[str] = []

    def site(self, host: str, robots: str | None, home: str | None, status: int = 200):
        if robots is not None:
            self.pages[f"https://{host}/robots.txt"] = (status, robots)
        if home is not None:
            self.pages[f"https://{host}"] = (status, home)
            self.pages[f"https://{host}/"] = (status, home)

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if request.url.host in self.offline:
            raise httpx.ConnectError("offline", request=request)
        status, body = self.pages.get(url, self.pages.get(url.rstrip("/"), (404, "not found")))
        return httpx.Response(status, text=body, headers={"content-type": "text/html"})


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shutil.copytree(ROOT / "config", home / "config")
    (home / "inbox").mkdir()
    monkeypatch.setenv("MQM_HOME", str(home))
    monkeypatch.setenv("MQM_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("MQM_REQUEST_DELAY", "0")
    monkeypatch.setenv("MQM_HTTP_RETRIES", "1")
    for key in ("GROQ_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY"):   # behave like CI: no real LLM keys
        monkeypatch.delenv(key, raising=False)
    for mod in ("mqm.config", "mqm.logs", "mqm.db", "mqm.fetch", "mqm.discovery", "mqm.llm", "mqm.verify",
                "mqm.quotes", "mqm.pipeline", "mqm.report", "mqm.export"):
        importlib.reload(importlib.import_module(mod))
    from mqm import fetch, pipeline
    web = FakeWeb()
    mock_client = httpx.Client(transport=httpx.MockTransport(web.handler), follow_redirects=True)
    monkeypatch.setattr(fetch, "client", lambda: mock_client)
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)      # no retry back-off in tests
    pipeline.sync_sources()
    yield {"web": web, "sources_yaml": home / "config" / "sources.yaml"}
    mock_client.close()


def _yaml_sources(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]


# ------------------------------------------------------------------ feed autodetection
def test_autodetect_finds_news_sitemaps_and_rss_links(env):
    from mqm.discovery import autodetect_feeds
    env["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    found = autodetect_feeds("https://www.example.org")
    assert found == [
        {"method": "news_sitemap", "url": "https://www.example.org/news-sitemap.xml"},  # not sitemap_index
        {"method": "rss", "url": "https://www.example.org/feed/"},                      # relative href joined
        {"method": "rss", "url": "https://www.example.org/atom.xml"},                   # atom too
    ]                                                                                   # dupes + no-href dropped


def test_autodetect_survives_blocked_or_offline_site(env):
    from mqm.discovery import autodetect_feeds
    env["web"].site("blocked.org", "denied", "denied", status=403)
    assert autodetect_feeds("https://blocked.org") == []
    env["web"].offline.add("down.org")
    assert autodetect_feeds("https://down.org") == []


# ------------------------------------------------------------------ registering a source
def test_register_source_with_detected_feeds(env):
    from mqm import pipeline
    from mqm.config import get_source
    from mqm.db import one, session
    env["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    header_before = env["sources_yaml"].read_text(encoding="utf-8").split("sources:", 1)[0]
    raw_before = env["sources_yaml"].read_bytes()

    entry = pipeline.register_source("  Atlantic Council ", "https://www.example.org/", None)
    raw_after = env["sources_yaml"].read_bytes()
    assert raw_after.startswith(raw_before)                    # appended: comments/formatting untouched
    assert b'# "War in Ukraine" topic' in raw_after
    assert b"\r\n" not in raw_after[len(raw_before):]          # no Windows line endings introduced

    assert entry["id"] == "atlantic_council"
    assert entry["name"] == "Atlantic Council"                 # whitespace stripped
    assert entry["domain"] == "example.org"                    # www. removed
    assert entry["access"] == "open"
    assert entry["mandatory"] is False
    assert [d["method"] for d in entry["discovery"]] == ["news_sitemap", "rss", "rss"]
    # persisted to YAML, comments header kept, existing sources untouched
    text = env["sources_yaml"].read_text(encoding="utf-8")
    assert text.startswith(header_before)
    ids = [s["id"] for s in _yaml_sources(env["sources_yaml"])]
    assert ids[:3] == ["bbc", "reuters", "guardian"] and ids[-1] == "atlantic_council"
    assert get_source("atlantic_council")["discovery"] == entry["discovery"]
    # and synced into the database so the dashboard / runs can reference it
    with session() as c:
        row = one(c, "SELECT * FROM sources WHERE id='atlantic_council'")
    assert row and row["domain"] == "example.org" and row["access"] == "open"


def test_register_source_url_filters_match_only_that_site(env):
    import re
    from mqm import pipeline
    env["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    entry = pipeline.register_source("Example", "www.example.org", None)  # scheme optional
    inc = re.compile(entry["url_include"][0])
    assert inc.match("https://www.example.org/blogs/ukraine-ceasefire/")
    assert inc.match("https://example.org/a")
    assert inc.match("https://news.example.org/a")
    assert not inc.match("https://www.example.org.evil.com/a")
    assert not inc.match("https://notexample.org/a")


def test_register_source_with_topic_page_when_site_blocks_feeds(env):
    """The real-world Atlantic Council case: robots.txt and homepage answer 403 to the crawler."""
    from mqm import pipeline
    env["web"].site("www.atlanticcouncil.org", "denied", "denied", status=403)
    env["web"].pages["https://www.atlanticcouncil.org/region/ukraine/"] = (200, TOPIC_PAGE)
    entry = pipeline.register_source("Atlantic Council", "https://www.atlanticcouncil.org",
                                     "https://www.atlanticcouncil.org/region/ukraine/")
    assert entry["discovery"] == [{"method": "topic_pages", "url": "https://www.atlanticcouncil.org/region/ukraine/",
                                   "max_pages": 40}]
    assert entry["access"] == "open"     # no readable robots.txt -> allowed, same as fetch.robots_allows


def test_fully_blocked_site_is_added_as_inbox_only(env):
    """The real atlanticcouncil.org answers 403 to everything: it is registered like Reuters
    (detect_only, nothing crawled) so pages saved into ./inbox can still be processed."""
    from mqm import pipeline
    env["web"].site("www.atlanticcouncil.org", "denied", "denied", status=403)
    entry = pipeline.register_source("Atlantic Council", "https://www.atlanticcouncil.org", "")
    assert entry["access"] == "detect_only"
    assert entry["discovery"] == []
    assert "HTTP 403" in entry["access_note"] and "./inbox" in entry["access_note"]


def test_inbox_page_from_inbox_only_source_is_imported(env, monkeypatch):
    from mqm import pipeline
    from mqm.config import INBOX_DIR
    env["web"].site("www.atlanticcouncil.org", "denied", "denied", status=403)
    pipeline.register_source("Atlantic Council", "https://www.atlanticcouncil.org")
    (INBOX_DIR / "saved.html").write_text(
        '<html><head><link rel="canonical" href="https://www.atlanticcouncil.org/blogs/ukraine-x/">'
        "</head><body>x</body></html>", encoding="utf-8")
    seen = []
    monkeypatch.setattr(pipeline, "process_article",
                        lambda art, src, *a, **k: seen.append((art["url"], src["id"])) or "extracted")
    out = pipeline.import_inbox(llm=object())
    assert seen == [("https://www.atlanticcouncil.org/blogs/ukraine-x", "atlantic_council")]
    assert out["imported"][0]["status"] == "extracted"


@pytest.mark.parametrize("setup,homepage,topic,err", [
    ("offline", "https://www.typo-site.org", "", "could not be reached"),
    ("site", "https://www.example.org", "https://www.example.org/region/ukrane/", "topic page .* HTTP 404"),
    ("site", "https://www.example.org", "https://other-site.org/ukraine", "same site as the homepage"),
])
def test_unreadable_or_wrong_addresses_are_rejected(env, setup, homepage, topic, err):
    from mqm import pipeline
    if setup == "offline":
        env["web"].offline.add("www.typo-site.org")
    else:
        env["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    before = env["sources_yaml"].read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=err):
        pipeline.register_source("Some Site", homepage, topic)
    assert env["sources_yaml"].read_text(encoding="utf-8") == before


def test_news_in_domain_name_is_not_mistaken_for_a_news_sitemap(env):
    """euronews.com bug: every sitemap URL contains 'news' via the domain; the old check took them all
    (hundreds of tag sitemaps, so the add-source job appeared to hang)."""
    from mqm.discovery import autodetect_feeds
    env["web"].site("www.euronews.com", "User-agent: *\nAllow: /\n"
                    "Sitemap: https://www.euronews.com/sitemaps/en/latest-news.xml\n"
                    "Sitemap: https://www.euronews.com/sitemap/lettered-pages/tags-a.xml\n"
                    "Sitemap: https://www.euronews.com/sitemaps/en/articles.xml\n", HOME_NO_FEEDS)
    assert autodetect_feeds("https://www.euronews.com") == [
        {"method": "news_sitemap", "url": "https://www.euronews.com/sitemaps/en/latest-news.xml"}]


def test_huge_sitemap_index_is_capped(env):
    from mqm import discovery
    index = ('<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
             + "".join(f"<sitemap><loc>https://www.example.org/sm-{i}.xml</loc></sitemap>" for i in range(300))
             + "</sitemapindex>")
    env["web"].pages["https://www.example.org/news-sitemap.xml"] = (200, index)
    list(discovery.news_sitemap("https://www.example.org/news-sitemap.xml"))
    children = [u for u in env["web"].requests if "/sm-" in u]
    assert len(children) == discovery.MAX_CHILD_SITEMAPS


def test_conventional_feed_location_is_found_when_not_advertised(env):
    from mqm.discovery import autodetect_feeds
    env["web"].site("quiet.org", "User-agent: *\nAllow: /\n", HOME_NO_FEEDS)
    env["web"].pages["https://quiet.org/feed/"] = (200, RSS_EMPTY)
    env["web"].pages["https://quiet.org/rss"] = (200, "<html>not a feed</html>")   # HTML is not a feed
    assert autodetect_feeds("https://quiet.org") == [{"method": "rss", "url": "https://quiet.org/feed/"}]


def test_homepage_is_used_as_listing_when_site_has_no_feed(env):
    """No sitemap/RSS anywhere: the homepage becomes an *untagged* listing, so only articles whose
    headline matches the Russia/Ukraine rules are in scope."""
    from datetime import date
    from mqm import discovery, pipeline
    from mqm.relevance import classify
    env["web"].site("plain.org", "User-agent: *\nAllow: /\n", HOME_NO_FEEDS)
    entry = pipeline.register_source("Plain Institute", "https://plain.org")
    assert entry["access"] == "open"
    assert entry["discovery"] == [{"method": "topic_pages", "url": "https://plain.org/", "max_pages": 1,
                                   "tagged": False}]
    cands = discovery.discover(entry, date(2026, 10, 1), date.today())
    assert sorted(c.url for c in cands) == ["https://plain.org/news/bank-holds-rates",
                                           "https://plain.org/news/ukraine-ceasefire-talks-resume"]
    relevant = [c.url for c in cands if classify(c.title, c.standfirst, c.url, c.tagged)]
    assert relevant == ["https://plain.org/news/ukraine-ceasefire-talks-resume"]


def test_pasted_section_link_is_used_as_untagged_listing(env):
    """User pastes a section/article link instead of the homepage: the site is registered and that page
    is also read for article links (untagged, so the relevance rules still apply)."""
    from mqm import pipeline
    env["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    env["web"].pages["https://www.example.org/region/ukraine/"] = (200, TOPIC_PAGE)
    entry = pipeline.register_source("Example", "https://www.example.org/region/ukraine/")
    assert entry["domain"] == "example.org"
    assert {"method": "topic_pages", "url": "https://www.example.org/region/ukraine/", "max_pages": 3,
            "tagged": False} in entry["discovery"]
    assert any(d["method"] == "news_sitemap" for d in entry["discovery"])


def test_register_source_robots_disallow_makes_it_detect_only(env):
    from mqm import pipeline
    env["web"].site("closed.org", ROBOTS_BLOCK_ALL, HOME_WITH_FEEDS)
    entry = pipeline.register_source("Closed Institute", "https://closed.org", "https://closed.org/ukraine")
    assert entry["access"] == "detect_only"


@pytest.mark.parametrize("name,homepage,topic,err", [
    ("", "https://www.example.org", "", "name is required"),
    ("   ", "https://www.example.org", "", "name is required"),
    ("Новая газета", "https://www.example.org", "", "at least one Latin letter"),
    ("X", "", "", "homepage is required"),
    ("X", "not a url", "", "not a valid web address"),
    ("X", "localhost", "", "not a valid web address"),
    ("X", "ftp://example.org", "", "not a valid web address"),
    ("X", "https://www.example.org", "javascript:alert(1)", "topic page URL is not a valid web address"),
    ("BBC News", "https://www.example.org", "", "already exists"),          # same name as an existing source
    ("bbc", "https://www.example.org", "", "already exists"),               # same id
    ("Beeb", "https://www.bbc.co.uk", "", "already registered as 'BBC News'"),  # same domain
])
def test_register_source_rejects_bad_input(env, name, homepage, topic, err):
    from mqm import pipeline
    env["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    before = env["sources_yaml"].read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=err):
        pipeline.register_source(name, homepage, topic)
    assert env["sources_yaml"].read_text(encoding="utf-8") == before
    assert env["web"].requests == []        # validated before any network access


def test_add_source_twice_is_rejected(env):
    from mqm import pipeline
    env["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    pipeline.add_source("Example", "https://www.example.org", run_backfill=False)
    with pytest.raises(ValueError, match="already exists"):
        pipeline.add_source("Example", "https://www.example.org", run_backfill=False)
    assert [s["id"] for s in _yaml_sources(env["sources_yaml"])].count("example") == 1


def test_add_source_backfills_only_the_new_source(env, monkeypatch):
    from datetime import date
    from mqm import pipeline
    env["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    calls = []
    monkeypatch.setattr(pipeline, "collect", lambda *a, **k: calls.append(a) or {"status": "ok"})
    out = pipeline.add_source("Example", "https://www.example.org", backfill_from="2026-10-01")
    assert out["backfill"] == {"status": "ok"}
    assert calls == [(date(2026, 10, 1), date.today(), ["example"], "add_source", None)]


class _NoCallLLM:
    """Stand-in model: the empty feed means no article reaches the LLM step."""

    def json(self, *a, **k):
        raise AssertionError("no LLM call expected")


def test_new_source_runs_through_collect(env, monkeypatch):
    """After registering, a real collect() run picks the source up (no LLM needed: feed is empty)."""
    from datetime import date
    from mqm import pipeline
    from mqm.db import one, session
    env["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    env["web"].pages["https://www.example.org/news-sitemap.xml"] = (
        200, '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>')
    env["web"].pages["https://www.example.org/feed/"] = (200, '<?xml version="1.0"?><rss><channel/></rss>')
    env["web"].pages["https://www.example.org/atom.xml"] = (
        200, '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>')
    pipeline.register_source("Example", "https://www.example.org")
    out = pipeline.collect(date(2026, 11, 1), date(2026, 11, 2), ["example"], "add_source",
                           llm=_NoCallLLM())
    assert out["status"] in ("ok", "partial"), out
    assert [d["source"] for d in out["detect"]] == ["example"]
    assert out["detect"][0]["candidates"] == 0
    assert any("example.org/news-sitemap.xml" in u for u in env["web"].requests)  # detected feed was used
    with session() as c:
        assert one(c, "SELECT kind FROM runs WHERE run_id=?", out["run_id"])["kind"] == "add_source"


# ------------------------------------------------------------------ dashboard form
@pytest.fixture()
def dash(env, monkeypatch):
    from fastapi.testclient import TestClient
    import mqm.web.app as webapp
    importlib.reload(webapp)
    from mqm import pipeline
    started = threading.Event()
    release = threading.Event()
    backfills = []

    def fake_backfill(source_id, *a, **k):
        backfills.append(source_id)
        started.set()
        release.wait(5)
        return {"status": "ok"}

    monkeypatch.setattr(pipeline, "backfill_source", fake_backfill)
    with TestClient(webapp.app) as client:
        yield {"client": client, "app": webapp, "backfills": backfills, "started": started,
               "release": release, **env}
    release.set()


def _wait_idle(app_module, timeout=5.0):
    assert app_module._job_lock.acquire(timeout=timeout)
    app_module._job_lock.release()


def test_home_page_has_the_add_source_form(dash):
    page = dash["client"].get("/").text
    assert '<form method="post" action="/run/add-source" id="add-source-form">' in page
    for field in ('name="name"', 'name="homepage"', 'name="topic_url"'):
        assert field in page
    for label in ('<label for="src-name">', '<label for="src-home">', '<label for="src-topic">'):
        assert label in page                                       # every field is labelled
    assert "Build pipeline" in page
    assert 'data-home="https://kyivindependent.com"' in page        # one-click example
    assert "What happens when I add a source?" in page


def test_form_adds_source_and_starts_backfill(dash):
    dash["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    r = dash["client"].post("/run/add-source", data={"name": "Atlantic Council",
                                                    "homepage": "https://www.example.org", "topic_url": ""},
                            follow_redirects=False)
    assert r.status_code == 303
    assert dash["started"].wait(5)
    assert dash["backfills"] == ["atlantic_council"]
    page = dash["client"].get(r.headers["location"]).text
    assert "Atlantic Council added. Articles are found through its RSS feed and news sitemap." in page
    assert "Collecting its articles since 2026-10-01 in the background" in page
    assert "Running now: <b>backfill new source Atlantic Council</b>" in page
    assert "<td><b>Atlantic Council</b><div class=\"small muted\">example.org</div></td>" in page  # Sources table
    assert "/api/status" in page                                    # page polls until the job finishes
    assert 'value="atlantic_council"' in page                      # selectable in "Collect a past period"
    dash["release"].set()
    _wait_idle(dash["app"])
    page = dash["client"].get("/").text
    assert "Last job" not in page and "Running now" not in page


def test_form_shows_error_when_source_cannot_be_built(dash):
    """Before the fix the page said 'Source pipeline being built' and the error was only in the logs."""
    dash["web"].offline.add("www.atlanticcouncil.org")
    before = dash["sources_yaml"].read_text(encoding="utf-8")
    r = dash["client"].post("/run/add-source", data={"name": "Atlantic Council",
                                                    "homepage": "https://www.atlanticcouncil.org"})
    assert r.status_code == 200
    assert ('class="msg err" role="alert"><span>!</span><div>Could not add source: cannot read '
            'https://www.atlanticcouncil.org/: the site could not be reached') in r.text
    assert "being built" not in r.text and "Atlantic Council added" not in r.text
    assert dash["backfills"] == []
    assert dash["sources_yaml"].read_text(encoding="utf-8") == before


def test_form_with_topic_url_works_for_blocked_site(dash):
    dash["web"].site("www.atlanticcouncil.org", "denied", "denied", status=403)
    dash["web"].pages["https://www.atlanticcouncil.org/region/ukraine/"] = (200, TOPIC_PAGE)
    r = dash["client"].post("/run/add-source", data={"name": "Atlantic Council",
                                                    "homepage": "https://www.atlanticcouncil.org",
                                                    "topic_url": "https://www.atlanticcouncil.org/region/ukraine/"})
    assert "Atlantic Council added. Articles are found through its topic page." in r.text
    assert dash["started"].wait(5) and dash["backfills"] == ["atlantic_council"]


def test_form_rejects_invalid_homepage_with_message(dash):
    r = dash["client"].post("/run/add-source", data={"name": "X", "homepage": "not a url"})
    assert "Could not add source: homepage is not a valid web address" in r.text


def test_form_escapes_user_input_in_error_message(dash):
    r = dash["client"].post("/run/add-source", data={"name": "<script>alert(1)</script>",
                                                    "homepage": "<b>bad</b>"})
    assert "<script>alert(1)</script>" not in r.text and "<b>bad</b>" not in r.text
    assert "&lt;b&gt;bad&lt;/b&gt;" in r.text


def test_form_requires_name_and_homepage(dash):
    assert dash["client"].post("/run/add-source", data={"homepage": "https://www.example.org"}).status_code == 422
    assert dash["client"].post("/run/add-source", data={"name": "X"}).status_code == 422


def test_form_refuses_while_another_job_runs(dash):
    dash["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    dash["web"].site("other.org", ROBOTS_OK, HOME_WITH_FEEDS)
    dash["client"].post("/run/add-source", data={"name": "Example", "homepage": "https://www.example.org"})
    assert dash["started"].wait(5)
    r = dash["client"].post("/run/add-source", data={"name": "Other", "homepage": "https://other.org"})
    assert "A job is already running" in r.text
    assert "other" not in [s["id"] for s in _yaml_sources(dash["sources_yaml"])]


def test_failed_background_job_is_shown_on_dashboard(dash, monkeypatch):
    from mqm import pipeline

    def boom(source_id, *a, **k):
        raise RuntimeError("LLM provider key missing")

    monkeypatch.setattr(pipeline, "backfill_source", boom)
    dash["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    dash["client"].post("/run/add-source", data={"name": "Example", "homepage": "https://www.example.org"})
    _wait_idle(dash["app"])
    page = dash["client"].get("/").text
    assert "Last job <b>backfill new source Example</b> failed: LLM provider key missing" in page


def test_status_api_reports_running_and_finished_job(dash):
    dash["web"].site("www.example.org", ROBOTS_OK, HOME_WITH_FEEDS)
    assert dash["client"].get("/api/status").json() == {"job": None, "last": None}
    dash["client"].post("/run/add-source", data={"name": "Example", "homepage": "https://www.example.org"})
    assert dash["started"].wait(5)
    assert dash["client"].get("/api/status").json()["job"] == "backfill new source Example"
    page = dash["client"].get("/").text
    assert 'id="add-source-btn" disabled' in page                   # can't start a second job meanwhile
    dash["release"].set()
    _wait_idle(dash["app"])
    assert dash["client"].get("/api/status").json() == {
        "job": None, "last": {"name": "backfill new source Example", "ok": True, "error": ""}}


@pytest.mark.parametrize("path,heading", [("/", "Overview"), ("/quotes", "Quotes"), ("/articles", "Articles"),
                                          ("/logs", "Logs"), ("/llm-calls", "LLM calls"), ("/reports", "Daily reports")])
def test_every_page_renders_with_navigation(dash, path, heading):
    r = dash["client"].get(path)
    assert r.status_code == 200
    assert f"<h1>{heading}</h1>" in r.text
    assert 'class="nav-links"' in r.text
    if path != "/":
        assert f'<a href="{path}" class="active">' in r.text


def test_form_keeps_what_the_user_typed_after_an_error(dash):
    r = dash["client"].post("/run/add-source", data={"name": "My Think Tank", "homepage": "not a url",
                                                    "topic_url": "https://x.org/ukraine"})
    assert "homepage is not a valid web address: &#39;not a url&#39;" in r.text   # shows what was typed
    assert 'value="My Think Tank"' in r.text and 'value="not a url"' in r.text
    assert 'value="https://x.org/ukraine"' in r.text
    card = r.text.index('id="add-source"')
    assert r.text.index("Could not add source") > card                            # error shown next to the form


def test_form_inbox_only_source_explains_itself_and_skips_backfill(dash):
    dash["web"].site("www.atlanticcouncil.org", "denied", "denied", status=403)
    r = dash["client"].post("/run/add-source", data={"name": "Atlantic Council",
                                                    "homepage": "https://www.atlanticcouncil.org"})
    assert "Atlantic Council added as an inbox-only source" in r.text
    assert "Import ./inbox pages" in r.text
    assert dash["backfills"] == [] and "Running now" not in r.text      # nothing to crawl -> no job
    assert '<span class="badge b-info" title="Not crawled: articles come from ./inbox">Inbox only</span>' in r.text


@pytest.mark.parametrize("rule,label", [
    (r"core:\bukrain", 'keyword "ukrain"'),
    (r"russia+context:\brussia\w*&\bdrone", 'keywords "russia" + "drone"'),
    ("topic_tag", "listed on the site's Ukraine topic page"),
    ("manual_import", "imported from ./inbox"),
    (None, ""),
])
def test_relevance_rules_are_shown_in_plain_language(rule, label):
    from mqm.web.app import _rule_label
    assert _rule_label(rule) == label
