"""Orchestration: daily collection, backfill for a chosen period, inbox import, new sources."""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

from . import discovery, fetch, relevance
from .config import DEFAULT_START, INBOX_DIR, append_source_config, get_source, sources_config
from .db import dumps, init_db, one, rows, session
from .extract import Article, blocks_from_json, extract
from .llm import LLM, LLMError, default_llm
from .logs import log, new_run_id, now_iso
from .quotes import SpanError, quote_id, rebuild_span, select_quotes
from .verify import verify_quote

MAX_ATTEMPTS = 3


# --------------------------------------------------------------------------- runs
def _start_run(kind: str, params: dict) -> str:
    rid = new_run_id()
    with session() as c:
        c.execute("INSERT INTO runs(run_id,kind,params_json,started_at,status) VALUES (?,?,?,?,?)",
                  (rid, kind, dumps(params), now_iso(), "running"))
    log("run.start", kind=kind, **params)
    return rid


def _finish_run(rid: str, status: str, stats: dict) -> None:
    with session() as c:
        c.execute("UPDATE runs SET finished_at=?, status=?, stats_json=? WHERE run_id=?",
                  (now_iso(), status, dumps(stats), rid))
    log("run.finish", status=status, **stats)


def sync_sources() -> None:
    init_db()
    with session() as c:
        for s in sources_config():
            c.execute(
                "INSERT INTO sources(id,name,domain,access,mandatory,config_json,added_at) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, domain=excluded.domain, "
                "access=excluded.access, mandatory=excluded.mandatory, config_json=excluded.config_json",
                (s["id"], s["name"], s["domain"], s.get("access", "open"), int(s.get("mandatory", False)),
                 dumps(s), now_iso()))


# --------------------------------------------------------------------------- detection
def detect(source: dict, day_from: date, day_to: date, run_id: str) -> dict:
    cands = discovery.discover(source, day_from, day_to)
    end = datetime.combine(day_to + timedelta(days=1), datetime.min.time(), timezone.utc).isoformat()
    new = relevant = 0
    restricted = source.get("access") == "detect_only"
    with session() as c:
        for cand in cands:
            if cand.published_at and cand.published_at >= end and day_to < date.today():
                continue
            rule = relevance.classify(cand.title, cand.standfirst, cand.url, cand.tagged)
            if not rule:
                continue
            relevant += 1
            status = "access_restricted" if restricted else "detected"
            cur = c.execute(
                "INSERT OR IGNORE INTO articles(source_id,url,title,standfirst,published_at,discovered_at,"
                "discovery_method,relevance_rule,status,run_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (source["id"], cand.url, cand.title, cand.standfirst, cand.published_at, now_iso(),
                 cand.method, rule, status, run_id))
            new += cur.rowcount
        # a backfill for an earlier period re-opens articles skipped as out of period
        c.execute("UPDATE articles SET status='detected' WHERE source_id=? AND status='out_of_period' "
                  "AND published_at>=?", (source["id"], day_from.isoformat()))
    stats = {"source": source["id"], "candidates": len(cands), "relevant": relevant, "new": new}
    log("detect.done", **stats)
    return stats


# --------------------------------------------------------------------------- processing
def _store_quotes(c, art: dict, source: dict, a: Article, picks: dict, call_id: int | None,
                  period_start: str, snapshot_blocks, run_id: str) -> tuple[int, int]:
    stored = rejected = 0
    for p in picks.get("quotes", []):
        try:
            span = rebuild_span(a.blocks, p["start"], p["end"])
        except SpanError as e:
            rejected += 1
            log("quote.rejected", "warning", url=art["url"], span=f"{p['start']}-{p['end']}", reason=str(e))
            continue
        q = {
            "quote_id": quote_id(a.url, span.text_plain),
            "article_id": art["id"], "source_id": source["id"], "source_name": source["name"],
            "publication_date": a.published_at, "article_name": a.title, "article_url": a.url,
            "extraction_date": now_iso(), "quote_text": span.text_md, "quote_plain": span.text_plain,
            "span_start": span.start, "span_end": span.end, "speaker": p.get("speaker", ""),
            "quote_type": p.get("quote_type"), "section_id": p["section"],
            "key_facts_json": dumps(p.get("key_facts", [])), "topic_key": p.get("topic_key", ""),
            "relevance_rationale": p.get("why_material", ""),
        }
        v = verify_quote(q, a.blocks, source, period_start, snapshot_blocks)
        q.update(text_sha256=hashlib.sha256(span.text_plain.encode()).hexdigest(),
                 verification_json=dumps(v), verified=int(v["passed"]), llm_call_id=call_id, run_id=run_id)
        cols = ",".join(q)
        c.execute(f"INSERT OR IGNORE INTO quotes({cols}) VALUES ({','.join('?' * len(q))})", tuple(q.values()))
        if v["passed"]:
            stored += 1
        else:
            rejected += 1
            log("quote.failed_verification", "warning", quote_id=q["quote_id"],
                failed=[k for k, ok in v["checks"].items() if not ok])
    return stored, rejected


def process_article(art: dict, source: dict, llm: LLM, period_start: str, run_id: str,
                    page_html: str | None = None) -> str:
    """Fetch (or use given HTML), extract, select quotes with the LLM, verify, store."""
    try:
        if page_html is None:
            f = fetch.get(art["url"], snapshot=True)
            page_html, sha = f.text, f.sha256
        else:
            raw = page_html.encode("utf-8")
            sha = hashlib.sha256(raw).hexdigest()
            fetch.save_snapshot(sha, raw)
        a = extract(page_html, art["url"], source.get("body_xpath"))
        if not urlparse(a.url).netloc.lower().endswith(source["domain"]):
            a.url = art["url"]                      # ignore canonicals pointing off-site
        if not a.published_at:
            a.published_at = art.get("published_at")
        if not a.title:
            a.title = art.get("title") or ""
        with session() as c:
            c.execute("UPDATE articles SET title=?, published_at=?, snapshot_sha256=?, body_json=?, "
                      "body_sha256=?, fetched_at=?, attempts=attempts+1 WHERE id=?",
                      (a.title, a.published_at, sha, a.to_json(),
                       hashlib.sha256(a.to_json().encode()).hexdigest(), now_iso(), art["id"]))
        if not a.published_at:
            raise ValueError("no publication date on page")
        if a.published_at < period_start:
            _set_status(art["id"], "out_of_period")
            return "out_of_period"
        if not relevance.classify(a.title, a.standfirst, a.url, tagged=art["relevance_rule"] == "topic_tag"):
            _set_status(art["id"], "irrelevant")      # deterministic re-check on the real headline
            return "irrelevant"
        if not a.blocks:
            raise ValueError("no body text extracted")

        picks, call_id = select_quotes(llm, a.title, source["name"], a.published_at, a.blocks)
        # independent re-extraction from the stored snapshot, used by verification
        snap_blocks = extract(fetch.load_snapshot(sha), art["url"], source.get("body_xpath")).blocks
        with session() as c:
            stored, rejected = _store_quotes(c, art, source, a, picks, call_id, period_start, snap_blocks, run_id)
            status = "extracted" if stored else ("irrelevant" if not picks.get("article_relevant") else "no_quotes")
            c.execute("UPDATE articles SET status=?, processed_at=?, llm_call_id=?, error=NULL, run_id=? "
                      "WHERE id=?", (status, now_iso(), call_id, run_id, art["id"]))
        log("article.processed", url=art["url"], status=status, quotes=stored, rejected=rejected,
            extractor=a.extractor)
        return status
    except fetch.FetchError as e:
        _fail(art, str(e), permanent=not e.retryable)
        return "error"
    except LLMError as e:
        _fail(art, f"llm: {e}", permanent=False)
        if "authentication" in str(e):
            raise
        return "error"
    except Exception as e:  # noqa: BLE001 - one bad page must not stop the run
        _fail(art, f"{type(e).__name__}: {e}", permanent=False)
        return "error"


def _set_status(article_id: int, status: str) -> None:
    with session() as c:
        c.execute("UPDATE articles SET status=?, processed_at=? WHERE id=?", (status, now_iso(), article_id))


def _fail(art: dict, error: str, permanent: bool) -> None:
    with session() as c:
        c.execute("UPDATE articles SET status=CASE WHEN ? OR attempts>=? THEN 'error' ELSE 'detected' END, "
                  "error=?, attempts=attempts+CASE WHEN fetched_at IS NULL THEN 1 ELSE 0 END WHERE id=?",
                  (int(permanent), MAX_ATTEMPTS, error, art["id"]))
    log("article.error", "error", url=art["url"], error=error)


def process_pending(source: dict, period_start: str, run_id: str, llm: LLM | None = None,
                    limit: int | None = None) -> dict:
    llm = llm or default_llm()
    with session() as c:
        pending = rows(c, "SELECT * FROM articles WHERE source_id=? AND status='detected' AND attempts<? "
                          "AND (published_at IS NULL OR published_at>=?) ORDER BY published_at DESC NULLS LAST, id",
                       source["id"], MAX_ATTEMPTS, period_start)
    counts: dict[str, int] = {}
    for art in pending[:limit]:
        s = process_article(art, source, llm, period_start, run_id)
        counts[s] = counts.get(s, 0) + 1
    return {"source": source["id"], "processed": sum(counts.values()), **counts}


# --------------------------------------------------------------------------- entry points
def collect(day_from: date, day_to: date, source_ids: list[str] | None = None, kind: str = "backfill",
            llm: LLM | None = None, limit: int | None = None) -> dict:
    sync_sources()
    srcs = [s for s in sources_config() if not source_ids or s["id"] in source_ids]
    rid = _start_run(kind, {"from": day_from.isoformat(), "to": day_to.isoformat(),
                            "sources": [s["id"] for s in srcs]})
    stats: dict = {"detect": [], "process": []}
    status = "ok"
    try:
        for s in srcs:
            stats["detect"].append(detect(s, day_from, day_to, rid))
            if s.get("access", "open") == "open":
                stats["process"].append(process_pending(s, day_from.isoformat(), rid, llm, limit))
    except LLMError as e:
        status, stats["error"] = "failed", str(e)
    except Exception as e:  # noqa: BLE001
        status, stats["error"] = "failed", f"{type(e).__name__}: {e}"
        log("run.error", "error", error=stats["error"])
    with session() as c:
        errs = one(c, "SELECT COUNT(*) n FROM articles WHERE run_id=? AND status='error'", rid)["n"]
    if status == "ok" and errs:
        status = "partial"
    _finish_run(rid, status, stats)
    return {"run_id": rid, "status": status, **stats}


def daily(llm: LLM | None = None) -> dict:
    """Check every source for the last 2 days (overlap makes missed runs self-healing)."""
    today = date.today()
    return collect(today - timedelta(days=2), today, kind="daily", llm=llm)


def import_inbox(llm: LLM | None = None) -> dict:
    """Process pages saved by a licensed reader (for detect-only sources such as Reuters)."""
    sync_sources()
    rid = _start_run("import", {"inbox": str(INBOX_DIR)})
    done = []
    llm = llm or default_llm()
    for path in sorted(INBOX_DIR.glob("*.htm*")):
        page = path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"', page) or \
            re.search(r'<meta[^>]+property="og:url"[^>]+content="([^"]+)"', page)
        if not m:
            log("inbox.no_url", "warning", file=path.name)
            continue
        url = discovery.normalise_url(m.group(1))
        host = urlparse(url).netloc.lower()
        src = next((s for s in sources_config() if host == s["domain"] or host.endswith("." + s["domain"])), None)
        if not src:
            log("inbox.unknown_source", "warning", file=path.name, url=url)
            continue
        with session() as c:
            c.execute("INSERT OR IGNORE INTO articles(source_id,url,title,discovered_at,discovery_method,"
                      "relevance_rule,status,run_id) VALUES (?,?,?,?,?,?,?,?)",
                      (src["id"], url, "", now_iso(), "inbox", "manual_import", "detected", rid))
            art = one(c, "SELECT * FROM articles WHERE url=?", url)
        status = process_article(art, src, llm, DEFAULT_START, rid, page_html=page)
        done.append({"file": path.name, "url": url, "status": status})
        path.rename(path.with_suffix(path.suffix + ".done"))
    _finish_run(rid, "ok", {"imported": done})
    return {"run_id": rid, "imported": done}


def add_source(name: str, homepage: str, topic_url: str | None = None, backfill_from: str = DEFAULT_START,
               llm: LLM | None = None, run_backfill: bool = True) -> dict:
    """Build a pipeline for a new media / think-tank source automatically:
    robots.txt + feed autodetection -> registry entry -> backfill from `backfill_from`."""
    entry = register_source(name, homepage, topic_url)
    result = {"source": entry}
    if run_backfill:
        result["backfill"] = backfill_source(entry["id"], backfill_from, llm)
    return result


def backfill_source(source_id: str, backfill_from: str = DEFAULT_START, llm: LLM | None = None) -> dict:
    return collect(date.fromisoformat(backfill_from), date.today(), [source_id], "add_source", llm)


_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")


def _site_url(url: str | None, what: str) -> tuple[str, str]:
    """Normalise a user-typed web address. Returns (full https/http URL, lower-case host)."""
    typed = url = (url or "").strip()
    if not url:
        raise ValueError(f"{what} is required")
    if "://" not in url:
        url = f"https://{url}"
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
    except ValueError:
        p, host = None, ""
    if p is None or p.scheme.lower() not in ("http", "https") or not _HOST_RE.match(host):
        raise ValueError(f"{what} is not a valid web address: {typed!r} (expected e.g. https://www.example.org)")
    return url, host


def _blocked(e: "fetch.FetchError") -> bool:
    """The publisher refuses automated access (as opposed to a typo / dead page / network problem)."""
    return e.reason == "disallowed by robots.txt" or e.status in (401, 403)


def _why(e: "fetch.FetchError") -> str:
    if e.reason == "disallowed by robots.txt":
        return "its robots.txt does not allow automated collection"
    if e.status in (401, 403):
        return f"the site refuses automated access (HTTP {e.status})"
    if e.status:
        return f"the page answered HTTP {e.status}"
    return f"the site could not be reached ({e.reason})"


def register_source(name: str, homepage: str, topic_url: str | None = None) -> dict:
    """Validate the input, work out how to list the source's articles and add it to
    config/sources.yaml (no backfill). Raises ValueError with a human-readable message when the
    source cannot be added.

    Discovery, in order: news sitemaps in robots.txt, RSS/Atom <link>s on the homepage, the topic page
    the user gave (an editorial Ukraine tag), the page pasted as homepage if it is a section/article
    link, conventional feed paths, and finally the homepage itself as an article listing. Pages that
    are not a topic page are 'untagged': the deterministic relevance rules decide what is in scope.
    A publisher that refuses automated access is registered as inbox-only (like Reuters)."""
    name = (name or "").strip()
    if not name:
        raise ValueError("source name is required")
    sid = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not sid:
        raise ValueError(f"source name {name!r} must contain at least one Latin letter or digit")
    home_url, host = _site_url(homepage, "homepage")
    domain = host.removeprefix("www.")
    if (topic_url or "").strip():
        topic_url, topic_host = _site_url(topic_url, "topic page URL")
        if topic_host.removeprefix("www.") != domain and not topic_host.endswith("." + domain):
            raise ValueError(f"the topic page URL must be on the same site as the homepage ({domain})")
    else:
        topic_url = None
    pasted = home_url if urlparse(home_url).path.strip("/") and home_url != topic_url else None

    for s in sources_config():
        if s["id"] == sid or str(s.get("name", "")).strip().lower() == name.lower():
            raise ValueError(f"a source named {s['name']!r} (id {s['id']!r}) already exists")
        if str(s.get("domain", "")).lower().removeprefix("www.") == domain:
            raise ValueError(f"{domain} is already registered as {s['name']!r}")

    home_allowed = fetch.robots_allows(f"https://{host}/")
    disc = discovery.autodetect_feeds(f"https://{host}", probe_common=not (topic_url or pasted))
    listings = []
    if topic_url:
        listings.append({"method": "topic_pages", "url": topic_url, "max_pages": 40})
    if pasted:
        listings.append({"method": "topic_pages", "url": pasted, "max_pages": 3, "tagged": False})
    if not disc and not listings:
        listings.append({"method": "topic_pages", "url": f"https://{host}/", "max_pages": 1, "tagged": False})
    errors: list[fetch.FetchError] = []
    for lst in listings:                     # keep only listing pages we are allowed to and can read
        try:
            fetch.get(lst["url"])
        except fetch.FetchError as e:
            log("add_source.listing_unreadable", "warning", url=lst["url"], error=str(e))
            if lst["url"] == topic_url and not _blocked(e):
                raise ValueError(f"cannot read the topic page {topic_url}: {_why(e)}") from None
            errors.append(e)
            continue
        disc.append(lst)

    note = None
    if disc:
        access = "open" if home_allowed else "detect_only"
    elif errors and all(_blocked(e) for e in errors):
        access, note = "detect_only", (
            f"{domain}: {_why(errors[0])}, and it publishes no feed or news sitemap, so nothing is crawled. "
            "Articles are processed from pages a reader saves into ./inbox ('Import ./inbox pages').")
    elif errors:
        raise ValueError(f"cannot read {errors[0].url}: {_why(errors[0])}. Check the address")
    else:
        raise ValueError(f"could not find any way to list articles on {domain}")

    entry = {"id": sid, "name": name, "domain": domain, "access": access, "mandatory": False}
    if note:
        entry["access_note"] = note
    entry |= {"discovery": disc, "url_include": [rf"^https://([a-z0-9-]+\.)?{re.escape(domain)}/"],
              "url_exclude": ["/video/", "/videos/", "/live/", "/gallery/", "/podcast", "/tag/", "/author/"]}
    append_source_config(entry)
    sync_sources()
    log("add_source", source=sid, discovery=disc, access=access)
    return entry
