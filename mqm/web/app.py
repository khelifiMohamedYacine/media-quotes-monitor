"""Dashboard: latest reports, logs, database browser, and controls to run the pipeline."""
from __future__ import annotations

import json
import re
import threading
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import markdown
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import export, pipeline, report
from ..config import DB_PATH, DEFAULT_START, REPORT_DIR, domain_config, sources_config
from ..db import init_db, one, rows, session
from ..logs import log, read_logs

app = FastAPI(title="Media Quotes Monitor")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["fromjson"] = json.loads


def _rule_label(rule: str | None) -> str:
    r"""Human-readable form of a relevance rule id, e.g. 'core:\bukrain' -> 'keyword "ukrain"'."""
    if not rule:
        return ""

    def clean(pattern: str) -> str:   # drop regex syntax: \b, \w, anchors, groups, quantifiers
        return re.sub(r"\\[bBwWsSdD]|[\^$()?*+\[\]{}|\\]", "", pattern).strip()

    if rule == "topic_tag":
        return "listed on the site's Ukraine topic page"
    if rule == "manual_import":
        return "imported from ./inbox"
    if rule.startswith("core:"):
        return f'keyword "{clean(rule[5:])}"'
    if rule.startswith("russia+context:"):
        ru, _, ctx = rule[15:].partition("&")
        return f'keywords "{clean(ru)}" + "{clean(ctx)}"'
    return rule


templates.env.filters["rule_label"] = _rule_label
templates.env.globals["today_label"] = lambda: f"{date.today():%A}, {date.today().day} {date.today():%B %Y}"
_job_lock = threading.Lock()
_job: dict = {"name": None, "last": None}


def _run_job(name: str, fn, *args, **kwargs) -> bool:
    """Run one pipeline job at a time in a background thread."""
    if not _job_lock.acquire(blocking=False):
        return False
    _job["name"] = name

    def target():
        try:
            out = fn(*args, **kwargs)
            failed = isinstance(out, dict) and out.get("status") == "failed"
            _job["last"] = {"name": name, "ok": not failed,
                            "error": out.get("error", "") if failed else ""}
        except Exception as e:  # noqa: BLE001
            log("dashboard.job_failed", "error", job=name, error=str(e))
            _job["last"] = {"name": name, "ok": False, "error": str(e)}
        finally:
            _job["name"] = None
            _job_lock.release()

    threading.Thread(target=target, daemon=True).start()
    return True


@app.on_event("startup")
def _startup() -> None:
    init_db()
    pipeline.sync_sources()


def _sections() -> dict[str, str]:
    return {s["id"]: s["title"] for s in domain_config()["sections"]}


@app.get("/", response_class=HTMLResponse)
def home(request: Request, msg: str = "", err: str = "", name: str = "", homepage: str = "", topic_url: str = ""):
    with session() as c:
        by_status = rows(c, "SELECT source_id, status, COUNT(*) n FROM articles GROUP BY 1,2 ORDER BY 1,2")
        quotes = one(c, "SELECT COUNT(*) total, SUM(verified) verified FROM quotes")
        by_section = rows(c, "SELECT section_id, COUNT(*) n FROM quotes WHERE verified=1 GROUP BY 1 ORDER BY 2 DESC")
        runs = rows(c, "SELECT * FROM runs ORDER BY started_at DESC LIMIT 10")
        cost = one(c, "SELECT COUNT(*) calls, ROUND(COALESCE(SUM(cost_usd),0),3) usd, "
                      "COALESCE(SUM(input_tokens),0) inp, COALESCE(SUM(output_tokens),0) outp FROM llm_calls")
        reports = rows(c, "SELECT * FROM reports ORDER BY report_date DESC LIMIT 7")
        latest = rows(c, "SELECT * FROM quotes WHERE verified=1 ORDER BY publication_date DESC, extraction_date DESC "
                         "LIMIT 6")
    status_table: dict[str, dict[str, int]] = {}
    for r in by_status:
        status_table.setdefault(r["source_id"], {})[r["status"]] = r["n"]
    statuses = sorted({r["status"] for r in by_status})
    return templates.TemplateResponse(request, "home.html", {
        "status_table": status_table, "statuses": statuses, "quotes": quotes, "by_section": by_section,
        "sections": _sections(), "runs": runs, "cost": cost, "reports": reports, "latest": latest, "sources": sources_config(),
        "job": _job["name"], "last_job": _job["last"], "msg": msg, "err": err,
        "form": {"name": name, "homepage": homepage, "topic_url": topic_url}, "default_start": DEFAULT_START, "today": date.today().isoformat()})


# ------------------------------------------------------------------ actions
@app.post("/run/daily")
def run_daily():
    def job():
        pipeline.daily()
        report.build_report()
        export.export_all()
    ok = _run_job("daily run", job)
    return RedirectResponse(f"/?msg={'Daily run started' if ok else 'A job is already running'}", 303)


@app.post("/run/report")
def run_report():
    ok = _run_job("report", report.build_report)
    return RedirectResponse(f"/?msg={'Report build started' if ok else 'A job is already running'}", 303)


@app.post("/run/backfill")
def run_backfill(day_from: str = Form(...), day_to: str = Form(...), source: str = Form("")):
    try:
        f, t = date.fromisoformat(day_from), date.fromisoformat(day_to)
    except ValueError:
        raise HTTPException(400, "dates must be YYYY-MM-DD")
    if f > t:
        raise HTTPException(400, "'from' must be before 'to'")
    ok = _run_job(f"backfill {f}..{t}", pipeline.collect, f, t, [source] if source else None, "backfill")
    return RedirectResponse(f"/?msg={'Backfill started' if ok else 'A job is already running'}", 303)


@app.post("/run/add-source")
def run_add_source(name: str = Form(...), homepage: str = Form(...), topic_url: str = Form("")):
    # Register synchronously so validation / feed-detection errors are shown to the user,
    # then backfill in the background (that part can take a long time).
    if _job["name"]:
        return RedirectResponse(f"/?{urlencode({'err': 'A job is already running; try again when it finishes'})}",
                                303)
    try:
        entry = pipeline.register_source(name, homepage, topic_url or None)
    except ValueError as e:
        log("dashboard.add_source_rejected", "warning", name=name, homepage=homepage, error=str(e))
        keep = {"name": name, "homepage": homepage, "topic_url": topic_url}   # refill the form
        return RedirectResponse(f"/?{urlencode({'err': f'Could not add source: {e}', **keep})}#add-source", 303)
    return RedirectResponse(f"/?{urlencode({'msg': _added_message(entry)})}", 303)


_METHOD_NAMES = {"news_sitemap": "news sitemap", "rss": "RSS feed", "topic_pages": "topic page"}


def _added_message(entry: dict) -> str:
    """Plain-language confirmation for a newly registered source; starts its backfill when useful."""
    name = entry["name"]
    if not entry["discovery"]:
        # Publisher refuses automated access and has no feed: nothing to crawl, so no backfill.
        return (f"{name} added as an inbox-only source: the site refuses automated access, so it is not "
                "crawled. Save its articles into ./inbox and click “Import ./inbox pages” to extract quotes.")
    found = sorted({_METHOD_NAMES.get(d["method"], d["method"]) for d in entry["discovery"]})
    msg = f"{name} added. Articles are found through its {' and '.join(found)}."
    if entry["access"] == "detect_only":
        msg += " Its robots.txt does not allow crawling, so new articles are listed but not downloaded."
    if _run_job(f"backfill new source {name}", pipeline.backfill_source, entry["id"]):
        msg += (f" Collecting its articles since {DEFAULT_START} in the background; quotes appear on the "
                "Quotes page as each article is processed.")
    else:
        msg += " Another job is running, so collection has not started yet: use “Collect a past period” later."
    return msg


@app.post("/run/audit")
def run_audit():
    ok = _run_job("re-check all quotes", export.audit_integrity, apply=True)
    return RedirectResponse(f"/?msg={'Re-checking every quote against its saved page' if ok else 'A job is already running'}",
                            303)


# What each deterministic check means (shown on the quote page)
CHECK_HELP = {
    "verbatim_in_article": "The exact text is in the extracted article body.",
    "verbatim_in_snapshot": "The exact text is found again when the saved raw page is re-read independently.",
    "belongs_to_story": "The text is part of this story, not a feed, promotion or summary the site repeats on many pages.",
    "length_ok": "Between 40 and 1,500 characters.",
    "no_boilerplate": "Not navigation, captions or promotional text.",
    "complete_sentence_end": "Ends at a sentence boundary.",
    "metadata_complete": "Source, dates, headline, link and section are all present.",
    "dates_valid": "Publication date is valid and not after the extraction date.",
    "url_matches_source": "The article link belongs to the source's own domain.",
    "in_period": "Published inside the collection period.",
}


@app.post("/run/import-inbox")
def run_import():
    ok = _run_job("import inbox", pipeline.import_inbox)
    return RedirectResponse(f"/?msg={'Inbox import started' if ok else 'A job is already running'}", 303)


# ------------------------------------------------------------------ reports
@app.get("/reports", response_class=HTMLResponse)
def reports_list(request: Request):
    with session() as c:
        reps = rows(c, "SELECT * FROM reports ORDER BY report_date DESC")
    return templates.TemplateResponse(request, "reports.html", {"reports": reps})


@app.get("/reports/{day}", response_class=HTMLResponse)
def report_view(request: Request, day: str):
    path = REPORT_DIR / ("latest.md" if day == "latest" else f"{date.fromisoformat(day).isoformat()}.md")
    if not path.exists():
        raise HTTPException(404, "no report for that date")
    body = markdown.markdown(path.read_text(encoding="utf-8"), extensions=["toc", "tables"])
    return templates.TemplateResponse(request, "report.html", {"body": body, "day": day})


@app.get("/reports/{day}.md")
def report_raw(day: str):
    path = REPORT_DIR / f"{day}.md"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="text/markdown", filename=path.name)


# ------------------------------------------------------------------ database
@app.get("/quotes", response_class=HTMLResponse)
def quotes_view(request: Request, section: str = "", source: str = "", q: str = "", since: str = "",
                verified: str = "1", page: int = 1):
    where, params = ["1=1"], []
    if verified in ("0", "1"):
        where.append("verified=?"); params.append(int(verified))
    if section:
        where.append("section_id=?"); params.append(section)
    if source:
        where.append("source_id=?"); params.append(source)
    if q:
        where.append("(quote_plain LIKE ? OR article_name LIKE ? OR speaker LIKE ?)"); params += [f"%{q}%"] * 3
    if since:
        where.append("publication_date>=?"); params.append(since)
    with session() as c:
        total = one(c, f"SELECT COUNT(*) n FROM quotes WHERE {' AND '.join(where)}", *params)["n"]
        items = rows(c, f"SELECT * FROM quotes WHERE {' AND '.join(where)} ORDER BY publication_date DESC "
                        f"LIMIT 50 OFFSET ?", *params, (page - 1) * 50)
    return templates.TemplateResponse(request, "quotes.html", {
        "items": items, "total": total, "page": page, "sections": _sections(), "sources": sources_config(),
        "f": {"section": section, "source": source, "q": q, "since": since, "verified": verified}})


@app.get("/quotes/{quote_id}", response_class=HTMLResponse)
def quote_detail(request: Request, quote_id: str):
    with session() as c:
        qrow = one(c, "SELECT * FROM quotes WHERE quote_id=?", quote_id)
        if not qrow:
            raise HTTPException(404)
        art = one(c, "SELECT id,url,status,snapshot_sha256,fetched_at,discovery_method,relevance_rule,run_id "
                     "FROM articles WHERE id=?", qrow["article_id"])
        call = one(c, "SELECT * FROM llm_calls WHERE id=?", qrow["llm_call_id"]) if qrow["llm_call_id"] else None
        hist = rows(c, "SELECT * FROM assessments WHERE quote_id=? ORDER BY report_date DESC", quote_id)
    return templates.TemplateResponse(request, "quote.html", {"q": qrow, "art": art, "call": call,
                                                              "hist": hist, "sections": _sections(),
                                                              "check_help": CHECK_HELP})


@app.get("/articles", response_class=HTMLResponse)
def articles_view(request: Request, status: str = "", source: str = "", page: int = 1):
    where, params = ["1=1"], []
    if status:
        where.append("status=?"); params.append(status)
    if source:
        where.append("source_id=?"); params.append(source)
    with session() as c:
        items = rows(c, f"SELECT a.id,a.source_id,a.url,a.title,a.published_at,a.status,a.discovery_method,"
                        f"a.relevance_rule,a.error,(SELECT COUNT(*) FROM quotes q WHERE q.article_id=a.id "
                        f"AND q.verified=1) nq FROM articles a WHERE {' AND '.join(where)} "
                        f"ORDER BY published_at DESC NULLS LAST LIMIT 100 OFFSET ?", *params, (page - 1) * 100)
        statuses = [r["status"] for r in rows(c, "SELECT DISTINCT status FROM articles ORDER BY 1")]
    return templates.TemplateResponse(request, "articles.html", {
        "items": items, "statuses": statuses, "sources": sources_config(), "page": page,
        "f": {"status": status, "source": source}})


@app.get("/logs", response_class=HTMLResponse)
def logs_view(request: Request, level: str = "", run_id: str = "", day: str = ""):
    recs = read_logs(day or None, 500, level or None, run_id or None)
    return templates.TemplateResponse(request, "logs.html", {"recs": recs, "f": {"level": level, "run_id": run_id,
                                                                                   "day": day}})


@app.get("/llm-calls", response_class=HTMLResponse)
def llm_calls(request: Request):
    with session() as c:
        calls = rows(c, "SELECT id,run_id,purpose,model,effort,input_tokens,output_tokens,cache_read_tokens,"
                        "cost_usd,stop_reason,error,created_at FROM llm_calls ORDER BY id DESC LIMIT 300")
    return templates.TemplateResponse(request, "llm_calls.html", {"calls": calls})


# ------------------------------------------------------------------ machine access
@app.get("/api/quotes")
def api_quotes(since: str = "", section: str = "", limit: int = 500):
    with session() as c:
        items = rows(c, "SELECT * FROM quotes WHERE verified=1 AND publication_date>=? AND (?='' OR section_id=?) "
                        "ORDER BY publication_date DESC LIMIT ?", since, section, section, limit)
    for i in items:
        i["key_facts"] = json.loads(i.pop("key_facts_json"))
        i["verification"] = json.loads(i.pop("verification_json"))
    return JSONResponse(items)


@app.get("/api/export/{name}")
def api_export(name: str):
    export.export_all()
    files = {"quotes.jsonl": "application/x-ndjson", "context_pack.md": "text/markdown"}
    if name not in files:
        raise HTTPException(404)
    from ..config import EXPORT_DIR
    return FileResponse(EXPORT_DIR / name, media_type=files[name], filename=name)


@app.get("/api/audit")
def api_audit():
    return JSONResponse(export.audit_integrity())


@app.get("/download/db")
def download_db():
    return FileResponse(DB_PATH, media_type="application/x-sqlite3", filename="quotes.db")


@app.get("/api/status")
def api_status():
    """Background job state, polled by the dashboard to refresh itself when a job finishes."""
    return JSONResponse({"job": _job["name"], "last": _job["last"]})


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"
