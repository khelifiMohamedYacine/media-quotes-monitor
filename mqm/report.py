"""Daily Markdown report: non-obsolete material quotes for the rolling 14-day window.

Probabilistic steps (LLM): (1) judge which quotes are superseded by newer facts and how
material each is; (2) write a 1-2 sentence inference per section.
Deterministic steps (code): window selection, validation of every LLM judgement, ranking,
and copying quote text + metadata from the database into the file byte-for-byte.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone

from .config import REPORT_DIR, SETTINGS, domain_config
from .db import dumps, rows, session
from .llm import LLM, LLMError, default_llm
from .logs import log, now_iso

OBS_CHUNK = 40   # max quotes per obsolescence request

OBSOLESCENCE_SYSTEM = """You maintain a briefing of media quotes on Russia's war on Ukraine.
You get every verified quote published in the last 14 days, oldest first, each with an id,
date, section, topic key and the facts it states.

For EVERY quote decide:
- status "superseded": a NEWER quote in the list reports facts that replace this quote's facts
  (e.g. an updated casualty toll, talks that have since happened or collapsed, a decision that
  was since taken, a forecast overtaken by events). Give superseded_by = the id of that newer
  quote. Only supersede on genuine factual replacement, not because a topic was mentioned again.
- status "duplicate": it says substantially the same thing as another quote that is at least as
  informative; superseded_by = the id of the quote to keep.
- status "current": still accurate and useful.
- materiality 1-5: how much the quote helps understand the war's current trends and
  expectations (5 = essential).
- reason: one short sentence justifying the status.
Return one entry per input quote."""

OBSOLESCENCE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["assessments"],
    "properties": {"assessments": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["quote_id", "status", "superseded_by", "materiality", "reason"],
        "properties": {
            "quote_id": {"type": "string"},
            "status": {"type": "string", "enum": ["current", "superseded", "duplicate"]},
            "superseded_by": {"type": "string"},
            "materiality": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "reason": {"type": "string"},
        }}}},
}

SUMMARY_SYSTEM = """For each section of a briefing on Russia's war on Ukraine you receive the
selected media quotes. Write ONE or TWO sentences (max 45 words) stating what the quotes,
taken together, indicate about the current trend or expectation -- e.g. "Commentators highlight
facts indicating that the war in Ukraine is intensifying." Attribute ("Reports indicate...",
"Officials signal...") rather than asserting as fact. Use only information in the quotes."""

SUMMARY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["summaries"],
    "properties": {"summaries": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["section_id", "summary"],
        "properties": {"section_id": {"type": "string"}, "summary": {"type": "string"}}}}},
}


def _window(report_date: date) -> tuple[str, str, date]:
    days = domain_config()["report"]["window_days"]
    start = report_date - timedelta(days=days - 1)
    end_excl = report_date + timedelta(days=1)
    return start.isoformat(), end_excl.isoformat(), start


def _fmt_day(d: date) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def _validate_assessments(raw: list[dict], quotes: dict[str, dict]) -> dict[str, dict]:
    """Deterministic guard-rails on the LLM's judgement."""
    out: dict[str, dict] = {}
    for a in raw:
        qid = a.get("quote_id")
        if qid not in quotes or qid in out:
            continue
        status, by = a["status"], a.get("superseded_by") or None
        if status in ("superseded", "duplicate"):
            newer_ok = by in quotes and by != qid and (
                status == "duplicate" or quotes[by]["publication_date"] >= quotes[qid]["publication_date"])
            if not newer_ok:
                log("report.assessment_overruled", "warning", quote_id=qid, status=status, superseded_by=by)
                status, by = "current", None
        out[qid] = {"status": status, "superseded_by": by, "materiality": int(a.get("materiality", 3)),
                    "reason": a.get("reason", "")}
    for qid in quotes:  # anything the model skipped stays visible (fail-open for recall)
        out.setdefault(qid, {"status": "current", "superseded_by": None, "materiality": 3,
                             "reason": "not assessed by model; kept by default"})
    # break chains/cycles: a quote that supersedes others must itself be current to remove them
    for qid, a in out.items():
        if a["status"] != "current" and out.get(a["superseded_by"], {}).get("status") != "current":
            target = a["superseded_by"]
            seen = {qid}
            while target and out[target]["status"] != "current" and target not in seen:
                seen.add(target)
                target = out[target]["superseded_by"]
            if not target or target in seen:
                a.update(status="current", superseded_by=None, reason=a["reason"] + " [cycle broken]")
    return out


def _valid_summary(s: str) -> bool:
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]
    return 1 <= len(sentences) <= 2 and len(s.split()) <= 60


def _quote_md(q: dict) -> str:
    lines = [f"> {ln}" if ln else ">" for ln in q["quote_text"].split("\n")]
    pub = q["publication_date"][:10]
    who = f"{q['speaker']}, as quoted in " if q.get("speaker") else ""
    lines.append(">")
    lines.append(f"> — {who}**{q['source_name']}**, {pub}: "
                 f"[{q['article_name']}]({q['article_url']}) · `{q['quote_id']}`")
    return "\n".join(lines)


def build_report(report_date: date | None = None, llm: LLM | None = None) -> dict:
    report_date = report_date or datetime.now(timezone.utc).date()
    llm = llm or default_llm()
    start, end_excl, start_day = _window(report_date)
    sections = domain_config()["sections"]
    max_per = domain_config()["report"]["max_quotes_per_section"]

    with session() as c:
        qs = rows(c, "SELECT * FROM quotes WHERE verified=1 AND publication_date>=? AND publication_date<? "
                     "ORDER BY publication_date, quote_id", start, end_excl)
        sources = [r["name"] for r in rows(c, "SELECT name FROM sources ORDER BY mandatory DESC, name")]
    quotes = {q["quote_id"]: q for q in qs}
    log("report.start", report_date=report_date.isoformat(), window_start=start, quotes=len(qs))

    assessed: dict[str, dict] = {}
    obs_calls: dict[str, int | None] = {}
    if qs:
        # One call per section (and per chunk of OBS_CHUNK quotes) keeps each request small enough
        # for free-tier token limits; quotes are compared with others on the same matters, which
        # in practice share a section. Payload is compact: facts + topic + a short text excerpt.
        raw: list[dict] = []
        for s in sections:
            sec = [q for q in qs if q["section_id"] == s["id"]]
            for i in range(0, len(sec), OBS_CHUNK):
                chunk = sec[max(0, i - 5):i + OBS_CHUNK]          # 5-quote overlap with the previous chunk
                payload = [{"id": q["quote_id"], "date": q["publication_date"][:10], "topic": q["topic_key"],
                            "facts": json.loads(q["key_facts_json"]), "text": q["quote_plain"][:220]}
                           for q in chunk]
                try:
                    data, call = llm.json("obsolescence", OBSOLESCENCE_SYSTEM,
                                          json.dumps(payload, ensure_ascii=False), OBSOLESCENCE_SCHEMA,
                                          SETTINGS.llm.effort_obsolescence, max_tokens=6000)
                    raw += data["assessments"]
                    obs_calls[s["id"]] = call
                except LLMError as e:
                    log("report.obsolescence_failed", "error", section=s["id"], error=str(e))
        assessed = _validate_assessments(raw, quotes)

    # deterministic selection: current quotes, most material first, newest first, capped per section
    chosen: dict[str, list[dict]] = {s["id"]: [] for s in sections}
    for sid in chosen:
        cur = [q for q in qs if q["section_id"] == sid and assessed[q["quote_id"]]["status"] == "current"]
        cur.sort(key=lambda q: (assessed[q["quote_id"]]["materiality"], q["publication_date"]), reverse=True)
        chosen[sid] = cur[:max_per]

    summaries: dict[str, str] = {}
    sum_calls: dict[str, int | None] = {}
    for s in sections:
        if not chosen[s["id"]]:
            continue
        payload = {s["id"]: {"title": s["title"], "quotes": [q["quote_plain"] for q in chosen[s["id"]]]}}
        try:
            data, sum_calls[s["id"]] = llm.json("section_summaries", SUMMARY_SYSTEM,
                                                json.dumps(payload, ensure_ascii=False), SUMMARY_SCHEMA,
                                                SETTINGS.llm.effort_summary, max_tokens=2000)
            for item in data["summaries"]:
                if item["section_id"] == s["id"] and _valid_summary(item["summary"]):
                    summaries[s["id"]] = item["summary"].strip()
                else:
                    log("report.summary_rejected", "warning", item=item)
        except LLMError as e:
            log("report.summary_failed", "error", section=s["id"], error=str(e))

    # ---------------------------------------------------------------- render (deterministic)
    period = f"{_fmt_day(start_day)} – {_fmt_day(report_date)}"
    shown = sum(len(v) for v in chosen.values())
    hidden = sum(1 for a in assessed.values() if a["status"] != "current")
    out = [f"# Russia's War on Ukraine — Material Media Quotes, {period}", "",
           f"*Rolling {domain_config()['report']['window_days']}-day window. Generated {now_iso()} UTC. "
           f"Sources: {', '.join(sources)}.*  ",
           f"*{shown} quotes shown; {hidden} obsolete or duplicate quotes withheld; "
           f"{len(qs)} verified quotes in the window.*", "",
           "Quotes and their metadata are copied verbatim from the audited database. "
           "The short inference under each heading is machine-generated (LLM) from the quotes shown.", "",
           "## Contents", ""]
    out += [f"- [{s['title']}](#{re.sub(r'[^a-z0-9 -]', '', s['title'].lower()).replace(' ', '-')}) "
            f"({len(chosen[s['id']])})" for s in sections]
    for s in sections:
        out += ["", f"## {s['title']}", ""]
        if not chosen[s["id"]]:
            out += ["*No material, non-obsolete quotes in this period.*"]
            continue
        if s["id"] in summaries:
            out += [f"**Inference:** {summaries[s['id']]}", ""]
        for q in chosen[s["id"]]:
            out += [_quote_md(q), ""]
    md = "\n".join(out).rstrip() + "\n"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{report_date.isoformat()}.md"
    path.write_text(md, encoding="utf-8")
    (REPORT_DIR / "latest.md").write_text(md, encoding="utf-8")
    audit = {"report_date": report_date.isoformat(), "window_start": start, "assessments": assessed,
             "shown": {k: [q["quote_id"] for q in v] for k, v in chosen.items()},
             "summaries": summaries, "llm_calls": {"obsolescence": obs_calls, "summaries": sum_calls}}
    (REPORT_DIR / f"{report_date.isoformat()}.audit.json").write_text(dumps(audit), encoding="utf-8")

    # verbatim check: every shown quote's text must appear unchanged in the file
    for v in chosen.values():
        for q in v:
            assert q["quote_text"].split("\n")[0] in md, f"verbatim check failed for {q['quote_id']}"

    shown_ids = {q["quote_id"] for v in chosen.values() for q in v}
    with session() as c:
        for qid, a in assessed.items():
            # current but cut by the per-section cap -> recorded as not_material for this report
            status = "not_material" if a["status"] == "current" and qid not in shown_ids else a["status"]
            c.execute("INSERT OR REPLACE INTO assessments(report_date,quote_id,status,superseded_by,reason,"
                      "llm_call_id) VALUES (?,?,?,?,?,?)",
                      (report_date.isoformat(), qid, status, a["superseded_by"], a["reason"],
                       obs_calls.get(quotes[qid]["section_id"])))
        c.execute("INSERT OR REPLACE INTO reports(report_date,period_start,period_end,path,sha256,created_at,stats_json) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (report_date.isoformat(), start, report_date.isoformat(), str(path),
                   hashlib.sha256(md.encode()).hexdigest(), now_iso(),
                   dumps({"shown": shown, "withheld": hidden, "in_window": len(qs)})))
    log("report.done", path=str(path), shown=shown, withheld=hidden)
    return {"path": str(path), "shown": shown, "withheld": hidden, "in_window": len(qs)}
