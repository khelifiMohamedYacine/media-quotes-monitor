"""Command line: python -m mqm <command>"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

from . import export, pipeline, report
from .config import DEFAULT_START
from .db import init_db


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mqm", description="Media Quotes Monitor")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create the database and register sources")
    d = sub.add_parser("daily", help="daily run: detect + extract (last 2 days) and build today's report")
    d.add_argument("--no-report", action="store_true")
    b = sub.add_parser("backfill", help="collect quotes for a past period")
    b.add_argument("--from", dest="day_from", default=DEFAULT_START)
    b.add_argument("--to", dest="day_to", default=date.today().isoformat())
    b.add_argument("--source", action="append", help="limit to source id(s)")
    b.add_argument("--limit", type=int, help="max articles per source (testing)")
    r = sub.add_parser("report", help="build the Markdown report for a date")
    r.add_argument("--date", default=date.today().isoformat())
    a = sub.add_parser("add-source", help="add a media/think-tank source and backfill it")
    a.add_argument("name")
    a.add_argument("homepage")
    a.add_argument("--topic-url")
    a.add_argument("--from", dest="day_from", default=DEFAULT_START)
    a.add_argument("--no-backfill", action="store_true")
    sub.add_parser("import-inbox", help="process saved article pages in ./inbox")
    sub.add_parser("export", help="write quotes.jsonl and context_pack.md")
    au = sub.add_parser("audit", help="re-verify all quotes against stored snapshots")
    au.add_argument("--apply", action="store_true", help="store the results (mark failing quotes unverified)")
    s = sub.add_parser("serve", help="run the dashboard")
    # Hosting platforms (e.g. Render) set PORT and need the server to listen on all interfaces.
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    s.add_argument("--host", default=os.environ.get("MQM_HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"))
    args = p.parse_args(argv)

    if args.cmd == "init":
        init_db()
        pipeline.sync_sources()
        out = {"ok": True}
    elif args.cmd == "daily":
        out = pipeline.daily()
        if not args.no_report:
            out["report"] = report.build_report()
            out["export"] = export.export_all()
    elif args.cmd == "backfill":
        out = pipeline.collect(date.fromisoformat(args.day_from), date.fromisoformat(args.day_to),
                               args.source, "backfill", limit=args.limit)
    elif args.cmd == "report":
        out = report.build_report(date.fromisoformat(args.date))
    elif args.cmd == "add-source":
        out = pipeline.add_source(args.name, args.homepage, args.topic_url, args.day_from,
                                  run_backfill=not args.no_backfill)
    elif args.cmd == "import-inbox":
        out = pipeline.import_inbox()
    elif args.cmd == "export":
        out = export.export_all()
    elif args.cmd == "audit":
        out = export.audit_integrity(apply=args.apply)
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("mqm.web.app:app", host=args.host, port=args.port)
        return
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
