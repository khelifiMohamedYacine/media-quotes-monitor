"""Structured, append-only event log (JSON lines) for auditability and traceability.

Every pipeline step writes events tagged with the current run id, so any quote in the
database can be traced back to the run, the HTTP fetch and the LLM call that produced it.
"""
from __future__ import annotations

import json
import sys
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from .config import LOG_DIR

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    _run_id.set(rid)
    return rid


def current_run_id() -> str | None:
    return _run_id.get()


def log(event: str, level: str = "info", **fields) -> None:
    rec = {"ts": now_iso(), "level": level, "run_id": _run_id.get(), "event": event, **fields}
    line = json.dumps(rec, ensure_ascii=False, default=str)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{rec['ts'][:10]}.jsonl"
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    if level in ("warning", "error") or event.startswith("run."):
        print(f"[{level}] {event} " + " ".join(f"{k}={v}" for k, v in fields.items())[:300],
              file=sys.stderr)


def read_logs(day: str | None = None, limit: int = 500, level: str | None = None,
              run_id: str | None = None) -> list[dict]:
    files = sorted(LOG_DIR.glob("*.jsonl"), reverse=True)
    if day:
        files = [f for f in files if f.stem == day]
    out: list[dict] = []
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if level and rec.get("level") != level:
                continue
            if run_id and rec.get("run_id") != run_id:
                continue
            out.append(rec)
            if len(out) >= limit:
                return out
    return out
