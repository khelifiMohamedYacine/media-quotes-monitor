"""SQLite storage. One file, plain SQL, foreign keys, append-only audit tables.

Why SQLite: zero-ops, auditable with any SQL client, trivially exported to JSONL for LLM
context (see export.py), and every row links to the run, snapshot and LLM call behind it.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from .config import DB_PATH, ensure_dirs

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sources (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    domain        TEXT NOT NULL,
    access        TEXT NOT NULL,
    mandatory     INTEGER NOT NULL DEFAULT 0,
    config_json   TEXT NOT NULL,
    added_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,               -- daily | backfill | report | import | add_source
    params_json   TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,               -- running | ok | partial | failed
    stats_json    TEXT
);

CREATE TABLE IF NOT EXISTS articles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id         TEXT NOT NULL REFERENCES sources(id),
    url               TEXT NOT NULL UNIQUE,
    title             TEXT,
    standfirst        TEXT,
    published_at      TEXT,                    -- ISO 8601 UTC, from the page / feed metadata
    discovered_at     TEXT NOT NULL,
    discovery_method  TEXT NOT NULL,
    relevance_rule    TEXT NOT NULL,           -- which deterministic rule matched
    status            TEXT NOT NULL,           -- detected | extracted | no_quotes | access_restricted | out_of_period | error
    snapshot_sha256   TEXT,                    -- raw HTML in data/snapshots/<sha>.html.gz
    body_json         TEXT,                    -- extracted blocks (paragraphs with formatting)
    body_sha256       TEXT,
    fetched_at        TEXT,
    processed_at      TEXT,
    attempts          INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    llm_call_id       INTEGER REFERENCES llm_calls(id),
    run_id            TEXT
);
CREATE INDEX IF NOT EXISTS ix_articles_status ON articles(status);
CREATE INDEX IF NOT EXISTS ix_articles_pub ON articles(published_at);

CREATE TABLE IF NOT EXISTS quotes (
    quote_id            TEXT PRIMARY KEY,      -- Q-<sha256(url|text)[:12]>, deterministic
    article_id          INTEGER NOT NULL REFERENCES articles(id),
    source_id           TEXT NOT NULL REFERENCES sources(id),
    source_name         TEXT NOT NULL,
    publication_date    TEXT NOT NULL,
    article_name        TEXT NOT NULL,
    article_url         TEXT NOT NULL,
    extraction_date     TEXT NOT NULL,
    quote_text          TEXT NOT NULL,         -- verbatim, original formatting (Markdown emphasis, paragraph breaks)
    quote_plain         TEXT NOT NULL,         -- verbatim, formatting stripped
    span_start          TEXT NOT NULL,         -- first sentence id, e.g. P4.S1
    span_end            TEXT NOT NULL,
    speaker             TEXT,                  -- LLM-attributed, metadata only
    quote_type          TEXT,                  -- direct_quote | reported_fact | analysis
    section_id          TEXT NOT NULL,
    key_facts_json      TEXT NOT NULL,         -- LLM-written paraphrase of the facts (never shown as the quote)
    topic_key           TEXT,
    relevance_rationale TEXT,
    text_sha256         TEXT NOT NULL,
    verification_json   TEXT NOT NULL,
    verified            INTEGER NOT NULL,
    llm_call_id         INTEGER REFERENCES llm_calls(id),
    run_id              TEXT
);
CREATE INDEX IF NOT EXISTS ix_quotes_pub ON quotes(publication_date);
CREATE INDEX IF NOT EXISTS ix_quotes_section ON quotes(section_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT,
    purpose          TEXT NOT NULL,
    model            TEXT NOT NULL,
    effort           TEXT,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    cache_read_tokens INTEGER,
    cost_usd         REAL,
    stop_reason      TEXT,
    request_sha256   TEXT,
    response_json    TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assessments (
    report_date    TEXT NOT NULL,
    quote_id       TEXT NOT NULL REFERENCES quotes(quote_id),
    status         TEXT NOT NULL,              -- current | superseded | not_material
    superseded_by  TEXT,
    reason         TEXT,
    llm_call_id    INTEGER REFERENCES llm_calls(id),
    PRIMARY KEY (report_date, quote_id)
);

CREATE TABLE IF NOT EXISTS reports (
    report_date   TEXT PRIMARY KEY,
    period_start  TEXT NOT NULL,
    period_end    TEXT NOT NULL,
    path          TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    run_id        TEXT,
    stats_json    TEXT
);
"""


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with session() as c:
        c.executescript(SCHEMA)


def rows(conn: sqlite3.Connection, sql: str, *params: Any) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def one(conn: sqlite3.Connection, sql: str, *params: Any) -> dict | None:
    r = conn.execute(sql, params).fetchone()
    return dict(r) if r else None


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
