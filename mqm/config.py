"""Paths, settings and YAML configuration loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("MQM_HOME", Path(__file__).resolve().parent.parent))
CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("MQM_DATA", ROOT / "data"))
DB_PATH = DATA_DIR / "quotes.db"
SNAPSHOT_DIR = DATA_DIR / "snapshots"      # gzipped raw HTML, named by sha256 (audit trail)
REPORT_DIR = DATA_DIR / "reports"          # daily Markdown reports
EXPORT_DIR = DATA_DIR / "export"           # JSONL + LLM context packs
LOG_DIR = DATA_DIR / "logs"                # JSON-lines event logs, one file per day
INBOX_DIR = ROOT / "inbox"                 # saved pages for detect-only sources

DEFAULT_START = "2026-10-01"
USER_AGENT = os.environ.get(
    "MQM_USER_AGENT",
    "MediaQuotesMonitor/1.0 (+https://github.com/khelifiMohamedYacine/media-quotes-monitor)",
)


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class LLMSettings:
    # One model for every reasoning step; each step sets its own effort level so that
    # cheap steps (span selection) spend few thinking tokens and hard ones (obsolescence) more.
    model: str = os.environ.get("MQM_MODEL", "claude-opus-5-5")
    effort_extract: str = os.environ.get("MQM_EFFORT_EXTRACT", "low")
    effort_obsolescence: str = os.environ.get("MQM_EFFORT_OBSOLESCENCE", "medium")
    effort_summary: str = os.environ.get("MQM_EFFORT_SUMMARY", "low")
    max_article_chars: int = int(os.environ.get("MQM_MAX_ARTICLE_CHARS", "60000"))


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    request_delay_s: float = float(os.environ.get("MQM_REQUEST_DELAY", "1.0"))
    http_timeout_s: float = float(os.environ.get("MQM_HTTP_TIMEOUT", "30"))
    http_retries: int = int(os.environ.get("MQM_HTTP_RETRIES", "3"))


SETTINGS = Settings()


def ensure_dirs() -> None:
    for d in (DATA_DIR, SNAPSHOT_DIR, REPORT_DIR, EXPORT_DIR, LOG_DIR, INBOX_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def domain_config() -> dict:
    return _yaml("domain.yaml")


def sources_config() -> list[dict]:
    """Not cached: sources can be added at runtime from the dashboard/CLI."""
    return _yaml("sources.yaml")["sources"]


def save_sources_config(sources: list[dict]) -> None:
    path = CONFIG_DIR / "sources.yaml"
    text = path.read_text(encoding="utf-8")
    header = text.split("sources:", 1)[0]
    body = yaml.safe_dump({"sources": sources}, sort_keys=False, allow_unicode=True, width=120)
    path.write_text(header + body, encoding="utf-8", newline="\n")


def append_source_config(entry: dict) -> None:
    """Append one source to config/sources.yaml without rewriting the rest of the file,
    so hand-written comments and formatting of the existing entries are preserved."""
    path = CONFIG_DIR / "sources.yaml"
    text = path.read_text(encoding="utf-8")
    block = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=120)
    block = "".join("  " + line if line.strip() else line for line in block.splitlines(keepends=True))
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(("" if text.endswith("\n") else "\n") + "\n" + block)
    if _yaml("sources.yaml")["sources"][-1] != entry:      # never leave a corrupted registry behind
        path.write_text(text, encoding="utf-8", newline="\n")
        raise ValueError("could not save the new source to config/sources.yaml")


def get_source(source_id: str) -> dict:
    for s in sources_config():
        if s["id"] == source_id:
            return s
    raise KeyError(f"unknown source: {source_id}")
