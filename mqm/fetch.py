"""Polite, fault-tolerant HTTP fetching.

- robots.txt is checked for every URL (urllib.robotparser) and disallowed URLs are never fetched.
- Per-domain rate limit, retries with exponential backoff on network errors / 429 / 5xx.
- Every fetched article page is saved gzipped under its SHA-256 so extractions can be re-verified.
"""
from __future__ import annotations

import gzip
import hashlib
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from .config import SETTINGS, SNAPSHOT_DIR, USER_AGENT
from .logs import log


class FetchError(Exception):
    def __init__(self, url: str, reason: str, status: int | None = None, retryable: bool = True):
        super().__init__(f"{reason} ({url})")
        self.url, self.reason, self.status, self.retryable = url, reason, status, retryable


@dataclass
class Fetched:
    url: str            # final URL after redirects
    status: int
    text: str
    content_type: str
    sha256: str


_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
_last_hit: dict[str, float] = {}
_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-GB,en;q=0.8"},
            timeout=SETTINGS.http_timeout_s,
            follow_redirects=True,
        )
    return _client


def _host(url: str) -> str:
    return urlparse(url).netloc.lower()


def robots_allows(url: str) -> bool:
    host = _host(url)
    if host not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = client().get(f"https://{host}/robots.txt")
            if r.status_code >= 400:
                rp = None  # no robots.txt -> allowed
            else:
                rp.parse(r.text.splitlines())
        except httpx.HTTPError as e:
            log("robots.unavailable", "warning", host=host, error=str(e))
            rp = None
        _robots[host] = rp
    rp = _robots[host]
    return True if rp is None else rp.can_fetch(USER_AGENT, url)


def _throttle(host: str) -> None:
    wait = SETTINGS.request_delay_s - (time.monotonic() - _last_hit.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_hit[host] = time.monotonic()


def get(url: str, *, snapshot: bool = False, check_robots: bool = True) -> Fetched:
    if check_robots and not robots_allows(url):
        raise FetchError(url, "disallowed by robots.txt", retryable=False)
    host = _host(url)
    last: Exception | None = None
    for attempt in range(1, SETTINGS.http_retries + 1):
        _throttle(host)
        try:
            r = client().get(url)
        except httpx.HTTPError as e:
            last = FetchError(url, f"network error: {type(e).__name__}: {e}")
            log("http.retry", "warning", url=url, attempt=attempt, error=str(e))
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = FetchError(url, f"HTTP {r.status_code}", r.status_code)
            retry_after = r.headers.get("retry-after", "")
            delay = int(retry_after) if retry_after.isdigit() else 2 ** attempt
            log("http.retry", "warning", url=url, attempt=attempt, status=r.status_code)
            time.sleep(min(delay, 60))
            continue
        if r.status_code >= 400:
            raise FetchError(url, f"HTTP {r.status_code}", r.status_code,
                             retryable=r.status_code in (408, 425))
        body = r.content
        sha = hashlib.sha256(body).hexdigest()
        if snapshot:
            save_snapshot(sha, body)
        log("http.get", url=url, final_url=str(r.url), status=r.status_code, bytes=len(body))
        return Fetched(str(r.url), r.status_code, r.text, r.headers.get("content-type", ""), sha)
    assert last is not None
    raise last


def save_snapshot(sha: str, body: bytes) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    p = SNAPSHOT_DIR / f"{sha}.html.gz"
    if not p.exists():
        p.write_bytes(gzip.compress(body))


def load_snapshot(sha: str) -> str:
    raw = gzip.decompress((SNAPSHOT_DIR / f"{sha}.html.gz").read_bytes())
    return raw.decode("utf-8", errors="replace")
