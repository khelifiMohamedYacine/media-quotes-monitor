"""Deterministic article discovery: news sitemaps, RSS/Atom feeds and paginated topic pages.

Each discoverer yields Candidate records. Nothing here uses an LLM: the same pages on the same
day give the same candidates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterator
from urllib.parse import urljoin, urlparse, urlunparse

from dateutil import parser as dtparse
from lxml import etree, html

from . import fetch
from .logs import log

NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}


@dataclass
class Candidate:
    url: str
    title: str = ""
    standfirst: str = ""
    published_at: str | None = None
    method: str = ""
    tagged: bool = False     # listed on a Ukraine-war topic page (editorial tag)


def to_utc_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        d = dtparse.parse(value)
    except (ValueError, OverflowError):
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).isoformat(timespec="seconds")


def normalise_url(url: str) -> str:
    p = urlparse(url.strip())
    return urlunparse((p.scheme or "https", p.netloc.lower(), p.path.rstrip("/") or "/", "", "", ""))


def _xml(text: str) -> etree._Element:
    return etree.fromstring(text.encode("utf-8"), parser=etree.XMLParser(recover=True, huge_tree=True))


MAX_CHILD_SITEMAPS = 10


def news_sitemap(url: str, max_children: int = MAX_CHILD_SITEMAPS) -> Iterator[Candidate]:
    """Google-News sitemap or sitemap index of them. Gives title + publication date per URL.
    Only the first `max_children` sitemaps of an index are read (news sitemaps cover ~48h; a huge
    index means it is not really a news sitemap and would take far too long to crawl)."""
    root = _xml(fetch.get(url).text)
    if root.tag.endswith("sitemapindex"):
        locs = [loc.text.strip() for loc in root.findall("sm:sitemap/sm:loc", NS) if loc.text]
        if len(locs) > max_children:
            log("discovery.sitemap_index_capped", "warning", url=url, children=len(locs), read=max_children)
        for loc in locs[:max_children]:
            try:
                yield from news_sitemap(loc, max_children)
            except fetch.FetchError as e:
                log("discovery.error", "warning", url=loc, error=str(e))
        return
    for u in root.findall("sm:url", NS):
        loc = u.findtext("sm:loc", default="", namespaces=NS).strip()
        yield Candidate(
            url=loc,
            title=(u.findtext("news:news/news:title", default="", namespaces=NS) or "").strip(),
            published_at=to_utc_iso(u.findtext("news:news/news:publication_date", namespaces=NS)
                                    or u.findtext("sm:lastmod", namespaces=NS)),
            method="news_sitemap",
        )


def rss(url: str) -> Iterator[Candidate]:
    root = _xml(fetch.get(url).text)
    items = root.findall(".//item")
    if items:
        for it in items:
            yield Candidate(
                url=(it.findtext("link") or "").strip(),
                title=(it.findtext("title") or "").strip(),
                standfirst=html.fromstring(it.findtext("description") or "<p></p>").text_content().strip()
                if it.findtext("description") else "",
                published_at=to_utc_iso(it.findtext("pubDate")),
                method="rss",
            )
        return
    atom = "{http://www.w3.org/2005/Atom}"
    for e in root.findall(f"{atom}entry"):
        link = e.find(f"{atom}link")
        yield Candidate(
            url=link.get("href", "") if link is not None else "",
            title=(e.findtext(f"{atom}title") or "").strip(),
            standfirst=(e.findtext(f"{atom}summary") or "").strip(),
            published_at=to_utc_iso(e.findtext(f"{atom}published") or e.findtext(f"{atom}updated")),
            method="rss",
        )


PAGE_DATE = re.compile(r'(?:lastPublished|datePublished|firstPublished)\\?"\s*:\s*\\?"(\d{4}-\d\d-\d\dT[\d:.]+Z?)'
                       r'|<time[^>]+datetime="(\d{4}-\d\d-\d\d[^"]*)"')


def page_median_date(text: str) -> str | None:
    """Median timestamp printed on a listing page (robust to one 'latest' sidebar item)."""
    dates = sorted(to_utc_iso(a or b) or "" for a, b in PAGE_DATE.findall(text))
    dates = [d for d in dates if d]
    return dates[len(dates) // 2] if dates else None


def topic_pages(url: str, max_pages: int, page_param: str = "page",
                url_filter=None, older_than: str | None = None, tagged: bool = True) -> Iterator[Candidate]:
    """Walk a paginated topic page; every article listed is in scope by editorial tag
    (unless tagged=False: a generic listing such as a homepage, where the relevance rules decide).
    Stops when a page's median date is older than `older_than` (the period start)."""
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        page_url = url if page == 1 else f"{url}{'&' if '?' in url else '?'}{page_param}={page}"
        try:
            text = fetch.get(page_url).text
            doc = html.fromstring(text)
        except fetch.FetchError as e:
            log("discovery.error", "warning", url=page_url, error=str(e))
            break
        median = page_median_date(text)
        found: list[Candidate] = []
        for a in doc.xpath("//a[@href]"):
            href = normalise_url(urljoin(page_url, a.get("href")))
            if href in seen or (url_filter and not url_filter(href)):
                continue
            seen.add(href)
            title = " ".join(a.text_content().split())
            found.append(Candidate(url=href, title=title[:300], method="topic_pages", tagged=tagged))
        log("discovery.topic_page", url=page_url, new_links=len(found), median_date=median)
        if not found:
            break
        yield from found
        if older_than and median and median < older_than:
            break


def discover(source: dict, day_from: date, day_to: date) -> list[Candidate]:
    """Run every configured discoverer for a source and merge results by URL."""
    include = [re.compile(p) for p in source.get("url_include", [])]
    exclude = [re.compile(p) for p in source.get("url_exclude", [])]

    def url_ok(u: str) -> bool:
        return (not include or any(p.search(u) for p in include)) and not any(p.search(u) for p in exclude)

    start = datetime.combine(day_from, datetime.min.time(), timezone.utc).isoformat()
    merged: dict[str, Candidate] = {}
    for d in source.get("discovery", []):
        m = d["method"]
        try:
            if m == "news_sitemap":
                it = news_sitemap(d["url"])
            elif m == "rss":
                it = rss(d["url"])
            elif m == "topic_pages":
                # walk until the listing is older than the period start (or max_pages)
                it = topic_pages(d["url"], d.get("max_pages", 60), d.get("page_param", "page"),
                                 url_filter=url_ok, older_than=start, tagged=d.get("tagged", True))
            elif m == "licensed_feed":
                # Plug-in point for a licensed publisher API (e.g. Reuters Connect / LSEG).
                log("discovery.licensed_feed_not_configured", "warning", source=source["id"])
                continue
            else:
                log("discovery.unknown_method", "warning", source=source["id"], method=m)
                continue
            n = 0
            for c in it:
                c.url = normalise_url(c.url)
                if not url_ok(c.url):
                    continue
                if c.published_at and c.published_at < start:
                    continue
                prev = merged.get(c.url)
                if prev:
                    prev.tagged = prev.tagged or c.tagged
                    prev.title = prev.title or c.title
                    prev.published_at = prev.published_at or c.published_at
                    prev.standfirst = prev.standfirst or c.standfirst
                else:
                    merged[c.url] = c
                    n += 1
            log("discovery.method", source=source["id"], method=m, candidates=n)
        except fetch.FetchError as e:
            log("discovery.error", "error", source=source["id"], method=m, error=str(e))
    return list(merged.values())


COMMON_FEED_PATHS = ("/news-sitemap.xml", "/sitemap-news.xml", "/sitemap_news.xml", "/feed/", "/rss",
                     "/rss.xml", "/feed.xml", "/atom.xml")


def _feed_kind(text: str) -> str | None:
    """'news_sitemap' / 'rss' if the document really is one, else None."""
    try:
        root = _xml(text)
    except etree.XMLSyntaxError:
        return None
    if root is None:
        return None
    tag = root.tag if isinstance(root.tag, str) else ""
    if tag.endswith("rss") or tag.endswith("}feed") or tag == "feed" or tag.endswith("RDF"):
        return "rss"
    if tag.endswith("urlset") and root.find(".//news:news", NS) is not None:
        return "news_sitemap"
    if tag.endswith("sitemapindex"):
        return "news_sitemap"
    return None


def autodetect_feeds(homepage: str, probe_common: bool = True) -> list[dict]:
    """For a new source: news sitemaps listed in robots.txt, <link rel=alternate> RSS/Atom feeds on
    the homepage and, if neither is advertised, a few conventional feed locations."""
    p = urlparse(homepage if "://" in homepage else f"https://{homepage}")
    base = f"https://{p.netloc or p.path}"
    found: list[dict] = []
    try:
        robots = fetch.get(f"{base}/robots.txt", check_robots=False).text
        for line in robots.splitlines():
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                # look at the path only: "news" in the domain (euronews.com, skynews.com) proves nothing
                if "news" in urlparse(sm).path.lower():
                    found.append({"method": "news_sitemap", "url": sm})
    except fetch.FetchError as e:
        log("add_source.robots_error", "warning", base=base, error=str(e))
    try:
        doc = html.fromstring(fetch.get(base).text)
        for link in doc.xpath("//link[@rel='alternate'][contains(@type,'rss') or contains(@type,'atom')]"):
            href = (link.get("href") or "").strip()
            if href:
                found.append({"method": "rss", "url": urljoin(base + "/", href)})
    except (fetch.FetchError, etree.ParserError) as e:
        log("add_source.home_error", "warning", base=base, error=str(e))
    if not found and probe_common:
        for path in COMMON_FEED_PATHS:
            try:
                r = fetch.get(base + path)
            except fetch.FetchError:
                continue
            kind = _feed_kind(r.text)
            if kind:
                found.append({"method": kind, "url": base + path})
                log("add_source.feed_probe_hit", base=base, path=path, kind=kind)
    seen: set[str] = set()
    return [d for d in found if not (d["url"] in seen or seen.add(d["url"]))]
