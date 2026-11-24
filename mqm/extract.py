"""Deterministic article extraction: metadata + clean body blocks with stable sentence ids.

The body is turned into numbered blocks (P1, P2, ...) and sentences (P3.S2, ...). The LLM later
selects quotes by *id range only*; the quote text is then rebuilt from these blocks, so the
model cannot alter, merge or invent wording ("frankenquotes").
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import trafilatura
from lxml import html

from .discovery import normalise_url, to_utc_iso

# Boilerplate that must never become part of a quote.
NOISE = re.compile(
    r"^(advertisement|read more|watch:|listen:|sign up|follow (us|bbc)|related (topics|stories|content)|"
    r"image (source|caption)|getty images|copyright|share this|save\b|more on this story|"
    r"get in touch|you can (now )?(sign up|follow)|download the|subscribe|this video can not be played|"
    r"media caption|skip (past|to)|explore more|story continues)",
    re.I,
)
EXCLUDED_ANCESTORS = (
    "ancestor::aside or ancestor::nav or ancestor::figure or ancestor::figcaption or ancestor::footer "
    "or ancestor::header or ancestor::*[contains(@data-testid,'promo')] or "
    "ancestor::*[contains(@class,'related') or contains(@class,'newsletter') or contains(@class,'advert') "
    "or contains(@class,'ad-slot') or contains(@class,'caption') or contains(@class,'promo') "
    "or contains(@class,'more-on-this')]"
)
BLOCK_TAGS = ("p", "h2", "h3", "li", "blockquote")
ABBREV = {
    "mr", "mrs", "ms", "dr", "gen", "lt", "col", "maj", "sgt", "capt", "st", "prof", "sen", "rep", "gov",
    "pres", "u.s", "u.k", "u.n", "e.g", "i.e", "etc", "no", "vs", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec", "inc", "ltd", "co", "corp", "jr", "sr", "adm",
}
BOUNDARY = re.compile(r"(?<=[.!?])([\"'”’)\]]*)\s+(?=[\"'“‘(\[]?[A-Z0-9])")


@dataclass
class Sentence:
    id: str
    md: str


@dataclass
class Block:
    id: str
    kind: str
    md: str
    sentences: list[Sentence] = field(default_factory=list)


@dataclass
class Article:
    url: str
    title: str
    published_at: str | None
    modified_at: str | None
    standfirst: str
    blocks: list[Block]
    extractor: str

    def to_json(self) -> str:
        return json.dumps([asdict(b) for b in self.blocks], ensure_ascii=False)


def plain(md: str) -> str:
    """Strip our Markdown emphasis markers and normalise whitespace."""
    return " ".join(re.sub(r"(\*\*|\*|__|_(?=\w)|(?<=\w)_)", "", md).split())


def norm_for_match(text: str) -> str:
    t = plain(text)
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    t = t.replace("–", "-").replace("—", "-").replace(" ", " ")
    return " ".join(t.split()).lower()


def split_sentences(text: str) -> list[str]:
    out, start = [], 0
    for m in BOUNDARY.finditer(text):
        end = m.start() + len(m.group(1))          # keep closing quotes/brackets with the sentence
        before = text[start:m.start()].rstrip(".!?")
        last_tok = before.split()[-1].lower().strip("(\"'“") if before.split() else ""
        if last_tok in ABBREV or re.fullmatch(r"[a-z]", last_tok):
            continue                                # "Mr. Putin", "U.S. aid", initials
        out.append(text[start:end].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def _inline_md(el) -> str:
    """Render an element's inline content as Markdown, keeping bold/italic, dropping links' URLs."""
    parts: list[str] = [el.text or ""]
    for child in el:
        tag = child.tag if isinstance(child.tag, str) else ""
        inner = _inline_md(child)
        if tag in ("strong", "b") and inner.strip():
            parts.append(f"**{inner.strip()}**" + (" " if inner.endswith(" ") else ""))
        elif tag in ("em", "i") and inner.strip():
            parts.append(f"*{inner.strip()}*" + (" " if inner.endswith(" ") else ""))
        elif tag == "br":
            parts.append(" ")
        elif tag in ("script", "style", "button", "svg", "figure", "img"):
            pass
        else:
            parts.append(inner)
        parts.append(child.tail or "")
    return " ".join("".join(parts).split())


def _jsonld(doc) -> dict:
    best: dict = {}
    for s in doc.xpath("//script[@type='application/ld+json']/text()"):
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            d = stack.pop()
            if isinstance(d, dict):
                if "@graph" in d:
                    stack.extend(d["@graph"])
                t = d.get("@type", "")
                t = " ".join(t) if isinstance(t, list) else str(t)
                if "Article" in t or "Posting" in t:
                    best = best or d
    return best


def _meta(doc, *names: str) -> str:
    for n in names:
        v = doc.xpath(f"//meta[@property='{n}' or @name='{n}' or @itemprop='{n}']/@content")
        if v and v[0].strip():
            return v[0].strip()
    return ""


def _number(raw_blocks: list[tuple[str, str]]) -> list[Block]:
    blocks: list[Block] = []
    seen: set[str] = set()
    for kind, md in raw_blocks:
        md = md.strip()
        p = plain(md)
        if not p or NOISE.match(p) or p in seen:
            continue
        if kind == "p" and len(p) < 25 and not re.search(r"[.!?\"”]$", p):
            continue  # captions / labels
        if re.fullmatch(r"[\d\s|/.\-–]+", p) or p.count("|") >= 3:
            continue  # page numbers, table debris
        seen.add(p)
        bid = f"P{len(blocks) + 1}"
        sents = [md] if kind in ("h2", "h3") else split_sentences(md)
        blocks.append(Block(bid, kind, md, [Sentence(f"{bid}.S{i + 1}", s) for i, s in enumerate(sents)]))
    return blocks


def _paragraph_chars(el) -> int:
    return sum(len(" ".join(p.text_content().split())) for p in el.iter("p") if not p.xpath(EXCLUDED_ANCESTORS))


def main_container(doc) -> list:
    """The element holding the story itself when no site-specific body XPath is configured.

    News pages often mark up every item of a "latest news" feed or related-stories rail as its own
    <article>. Reading all of them would mix other stories into this one, so when there are several
    candidates only the one carrying the most paragraph text is used (the story, not the feed)."""
    for xp in ("//article", "//main"):
        candidates = doc.xpath(xp)
        if candidates:
            best = max(candidates, key=_paragraph_chars)
            return [best] if _paragraph_chars(best) else []
    return []


def extract(page_html: str, url: str, body_xpath: str | None = None) -> Article:
    doc = html.fromstring(page_html)
    ld = _jsonld(doc)
    title = (ld.get("headline") or _meta(doc, "og:title", "twitter:title")
             or (doc.findtext(".//title") or "")).strip()
    published = to_utc_iso(ld.get("datePublished") or _meta(doc, "article:published_time", "datePublished",
                                                             "dc.date", "pubdate"))
    if not published:
        t = doc.xpath("//time/@datetime")
        published = to_utc_iso(t[0]) if t else None
    modified = to_utc_iso(ld.get("dateModified") or _meta(doc, "article:modified_time"))
    canonical = doc.xpath("//link[@rel='canonical']/@href") or [_meta(doc, "og:url") or url]
    standfirst = (ld.get("description") or _meta(doc, "description", "og:description")).strip()

    raw: list[tuple[str, str]] = []
    containers = doc.xpath(body_xpath) if body_xpath else []
    if not containers:
        containers = main_container(doc)
    for c in containers:
        nested = set(c.iterdescendants("article"))      # embedded story cards inside the story
        for el in c.iter(*BLOCK_TAGS):
            if el.xpath(EXCLUDED_ANCESTORS):
                continue
            if nested and any(a in nested for a in el.iterancestors("article")):
                continue
            if el.tag == "p" and el.xpath("ancestor::blockquote|ancestor::li"):
                continue
            raw.append((el.tag, _inline_md(el)))
    blocks = _number(raw)
    extractor = "dom"
    if sum(1 for b in blocks if b.kind == "p") < 3:
        md = trafilatura.extract(page_html, output_format="markdown", include_comments=False,
                                 include_tables=False, include_images=False, favor_precision=True) or ""
        raw = []
        for para in re.split(r"\n\s*\n", md):
            para = para.strip()
            if para.startswith("#"):
                raw.append(("h2", para.lstrip("# ")))
            elif para.startswith(("- ", "* ")):
                raw.extend(("li", ln[2:]) for ln in para.splitlines() if ln[:2] in ("- ", "* "))
            elif para.startswith(">"):
                raw.append(("blockquote", para.lstrip("> ")))
            else:
                raw.append(("p", " ".join(para.split())))
        blocks = _number(raw)
        extractor = "trafilatura"
    return Article(normalise_url(canonical[0]), " ".join(title.split()), published, modified,
                   " ".join(standfirst.split()), blocks, extractor)


def blocks_from_json(body_json: str) -> list[Block]:
    return [Block(b["id"], b["kind"], b["md"], [Sentence(**s) for s in b["sentences"]])
            for b in json.loads(body_json)]


def numbered_text(blocks: list[Block], hidden: set[str] | None = None) -> str:
    """The view given to the LLM: every sentence prefixed with its id. Sentence ids in `hidden`
    (site template text such as feeds and promos) are left out, so they cannot be selected."""
    hidden = hidden or set()
    lines = []
    for b in blocks:
        if b.kind in ("h2", "h3"):
            if b.sentences[0].id not in hidden:
                lines.append(f"[{b.sentences[0].id}] ## {plain(b.md)}")
        else:
            line = " ".join(f"[{s.id}] {plain(s.md)}" for s in b.sentences if s.id not in hidden)
            if line:
                lines.append(line)
    return "\n".join(lines)
