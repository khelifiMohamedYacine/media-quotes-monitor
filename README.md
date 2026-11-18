# Media Quotes Monitor: Russia's war on Ukraine

An automated system that detects, collects, verifies and presents media quotes that help
understand current trends and expectations about Russia's war on Ukraine. It checks every source
each day, keeps an auditable database of verbatim quotes with full metadata, and publishes a daily
Markdown briefing of the **non-obsolete** material quotes for the rolling past 14 days, grouped into
sections, each with a short LLM inference.

The core design principle: **deterministic code does everything that must be exact; the LLM is
only used for judgement, and never writes text that ends up presented as a quote.**

## Contents
- [How it works](#how-it-works)
- [Deterministic vs probabilistic: the split](#deterministic-vs-probabilistic-the-split)
- [Anti-hallucination: how quotes stay verbatim](#anti-hallucination-how-quotes-stay-verbatim)
- [Obsolete quotes](#obsolete-quotes)
- [Sources and access policy (Reuters)](#sources-and-access-policy-reuters)
- [Database and LLM-ready storage](#database-and-llm-ready-storage)
- [Dashboard](#dashboard)
- [Running it](#running-it)
- [Cost and token efficiency](#cost-and-token-efficiency)
- [Error handling and auditability](#error-handling-and-auditability)
- [Limitations and next steps](#limitations-and-next-steps)

## How it works

```mermaid
flowchart LR
  subgraph D[Deterministic]
    A[Discover<br>news sitemaps, RSS,<br>topic pages] --> B[Relevance rules<br>regex on headline/slug<br>or editorial topic tag]
    B --> C[Fetch politely<br>robots.txt, retries,<br>raw HTML snapshot]
    C --> E[Extract<br>metadata + numbered<br>paragraphs/sentences]
  end
  E --> F[[LLM: pick material<br>quotes as sentence-id<br>ranges + section]]
  F --> G[Rebuild text from ids<br>+ verify vs article<br>and vs snapshot]
  G --> H[(SQLite<br>quotes, articles,<br>runs, llm_calls)]
  H --> I[[LLM: which quotes are<br>superseded / duplicate,<br>materiality 1-5]]
  I --> J[Validate judgements,<br>rank, copy verbatim<br>into Markdown]
  J --> K[[LLM: 1-2 sentence<br>inference per section]]
  K --> L[data/reports/YYYY-MM-DD.md]
```

| Step | Module | Kind |
|---|---|---|
| Discover candidate articles (news sitemap, RSS, paginated topic pages) | `mqm/discovery.py` | deterministic |
| Decide relevance (regex rules in `config/domain.yaml`, or source's Ukraine-war topic tag) | `mqm/relevance.py` | deterministic |
| Fetch with robots.txt check, rate limit, retries, gzipped raw snapshot named by SHA-256 | `mqm/fetch.py` | deterministic |
| Extract title, dates, canonical URL (JSON-LD/meta) and body blocks with stable sentence ids | `mqm/extract.py` | deterministic |
| Read the article, choose material quotes **as id ranges**, label section, list key facts | `mqm/quotes.py` | **LLM** |
| Rebuild verbatim text from ids; run 9 verification checks | `mqm/quotes.py`, `mqm/verify.py` | deterministic |
| Judge obsolescence (superseded by newer facts), duplicates, materiality | `mqm/report.py` | **LLM** |
| Validate those judgements, rank, cap per section, write Markdown verbatim | `mqm/report.py` | deterministic |
| Write the 1-2 sentence inference under each section heading | `mqm/report.py` | **LLM** |

## Deterministic vs probabilistic: the split

**Deterministic (same input, same output, fully auditable):**
- *Which articles are relevant* (the brief requires this to be deterministic). Each article row
  stores the exact rule that matched (`core:\bkyiv\b`, `russia+context:\bputin&\bsanction`, or
  `topic_tag`). Rules live in `config/domain.yaml`.
- All metadata (title, publication date, URL, source) is taken from the page's JSON-LD / meta tags,
  never from the model.
- Quote text, which is rebuilt from sentence ids (below).
- Verification, ranking, the 14-day window and Markdown rendering.

**Probabilistic (LLM, with structured JSON output and validation):**
- Which passages of a relevant article are *material* quotes, and which report section each belongs to.
- Whether an older quote's facts are *superseded* by a newer quote.
- The short inference under each section heading. It is labelled as machine-generated in the report.

Every LLM call (open-weight model via Groq by default, Hugging Face or Claude) is recorded in `llm_calls` (model, effort, tokens, cost, full JSON response), and each
quote links to the call that selected it.

## Anti-hallucination: how quotes stay verbatim

1. The article body is split into blocks `P1..Pn` and sentences `P4.S1, P4.S2, ...`.
2. The LLM sees the numbered text and may only answer with **id ranges** (`start: P4.S1, end: P5.S2`),
   at most 4 consecutive sentences, no headings. It never returns quote text.
3. Code rebuilds the quote from those ids, preserving the original paragraph breaks and bold/italic.
   An invented id, a reversed range, an over-long span or a heading is rejected and logged.
   So a quote cannot be a "frankenquote": it is always one contiguous run of the original text.
4. `verify.py` then checks: verbatim in the extracted article; verbatim in an *independent
   re-extraction from the stored raw HTML*; length bounds; no boilerplate (ads, captions, "Read
   more", URLs, table debris); ends as a complete sentence; all metadata present; valid dates
   (published ≤ extracted); URL is https and on the source's domain; inside the requested period.
5. The result of every check is stored on the quote (`verification_json`). Only `verified=1`
   quotes reach reports and exports. `python -m mqm audit` re-runs all checks against the snapshots.
6. The report writer copies `quote_text` byte-for-byte and asserts it appears unchanged in the file.

## Obsolete quotes

For each daily report the LLM gets every verified quote from the 14-day window (oldest first) with
its date, section, `topic_key` and paraphrased `key_facts`, and labels each one `current`,
`superseded` (with the id of the **newer** quote that replaces its facts) or `duplicate`, plus a
materiality score of 1-5. Code then enforces guard-rails:
- a quote can only be superseded by a quote that exists, is a different quote and is not older;
- quotes the model skipped stay visible (fail-open for recall);
- supersession chains and cycles are resolved so that something current always remains.

Withheld quotes are not shown in the Markdown. Their status, the superseding quote and the reason
are kept in `assessments` and `data/reports/<date>.audit.json`, and appear on each quote's dashboard page.

## Sources and access policy (Reuters)

| Source | Access | Discovery |
|---|---|---|
| BBC News (mandatory) | open | Google-News sitemap (titles + dates), "War in Ukraine" topic pages (paginated, stops at period start), Europe RSS |
| Reuters (mandatory) | **detect-only** | inbox / licensed feed |
| The Guardian | open | news sitemap + Ukraine RSS |
| Al Jazeera | open | news sitemap + RSS |

**Reuters.** `reuters.com/robots.txt` states: *"Collection of content, data and/or information from
reuters.com through automated means is prohibited unless you have prior written consent from
Reuters"*, and the site returns HTTP 401 to automated clients. Google News search RSS (a possible
headline source) is disallowed by Google's robots.txt. The system therefore **does not crawl
Reuters and does not attempt to get around these controls**. Reuters content enters through:
- `inbox/`: a licensed reader saves article pages and runs `python -m mqm import-inbox`.
  The same extraction, LLM selection and verification apply, and the raw HTML is snapshotted.
- a `licensed_feed` discovery method in the registry, the plug-in point for Reuters Connect / LSEG
  APIs once the business has a licence.

**Adding a source** (`python -m mqm add-source "Atlantic Council" https://www.atlanticcouncil.org`,
or the dashboard form) builds a pipeline automatically. It reads the site's robots.txt for news
sitemaps, finds RSS/Atom `<link>` feeds on the homepage, optionally takes a topic-page URL, sets the
access mode from robots.txt, writes the entry to `config/sources.yaml`, and backfills from
1 October 2026.

## Database and LLM-ready storage

SQLite (`data/quotes.db`) is a single auditable file that any SQL tool can open:

| Table | Holds |
|---|---|
| `quotes` | Quote ID, quote text (formatted) and plain text, source ID and name, publication date, article name, link, extraction date, speaker, type, section, sentence span, key facts, topic key, SHA-256, verification results, LLM call, run |
| `articles` | every detected article: discovery method, matched relevance rule, status, snapshot hash, extracted body |
| `runs` | each daily / backfill / import / add-source run with parameters and statistics |
| `llm_calls` | every model call: purpose, model, effort, tokens, cost, stop reason, full response |
| `assessments` | per report: current / superseded / duplicate / not_material, superseding quote, reason |
| `reports` | each Markdown file with its window and SHA-256 |

Exports for LLM contextualisation (`python -m mqm export`, also from the dashboard):
- `data/export/quotes.jsonl`: one quote per line with all metadata, ready for any RAG or vector store;
- `data/export/context_pack.md`: a compact, token-efficient Markdown pack grouped by section, with
  `[Q-id]` tags so a model can cite exactly which quote supports a claim.

## Dashboard

`python -m mqm serve`, then open http://127.0.0.1:8000

- **Overview:** verified/rejected counts, LLM spend, articles by source and status, recent runs, sources.
- **Run controls:** daily check + report; rebuild report; **collect any past period** (for example
  2026-08-01 to 2026-09-30, per source or all); **add a new source**; import the inbox.
- **Reports:** every daily Markdown file, rendered or as a download, with its SHA-256.
- **Quotes:** search and filter. Each quote page shows all verification checks, its provenance
  (snapshot, LLM call, run) and its history across reports.
- **Articles, Logs, LLM calls:** the full audit trail. You can download the SQLite database.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env                                  # add GROQ_API_KEY (or HF_TOKEN / ANTHROPIC_API_KEY)
python -m mqm init
python -m mqm backfill --from 2026-10-01              # initial collection from 1 October 2026
python -m mqm daily                                   # daily: detect + extract + report + export
python -m mqm report --date 2026-11-18                # rebuild a given day's report
python -m mqm backfill --from 2026-08-01 --to 2026-09-30   # optional past period on request
python -m mqm serve                                   # dashboard
pytest -q                                             # offline tests with a scripted LLM
```

**Daily schedule.** The daily run looks back 2 days, so a missed day heals itself.
- Windows: `schtasks /Create /SC DAILY /ST 07:00 /TN mqm-daily /TR "C:\path\.venv\Scripts\python.exe -m mqm daily"`
- Linux/macOS: `0 7 * * * cd /path && .venv/bin/python -m mqm daily`

## Cost and token efficiency

- Deterministic relevance rules filter before any LLM call. Off-topic articles (sport, culture) are
  dropped for free, and articles older than the period are dropped after fetching, with no LLM call.
- The model returns **ids, not text**, so output tokens per article are small.
- **Open-weight model by default, on a free tier:** an open-weight model served by Groq
  (`MQM_LLM_PROVIDER=groq`, model set by `MQM_GROQ_MODEL`). Hugging Face Inference Providers
  (`huggingface`, e.g. Qwen3-235B) and Claude (`anthropic`) are one-line switches.
  All of them sit behind the `LLM` protocol in `mqm/llm.py`, so models can be benchmarked against the same
  deterministic verification checks: the checks, not the model, decide what is stored.
- Structured output: JSON-schema mode where the provider supports it. The schema is always
  re-validated in code, and invalid output triggers a self-repair retry that shows the model the
  exact validation errors.
- Claude path: the stable system prompt (instructions + section definitions) is prompt-cached, and
  effort is set per step (`low` for selection and summaries, `medium` for obsolescence).
- Spend is visible per call and in total on the dashboard.

## Error handling and auditability

- Network errors, 429 and 5xx are retried with exponential backoff (honouring `Retry-After`).
  4xx errors and robots.txt refusals are permanent and logged. Failed articles are retried on later
  runs (up to 3 attempts).
- One bad page never stops a run. Each run ends `ok`, `partial` or `failed` with statistics.
- Free-tier rate limits (HTTP 429) are waited out using `Retry-After`, without using up retries.
- LLM errors are typed (authentication, exhausted credits, rate limit, refusal, truncation,
  invalid JSON) and recorded. Each failed attempt is logged in `llm_calls`.
- Large articles are handled: there is no size limit on fetching, and the text sent to the model is
  capped at 60k characters per article (configurable).
- Extraction uses a DOM extractor with a site-specific body XPath when one is configured, and falls
  back to trafilatura when a layout is unknown. Boilerplate is removed by explicit rules.
- `data/logs/YYYY-MM-DD.jsonl` holds an append-only, structured event log tagged with the run ID.

## Limitations and next steps

- Reuters needs a licence (Reuters Connect / LSEG) for fully automated coverage; the plug-in point exists.
- Paywalled sources (FT, Bloomberg, The Economist) need subscriber API access in the same way.
- Next: an evaluation set of hand-labelled articles to measure quote recall and precision, plus a
  benchmark of cheaper and open-source models against it; a Neo4j or Obsidian export of
  quote–topic–speaker links; batch processing for large backfills.
