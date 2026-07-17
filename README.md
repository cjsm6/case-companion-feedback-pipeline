# Case Companion Feedback Pipeline

Offline, deterministic feedback analysis for Case Companion (AI tool for personal injury
law firms). No LLM or API calls at runtime — tagging is rules-based, clustering is
TF-IDF + cosine similarity.

Process write-up (prompting, iteration, and evaluation of AI-generated output) is included
with the submission rather than in this repo.

## Setup

```
pip install -r requirements.txt
```

## Running the pipeline

```
python pipeline.py
```

- First run seeds `data/processed/master.csv` from `data-context/feedback_raw.csv`.
- Every run after that absorbs any CSVs sitting in `data/incoming/` (same column
  schema as the seed file). Drop new files there — the pipeline needs zero manual
  intervention to pick them up.
- Reruns are idempotent: each row is hashed on `(user_id, feedback_date,
  feedback_text)` and checked against `data/processed/seen_hashes.json` before
  ingestion, so running the pipeline twice never duplicates a row.
- Prints a run summary: rows ingested/skipped, full theme/severity/sentiment
  distributions, unclassified/unthemed counts (with the actual unthemed row texts),
  and which issue clusters are NEW, GREW, or MERGED this run.

### Re-tagging after a taxonomy change

```
python pipeline.py --retag
```

Re-tags every existing row in `master.csv` with the current keyword rules before
ingesting anything new. `row_hash` and the dedupe ledger are untouched — only
`theme`/`severity`/`sentiment` are recomputed. Use this whenever the keyword lists in
`pipeline.py` change, so `master.csv` never holds a mix of old- and new-rules labels.

## Running the dashboard

```
streamlit run app.py
```

Reads `data/processed/master.csv` only — it never re-runs the TF-IDF clustering, just
groups by the `cluster_id` column `pipeline.py` already computed. Six sections:

- **Overview** — KPI tiles summarizing the corpus at a glance.
- **This Week's Priorities** — criticals only, ranked customer-size-first then
  severity.
- **Who's Unhappy** — sentiment breakdown by role and firm size.
- **Open Issues by Theme** — severity-stacked chart of issues by theme.
- **Issue Clusters** — the full deduplicated table, one row per issue, with a
  per-cluster inspector to see every member report.
- **Raw Feedback Explorer** — the full labeled table, filterable by channel, role,
  firm size, sentiment, severity, theme, date range, and free-text search.

The app picks up a fresh `pipeline.py` run automatically (cache is keyed on the
file's mtime) — no restart needed.

## Structure

```
data-context/feedback_raw.csv   seed corpus (100 rows), read once on first run
data/incoming/                  drop zone for new daily CSVs
data/processed/master.csv       labeled corpus: every row + theme/severity/sentiment/cluster_id
data/processed/seen_hashes.json dedupe ledger
pipeline.py                     ingest, tag, cluster
app.py                          Streamlit viewer
```

## How tagging works

Rules-based, in `pipeline.py`:

- **Theme** — first keyword match wins, checked in a deliberate priority order (see
  `THEME_PRIORITY`). Falls back to `"unthemed"` (not to a real category) when nothing
  matches, so taxonomy coverage gaps stay visible instead of silently inflating
  whichever bucket is checked last.
- **Sentiment** is decided first and gates severity — a row with praise language and
  no defect language never enters the critical/high/medium ladder, because risk
  vocabulary ("sanctions", "discoverable", "statute of limitations") reads identically
  whether it's describing a bug or the feature that *prevents* that exact bug.
- **Severity** ladder: critical (malpractice/ethics exposure) → high (wrong numbers
  driving money decisions, crashes, data loss) → medium (friction, integration gaps) →
  low (explicit soft asks) → unknown (lexicon found no signal at all, honestly
  reported rather than guessed).
- All lexicon matching uses `has_term()` — front-anchored, suffix-open word matching
  (`\bterm\w*`), not plain substring. Plain substring equates "inaccurate" with
  "accurate"; full word-boundary matching breaks "crash" → "crashes". See the
  docstring on `has_term` for the reasoning and the one documented residual case
  (a row containing both "inaccurate" and standalone "accurate" in the same sentence)
  that bag-of-words genuinely can't resolve.

## Clustering

`TfidfVectorizer(stop_words='english', ngram_range=(1,2), sublinear_tf=True)` +
cosine similarity, threshold 0.35, connected components (union-find) so 3+ reports of
the same issue collapse into one cluster. `cluster_id` is the `row_hash` of the
chronologically-earliest member — not a sequential number — so it stays stable across
reruns even as clusters grow or merge.

Two duration fields per cluster: `report_span_days` (last report − first report) and
`days_open` (as-of date − first report), where the as-of date is the max
`feedback_date` across the whole corpus, not the real wall clock — the seed data is
from mid-2024, so "days since today" would be meaningless for triage.

## Known limitations

- Rules-based tagging tops out around what keyword matching can resolve semantically
  (documented in `has_term`'s docstring) — a handful of rows land as `mixed` when a
  human would call them clearly negative, because the same word ("outdated",
  "accurate") can describe either the product's own quality or the input content the
  product correctly analyzes.
- `medium` severity is a wide catch-all (~40% of the seed corpus) because most
  negative feedback in this product also contains complaint language before any
  soft-request phrasing is ever checked. Critical/high still surface correctly, and
  Widget 1 sorts by `days_open` within a tier.
- "unthemed" rows are a real signal, not a bug: `pipeline.py`'s run summary prints
  their full text every run so the taxonomy can be extended deliberately.
