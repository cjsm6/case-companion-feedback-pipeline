"""Ingest, dedupe, tag, and cluster Case Companion customer feedback.

Run with: python pipeline.py

On first run this seeds the master store from data-context/feedback_raw.csv.
On every run it then absorbs whatever CSVs sit in data/incoming/, so a cron
job that drops ~10 rows/day needs zero manual intervention -- just re-run
this script (or schedule it) and it picks up only what it hasn't seen.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED_PATH = Path("data-context/feedback_raw.csv")
INCOMING_DIR = Path("data/incoming")
MASTER_PATH = Path("data/processed/master.csv")
HASHES_PATH = Path("data/processed/seen_hashes.json")

SIM_THRESHOLD = 0.35  # validated on the seed corpus: 18/18 recall, 0 false positives

RAW_COLUMNS = [
    "user_id", "channel", "feedback_date", "feedback_text", "user_role",
    "firm_size", "months_subscribed", "monthly_queries",
    "avg_session_duration_min", "case_types_handled",
]
MASTER_COLUMNS = RAW_COLUMNS + [
    "row_hash", "theme", "severity", "sentiment", "cluster_id", "is_representative",
]

# Severity ranking used to pick a cluster's worst-case severity and to sort
# Widget 1 -- higher number = more urgent. "unknown" (lexicon found nothing
# at all: no praise, no defect, no soft-request) ranks below "low" (an
# explicit soft-request signal was found) since it's a weaker claim than
# even a nice-to-have -- and above "praise" since an unreadable row is not
# the same claim as a confirmed positive one.
SEVERITY_RANK = {"critical": 5, "high": 4, "medium": 3, "low": 2, "unknown": 1, "praise": 0}
SEVERITY_ORDER = ["critical", "high", "medium", "low", "unknown", "praise"]


def has_term(text, terms):
    """Front-anchored, suffix-open word match: \\bterm\\w* -- a word boundary
    before the term, nothing constrained after it.

    Plain substring matching ('t in text') silently equates 'inaccurate' with
    'accurate' and 'incorrect' with 'correct', which flips real complaints to
    'mixed' sentiment (measured: ~14 rows in this corpus). Full \\bterm\\b
    over-corrects the other way and breaks legitimate inflections this
    lexicon depends on -- 'crash' stops matching 'crashes', 'duplicat' stops
    matching 'duplicates', 'struggl' stops matching 'struggling'. Anchoring
    only the front gets both: prefixes ('in'+accurate) are excluded because
    there's no word boundary before 'accurate' inside 'inaccurate', while
    suffixes (crash+es) are included because \\w* absorbs them.

    Known residual gap (not fixed -- bag-of-words can't fix it): row 96
    contains both 'inaccurate' and a standalone 'accurate' in one sentence
    ("...when accurate market rate is $800-1200/hour"), so it reads as mixed
    when it's really negative. Same root cause makes two citation-checking
    praise rows ("catches outdated case law") register as mixed, since
    'outdated' can't distinguish stale input content the AI correctly flags
    from the product itself being stale. This is the honest argument for an
    LLM-based classifier down the line, not a pattern to chase further here.
    """
    return any(re.search(r"\b" + re.escape(t) + r"\w*", text) for t in terms)


# Theme keyword lists, checked in this order (first match wins). Order matters:
# narrower/more diagnostic themes are checked before broad ones so an overlap
# (e.g. a calculator crash mentions both "benchmarking" and "crash") resolves
# to the theme that best explains *why* it's a problem, not just what feature
# it touched. document_ai_quality is last -- it's also the catch-all fallback
# for future feedback that doesn't match anything else, since "AI got the
# document wrong" is the closest bucket for novel complaints in this product.
THEME_PRIORITY = [
    ("jurisdiction_coverage", [
        "federal court", "state court", "maritime", "jones act", "local rules",
        "court holiday", "multi-state", "different states", "statute of limitations",
        "jurisdiction requirements", "federal procedures", "local practice",
    ]),
    ("scoring_merit", [
        "merit scoring", "case merit", "merit assessment", "outcome prediction",
        "predict outcome", "case scoring",
    ]),
    ("trial_prep", [
        "exhibit", "jury", "voir dire", "expert witness prep", "expert witness matching",
        "expert witness preparation", "deposition summary", "trial prep", "trial exhibit",
        "trial timeline", "jury instruction",
    ]),
    ("integrations_sync", [
        "outlook", "billing software", "accounting software", "needles", "lexisnexis",
        "court filing integration", "deposition scheduling", "calendar sync",
        "time tracking integration",
    ]),
    ("scale_performance", [
        "crash", "slower", "server issue", "upload limit", "50mb", "10,000+",
        "unwieldy", "document volume",
    ]),
    ("data_freshness", [
        "outdated contact", "duplicate contact", "duplicate entries", "duplicate records",
        "archival", "stale", "verification date", "case status dashboard",
        "progress percentage", "expert witness database", "contact information",
    ]),
    ("client_experience", [
        # "client communication" (bare) would match inside "attorney-client
        # communications" -- a hyphen is a real word boundary, so front-
        # anchoring alone can't tell "client-facing comms feature" apart from
        # "attorney-client privileged communications". Narrowed to the actual
        # phrases this product's client_experience rows use instead.
        "client portal", "intake chatbot", "client communication ai",
        "client communication template", "client satisfaction",
        "communication scheduler", "client intake",
    ]),
    ("training_enablement", [
        "training material", "tutorial", "documentation", "training request",
    ]),
    ("calculation_accuracy", [
        "damages calculator", "damage calculation", "settlement demand",
        "settlement authority", "settlement calculation", "settlement tracking",
        "case value benchmark", "benchmarking tool", "expert witness cost calculator",
        "expert cost calculator", "medical expense projection", "comparative negligence",
        "billing breakdown analysis", "roi calculation", "lien resolution",
    ]),
    ("strategic_intelligence", [
        # Rows the original taxonomy didn't name: opposing-counsel research,
        # insurance coverage analysis, settlement-conference prep, case
        # similarity search -- a coherent, heavily-praised product area, not
        # scattered noise. Placed just before document_ai_quality so it
        # doesn't shadow any earlier, more specific theme.
        #
        # "jury selection bias" and "outcome prediction" are listed per spec
        # but are currently unreachable in practice: any text containing
        # "jury selection bias" also contains "jury" (trial_prep, checked
        # first) and any text containing "outcome prediction" also matches
        # scoring_merit's own "outcome prediction" keyword (checked before
        # this theme). Kept anyway since removing them wasn't asked for and
        # they're harmless dead weight, not incorrect.
        "opposing counsel", "insurance policy analysis", "coverage gap", "bad faith",
        "settlement conference", "case similarity", "jury selection bias",
        "outcome prediction",
    ]),
    ("document_ai_quality", [
        "ocr", "medical terminology", "privilege review", "privileged", "redaction",
        "redact", "summariz", "chronology", "citation check", "voice-to-text",
        "transcri", "handwritten", "handwriting", "case summary generator",
        "research memo", "demand letter", "mediation brief", "fee agreement generation",
        "medical illustration", "witness statement analysis", "discovery compliance",
        "medical records", "medical bill review",
    ]),
]

# Sentiment/severity lexicons. Kept deliberately separate from THEME_PRIORITY
# because severity is about blast radius, not topic -- the same theme
# (calculation_accuracy) shows up in both a "crashes and loses my case" report
# and a "saves me hours" compliment. Stems (e.g. "duplicat", "struggl") are
# used instead of enumerating every inflection, since has_term's suffix-open
# matching already covers plurals/tenses -- listing "bug" AND "buggy" would
# just be redundant now.
STRONG_NEGATIVE = [
    "inaccurate", "incorrect", "wrong", "bug", "crash", "error", "fail",
    "dangerous", "embarrassing", "unreliable", "broken", "backwards",
    "inconsistent", "false positive",
]
MODERATE_NEGATIVE = [
    "doesn't", "does not", "don't", "won't", "can't", "cannot", "struggl", "losing",
    "duplicat", "defeat", "stale", "outdated", "slow", "timing out", "timeout",
    "double-book", "double book", "unwieldy", "glitch", "hit or miss",
    "too aggressive", "too complex", "confus", "irrelevant", "jargon",
]
SOFT_REQUEST = [
    "would love", "need better", "needs better", "need more", "needs more",
    "want to see", "missing", "lacks", "could use", "would benefit",
    "more customization", "more templates", "more options", "better filtering",
    "more video", "need",
]
PRAISE_TERMS = [
    # "love" alone would match inside "would love to see" (a feature request,
    # not praise for what exists today) -- narrowed to "love the", the only
    # form this corpus actually uses as genuine praise ("Love the new case
    # timeline feature!").
    "excellent", "fantastic", "great", "love the", "impressive", "huge", "massive",
    "game-changing", "game changing", "essential", "brilliant", "incredible",
    "groundbreaking", "transformed", "highly recommend", "spot-on", "spot on",
    "perfect", "comprehensive", "thorough", "helpful", "helps", "helped", "useful",
    "save", "lifesaver", "wonderful", "amazing", "best", "efficient", "streamlined",
    "eerily accurate",
]


def make_hash(user_id, feedback_date, feedback_text):
    """Hash (user_id, feedback_date, feedback_text) so reruns are idempotent --
    the same physical report always produces the same ledger key even if it
    shows up again in a later incoming batch (e.g. a retried upload)."""
    payload = f"{str(user_id).strip()}|{str(feedback_date).strip()}|{str(feedback_text).strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_seen_hashes(path):
    """Read the dedupe ledger; absent file just means nothing's been seen yet."""
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_seen_hashes(path, hashes):
    """Persist the ledger sorted so diffs in version control are readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(hashes), f, indent=2)


def load_master(path):
    """Empty-but-typed frame when master.csv doesn't exist yet, so downstream
    concat/groupby code never has to special-case 'first run ever'."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=MASTER_COLUMNS)


def collect_new_rows(paths, seen_hashes):
    """Read every source CSV, hash each row, and keep only what's genuinely
    new -- both against the ledger and against duplicates within this same
    batch (two incoming files could carry the same report)."""
    frames = []
    n_read = 0
    for p in paths:
        df = pd.read_csv(p)
        df["row_hash"] = [
            make_hash(u, d, t)
            for u, d, t in zip(df["user_id"], df["feedback_date"], df["feedback_text"])
        ]
        n_read += len(df)
        frames.append(df[~df["row_hash"].isin(seen_hashes)])
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS + ["row_hash"]), 0, 0
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset="row_hash", keep="first")
    n_skipped = n_read - len(combined)
    return combined, n_read, n_skipped


def classify_theme(text):
    """First keyword match wins, in THEME_PRIORITY order -- see that constant
    for why the order is chosen deliberately rather than alphabetically.

    Falls back to "unthemed", not to document_ai_quality or any other real
    category -- silently dumping unmatched rows into the taxonomy's broadest
    bucket would inflate it and hide the fact that the taxonomy doesn't cover
    everything. "unthemed" is a visible admission of a coverage gap, and
    print_summary reports both the count and the actual texts every run so
    the taxonomy can be extended deliberately, not by accident.
    """
    for theme, keywords in THEME_PRIORITY:
        if has_term(text, keywords):
            return theme
    return "unthemed"


def _severity_ladder(text, theme, has_defect):
    """Pure severity determination, independent of sentiment. Some rules here
    (privilege+discoverable, data loss) fire without requiring has_defect,
    because they encode domain knowledge ("a privileged doc marked
    discoverable is malpractice exposure") that the general-purpose
    negative-word lexicon has no way to express -- the phrasing that reports
    this kind of issue often doesn't use ordinary complaint words at all
    ("Need more conservative settings for privilege protection" contains no
    "wrong"/"broken"/"incorrect"). classify_row uses that same fact to
    upgrade sentiment afterward: if a hard rule alone was enough to call a
    row critical, that rule WAS the defect signal, even though the lexicon's
    own vocabulary didn't recognize it as one.

    Returns None (not "unknown"/"low") when nothing in the ladder fires, so
    the caller -- which also knows has_soft -- can decide between an explicit
    nice-to-have ("low") and a row with no signal at all ("unknown").

    Known limitation, not fixed here: "medium" (has_defect with no other
    rule firing) is a wide catch-all -- it holds ~40% of this corpus,
    because most feature requests in this product also contain complaint
    language ("X doesn't do Y, would be great if..."), so has_defect trips
    before has_soft ever gets evaluated. SOFT_REQUEST/"low" only fires for
    the minority of rows that are pure asks with zero complaint wording.
    That's an accepted imprecision, not a bug -- critical/high still surface
    correctly, and Widget 1 sorts by days_open within a severity tier, so a
    crowded "medium" bucket doesn't bury the oldest issues. Not adding more
    tiers to chase this; the taxonomy has five severities by design.
    """
    if has_term(text, ["dangerous"]):
        return "critical"
    if has_term(text, ["privilege"]) and has_term(text, ["discoverable"]):
        return "critical"
    if has_term(text, ["statute of limitations"]) and has_defect:
        return "critical"
    if has_term(text, ["sanctions"]) and has_defect:
        return "critical"
    if has_term(text, ["exhibit"]) and (
        has_term(text, ["reset"]) or has_term(text, ["losing track"]) or has_term(text, ["lost"])
    ):
        return "critical"
    if has_term(text, ["court filing"]) and has_term(text, ["wrong format"]):
        return "critical"
    if has_term(text, ["medical terminology"]) and has_defect:
        return "critical"

    if has_term(text, ["crash"]):
        return "high"
    if has_term(text, ["losing entries"]) or has_term(text, ["data loss"]):
        return "high"
    if theme == "calculation_accuracy" and has_defect:
        # Every calculation_accuracy feature (damages, settlement demand,
        # case value) feeds a money decision, so a defect here is high by
        # definition even without an explicit dollar figure in the text.
        return "high"

    if has_defect:
        return "medium"

    return None


def classify_row(text, theme):
    """Decide sentiment BEFORE severity, and let sentiment gate the severity
    ladder -- not the other way around.

    Risk vocabulary ("sanctions", "discoverable", "statute of limitations",
    "dangerous") reads identically whether it's describing a bug or
    describing the feature that PREVENTS that exact bug ("flags potential
    sanctions risks... saved us from a major screw-up"). A row with praise
    language and zero defect language is describing prevention, so it
    short-circuits straight to praise/positive before any critical/high/
    medium check runs at all.

    Invariant enforced below: severity in (critical, high, medium) implies
    sentiment in (negative, mixed) -- a row can't be simultaneously "this is
    malpractice exposure" and "we detected no problem here." Most of the
    ladder already guarantees this because has_defect is required to reach
    those tiers, but a few hard rules (privilege+discoverable, dangerous,
    data loss) fire on domain knowledge the lexicon doesn't otherwise
    recognize as a defect signal -- so when one of those fires without
    has_defect/has_praise having caught it, the severity finding itself
    upgrades sentiment rather than leaving a self-contradictory row.
    """
    has_strong = has_term(text, STRONG_NEGATIVE)
    has_moderate = has_term(text, MODERATE_NEGATIVE)
    has_soft = has_term(text, SOFT_REQUEST)
    has_praise = has_term(text, PRAISE_TERMS)
    has_defect = has_strong or has_moderate

    if has_praise and not has_defect:
        return "praise", "positive"

    if has_defect and has_praise:
        sentiment = "mixed"
    elif has_defect:
        sentiment = "negative"
    else:
        # Neither praise nor defect language matched (yet -- see invariant
        # upgrade below). Don't guess: this is the honest outcome for text
        # the general lexicon genuinely doesn't cover.
        sentiment = "unclassified"

    severity = _severity_ladder(text, theme, has_defect)
    if severity is None:
        # has_defect is False here (the ladder always returns at least
        # "medium" when has_defect is True), so has_soft is the only
        # remaining signal that distinguishes an explicit nice-to-have
        # ("would love X", "needs better Y") from a row where the lexicon
        # found nothing at all.
        severity = "low" if has_soft else "unknown"

    if severity in ("critical", "high", "medium") and sentiment == "unclassified":
        sentiment = "mixed" if has_praise else "negative"

    assert not (severity in ("critical", "high", "medium") and sentiment not in ("negative", "mixed")), (
        f"severity/sentiment incoherent: severity={severity!r} sentiment={sentiment!r} text={text!r}"
    )

    return severity, sentiment


def tag_rows(df):
    """Tag only rows that are actually new -- re-tagging the whole master
    every run would be wasted work and would risk drifting old rows' labels
    if the taxonomy keyword lists are edited later."""
    df = df.copy()
    lowered = df["feedback_text"].astype(str).str.lower()
    themes, severities, sentiments = [], [], []
    for text in lowered:
        theme = classify_theme(text)
        severity, sentiment = classify_row(text, theme)
        themes.append(theme)
        severities.append(severity)
        sentiments.append(sentiment)
    df["theme"] = themes
    df["severity"] = severities
    df["sentiment"] = sentiments
    return df


class UnionFind:
    """Connected components without a networkx dependency -- the stack is
    pandas/sklearn/streamlit only, and a plain union-find is a dozen lines."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


CLUSTER_STATS_COLUMNS = [
    "cluster_id", "theme", "representative_text", "report_count",
    "first_seen", "last_seen", "report_span_days", "days_open", "max_severity",
    "channels", "roles", "firm_sizes", "member_hashes",
]


def cluster_rows(df, as_of_date, threshold=SIM_THRESHOLD):
    """Cluster ALL rows every run (not just new ones) so that a new report
    can join an existing cluster or bridge two previously-separate ones --
    connected components, not just pairwise matches, is what makes 3+ reports
    of the same issue collapse into a single cluster.

    cluster_id is the row_hash of the chronologically-earliest member, not a
    sequential number. Union-find's root labels are arbitrary and reshuffle
    every run, which would make "cluster grew 2->3" meaningless across runs.
    row_hash never changes for a given report, so anchoring the id to the
    earliest member's hash is deterministic and survives re-clustering --
    including merges: if a new row bridges two previously-separate clusters,
    they naturally combine under whichever anchor is chronologically
    earlier, and diff_clusters (below) detects and logs that as a merge.

    days_open is measured against as_of_date (the max feedback_date across
    the WHOLE corpus, recomputed every run), not against the real wall clock
    -- the seed corpus is from mid-2024, and computing against today's date
    would produce a meaningless "670 days open" figure that has nothing to
    do with actual triage urgency. report_span_days (last_seen - first_seen)
    is kept as a separate field since it answers a different question ("how
    long has this been recurring") from days_open ("how long has this been
    unresolved as of the latest data we have").

    Returns (cluster_stats_df, cluster_id_per_row, is_representative_per_row).
    """
    n = len(df)
    if n == 0:
        return (
            pd.DataFrame(columns=CLUSTER_STATS_COLUMNS),
            pd.Series(dtype=object),
            pd.Series(dtype=bool),
        )

    df = df.reset_index(drop=True)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    tfidf = vectorizer.fit_transform(df["feedback_text"].astype(str))
    sim = cosine_similarity(tfidf)

    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                uf.union(i, j)

    components = {}
    for idx in range(n):
        components.setdefault(uf.find(idx), []).append(idx)

    cluster_id_per_row = [None] * n
    is_rep_per_row = [False] * n
    records = []
    for idxs in components.values():
        anchor_idx = min(idxs, key=lambda i: (df.loc[i, "feedback_date"], df.loc[i, "row_hash"]))
        cluster_id = df.loc[anchor_idx, "row_hash"]
        for idx in idxs:
            cluster_id_per_row[idx] = cluster_id

        if len(idxs) == 1:
            medoid_idx = idxs[0]
        else:
            # Medoid = member with highest average similarity to the rest of
            # the cluster, i.e. the report that best speaks for the group --
            # more honest than "whichever arrived first."
            avg_sim = [(sim[i, idxs].sum() - 1) / (len(idxs) - 1) for i in idxs]
            medoid_idx = idxs[avg_sim.index(max(avg_sim))]
        is_rep_per_row[medoid_idx] = True

        sub = df.loc[idxs]
        first_seen = sub["feedback_date"].min()
        last_seen = sub["feedback_date"].max()
        records.append({
            "cluster_id": cluster_id,
            # Known limitation, not fixed: a cluster's theme is the medoid's
            # theme alone, so members can individually disagree (e.g. one
            # "expert witness database" report themes as data_freshness via
            # "outdated" while a near-duplicate using "inaccurate contact
            # information" instead could theme differently pre-keyword-fix).
            # Mode-of-members would be more correct but adds a tie-breaking
            # question this prototype doesn't need to answer.
            "theme": df.loc[medoid_idx, "theme"],
            "representative_text": df.loc[medoid_idx, "feedback_text"],
            "report_count": len(idxs),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "report_span_days": (pd.to_datetime(last_seen) - pd.to_datetime(first_seen)).days,
            "days_open": (as_of_date - pd.to_datetime(first_seen)).days,
            "max_severity": max(sub["severity"], key=lambda s: SEVERITY_RANK[s]),
            "channels": ",".join(sorted(sub["channel"].unique())),
            "roles": ",".join(sorted(sub["user_role"].unique())),
            "firm_sizes": ",".join(sorted(sub["firm_size"].unique())),
            "member_hashes": ",".join(sub["row_hash"]),
        })

    cluster_stats = pd.DataFrame.from_records(records, columns=CLUSTER_STATS_COLUMNS)
    return (
        cluster_stats,
        pd.Series(cluster_id_per_row, index=df.index, name="cluster_id"),
        pd.Series(is_rep_per_row, index=df.index, name="is_representative"),
    )


def build_cluster_membership(df):
    """Snapshot cluster_id -> set(row_hash) from the PREVIOUS run's master so
    this run can tell what's new, grew, or merged."""
    if df.empty or "cluster_id" not in df.columns or "row_hash" not in df.columns:
        return {}
    return {cid: set(sub["row_hash"]) for cid, sub in df.groupby("cluster_id")}


def diff_clusters(old_members, cluster_stats, new_members):
    """Join old vs. new clusters by row_hash overlap, not by cluster_id.

    row_hash is the stable identity for a report; cluster_id can legitimately
    change when two old clusters merge under an earlier anchor. A new
    cluster whose members were split across 2+ old cluster_ids is a MERGE,
    not a GREW -- reported separately since "these two issues turned out to
    be the same issue" is a materially different finding than "more people
    reported the same issue."
    """
    new_list, grew_list, merge_list = [], [], []
    for _, row in cluster_stats.iterrows():
        cid = row["cluster_id"]
        members = new_members[cid]
        absorbed = {oid: oset for oid, oset in old_members.items() if oset & members}
        if not absorbed:
            new_list.append(row)
        elif len(absorbed) == 1:
            (old_id, old_set), = absorbed.items()
            if len(members) > len(old_set):
                grew_list.append((row, old_id, len(old_set)))
        else:
            merge_list.append((row, absorbed))
    return new_list, grew_list, merge_list


def print_summary(n_sources, n_read, n_skipped, n_new, as_of_date, cluster_stats,
                   new_list, grew_list, merge_list, tagged_df, retagged=False):
    print(f"Sources scanned: {n_sources}")
    print(f"Rows read: {n_read} | new: {n_new} | duplicates skipped: {n_skipped}"
          + (" | retagged: all existing rows" if retagged else ""))
    print(f"As-of date (max feedback_date in corpus): {as_of_date.date()}")
    print(f"Total clusters: {len(cluster_stats)}")

    if not tagged_df.empty:
        n = len(tagged_df)
        print(f"\nTheme distribution ({n} rows):")
        for theme, count in tagged_df["theme"].value_counts().items():
            flag = "  <-- taxonomy coverage gap" if theme == "unthemed" else ""
            print(f"  {theme}: {count}{flag}")

        print(f"\nSeverity distribution ({n} rows):")
        for sev in SEVERITY_ORDER:
            count = (tagged_df["severity"] == sev).sum()
            if count:
                print(f"  {sev}: {count}")

        print(f"\nSentiment distribution ({n} rows):")
        for sent, count in tagged_df["sentiment"].value_counts().items():
            print(f"  {sent}: {count}")

        unclassified = (tagged_df["sentiment"] == "unclassified").sum()
        print(f"\nUnclassified sentiment: {unclassified}/{n} rows "
              f"({unclassified / n:.0%}) -- lexicon found neither praise nor defect language")

        unthemed = tagged_df[tagged_df["theme"] == "unthemed"]
        if not unthemed.empty:
            print(f"\nUnthemed rows ({len(unthemed)}) -- taxonomy doesn't cover these:")
            for _, row in unthemed.iterrows():
                print(f"  [{row['row_hash'][:8]}] {row['feedback_text'][:110]}")

    if new_list:
        print("\nNEW clusters this run:")
        for row in new_list:
            print(f"  {row['cluster_id'][:8]} [{row['theme']}/{row['max_severity']}] "
                  f"n={row['report_count']} days_open={row['days_open']}: "
                  f"{row['representative_text'][:90]}")
    if grew_list:
        print("\nCLUSTERS THAT GREW this run:")
        for row, old_id, old_count in grew_list:
            print(f"  {row['cluster_id'][:8]} [{row['theme']}/{row['max_severity']}] "
                  f"{old_count} -> {row['report_count']}: {row['representative_text'][:90]}")
    if merge_list:
        print("\nCLUSTERS THAT MERGED this run:")
        for row, absorbed in merge_list:
            old_ids = ", ".join(f"{oid[:8]} (n={len(oset)})" for oid, oset in absorbed.items())
            print(f"  {row['cluster_id'][:8]} absorbed [{old_ids}] -> now n={row['report_count']}: "
                  f"{row['representative_text'][:90]}")
    if not new_list and not grew_list and not merge_list:
        print("\nNo new, grown, or merged clusters this run.")


def main():
    parser = argparse.ArgumentParser(description="Ingest and tag Case Companion feedback.")
    parser.add_argument(
        "--retag", action="store_true",
        help="Re-tag every existing row in master.csv with the current taxonomy "
             "rules before ingesting anything new. row_hash and the dedupe ledger "
             "are untouched -- only theme/severity/sentiment are recomputed. Use "
             "this after editing the keyword lists so master.csv never holds a mix "
             "of old- and new-rules labels.",
    )
    args = parser.parse_args()

    seen = load_seen_hashes(HASHES_PATH)
    master_before = load_master(MASTER_PATH)
    if args.retag and not master_before.empty:
        master_before = tag_rows(master_before)
    old_members = build_cluster_membership(master_before)

    sources = []
    if not MASTER_PATH.exists():
        sources.append(SEED_PATH)
    sources.extend(sorted(INCOMING_DIR.glob("*.csv")))

    new_rows, n_read, n_skipped = collect_new_rows(sources, seen)

    if new_rows.empty:
        master_all = master_before.drop(columns=["cluster_id", "is_representative"], errors="ignore")
    else:
        new_rows = tag_rows(new_rows)
        seen.update(new_rows["row_hash"])
        if master_before.empty:
            master_all = new_rows
        else:
            master_all = pd.concat(
                [master_before.drop(columns=["cluster_id", "is_representative"], errors="ignore"), new_rows],
                ignore_index=True, sort=False,
            )

    as_of_date = pd.to_datetime(master_all["feedback_date"]).max()
    cluster_stats, cluster_id_series, is_rep_series = cluster_rows(master_all, as_of_date)
    master_all = master_all.reset_index(drop=True)
    master_all["cluster_id"] = cluster_id_series
    master_all["is_representative"] = is_rep_series
    master_all = master_all.reindex(columns=MASTER_COLUMNS)

    new_members = build_cluster_membership(master_all)
    new_list, grew_list, merge_list = diff_clusters(old_members, cluster_stats, new_members)

    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    master_all.to_csv(MASTER_PATH, index=False)
    save_seen_hashes(HASHES_PATH, seen)

    print_summary(len(sources), n_read, n_skipped, len(new_rows), as_of_date, cluster_stats,
                  new_list, grew_list, merge_list, master_all, retagged=args.retag)


if __name__ == "__main__":
    main()
