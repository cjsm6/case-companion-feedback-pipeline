"""Streamlit viewer for the Case Companion feedback pipeline's output.

Run with: streamlit run app.py

Read-only by design: all the expensive work (tagging, TF-IDF clustering) is
pipeline.py's job, run offline on a schedule. This just groups by the
cluster_id column pipeline.py already wrote to master.csv and renders it --
no sklearn import needed here at all, and no risk of the UI's view of a
cluster drifting from what the pipeline actually computed.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import MASTER_PATH, SEVERITY_ORDER, SEVERITY_RANK

DISPLAY_COLUMNS = [
    "feedback_text", "sentiment", "firm_size",
    "channel", "user_role", "theme", "severity", "feedback_date",
]

# Severity is a STATE (good -> critical), not an arbitrary series identity, so
# it wears the dataviz skill's reserved status palette rather than generic
# categorical hues -- these four hex values are fixed, not themed, and never
# reused for anything else in this app. "unknown" isn't a real severity tier
# (the lexicon found no signal at all), so it gets the neutral/muted ink
# color instead of a status color, keeping it visually distinct from "this
# was actually triaged as low."
SEVERITY_COLORS = {
    "critical": "#d03b3b",
    "high": "#ec835a",
    "medium": "#fab219",
    "low": "#0ca30c",
    "unknown": "#898781",
}
# Critical first so every bar's critical segment shares the same baseline --
# stacked bars only let you compare the first segment at a glance, and
# critical-by-theme is the comparison that matters most here.
ISSUE_SEVERITIES = ["critical", "high", "medium", "low", "unknown"]

# Enterprise support triage model: customer size outranks severity, not the
# other way around -- a large firm's high-severity issue outranks a small
# firm's critical one, because that's how the business actually prioritizes
# support load. FIRM_TIER_LABEL is the inverse lookup for the callout badges.
FIRM_TIER = {"large": 2, "medium": 1, "small": 0}
FIRM_TIER_LABEL = {2: "Large firm", 1: "Medium firm", 0: "Small firm"}

# Theme keys that read badly under generic title-case; everything else falls
# through to THEME.replace("_", " ").title().
THEME_DISPLAY_NAMES = {
    "document_ai_quality": "Document AI Quality",
    "scale_performance": "Scale & Performance",
}

# Sentiment is a status too (good -> bad), so it gets the same fixed,
# reserved-meaning treatment as SEVERITY_COLORS rather than a categorical hue.
SENTIMENT_COLORS = {
    "positive": "#0ca30c",
    "mixed": "#fab219",
    "negative": "#d03b3b",
    "unclassified": "#898781",
}
# Negative first, same baseline-first logic as ISSUE_SEVERITIES (critical
# first): "Who's Unhappy" is a negative-share comparison across roles/firm
# sizes, so every bar's negative segment needs to start at the same edge.
SENTIMENT_ORDER = ["negative", "mixed", "unclassified", "positive"]


def theme_display_name(theme):
    return THEME_DISPLAY_NAMES.get(theme, theme.replace("_", " ").title())


def truncate_at_word(text, limit):
    """Cuts at the last whole word inside the limit rather than mid-word --
    a snippet ending "...calculat" reads as broken; "...calculator" doesn't."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


@st.cache_data
def _read_master(path_str, mtime):
    """mtime is part of the cache key, not used otherwise -- it's what makes
    a `python pipeline.py` rerun (which changes the file's mtime) visible
    here without restarting Streamlit. pipeline.py and app.py are meant to
    run as separate long-lived processes, so the app can't just hold the
    DataFrame in memory across pipeline runs."""
    df = pd.read_csv(path_str, parse_dates=["feedback_date"])
    return df


def load_master():
    if not MASTER_PATH.exists():
        return pd.DataFrame()
    return _read_master(str(MASTER_PATH), MASTER_PATH.stat().st_mtime)


def build_cluster_view(df):
    """One row per cluster_id, aggregated from master.csv's per-row columns.
    Every number here (days_open, max_severity, ...) is a groupby over
    labels pipeline.py already computed -- this function never re-derives
    anything pipeline.py is responsible for, it just presents it.

    Default sort is a tuple rank matching an enterprise support triage model:
    (firm_tier desc, severity_rank desc, has_paralegal desc, days_open desc).
    Customer size outranks severity on purpose -- a large firm's high-severity
    issue outranks a small firm's critical one, because that's how enterprise
    support actually triages. This is the single order every priority-facing
    widget shares by default; render_prioritize_widget's "Severity only" lens
    re-sorts a copy rather than changing this one, so "what ships by default"
    and "what you get when you ask for severity-only" are never the same
    function computing two different things. Praise clusters always rank last
    on severity_rank (0), so they sink to the bottom here regardless of firm
    size -- harmless, since the Praise widget re-sorts by report_count.
    """
    as_of = df["feedback_date"].max()
    rep_rows = df[df["is_representative"]].set_index("cluster_id")

    records = []
    for cid, sub in df.groupby("cluster_id"):
        first_seen = sub["feedback_date"].min()
        last_seen = sub["feedback_date"].max()
        rep = rep_rows.loc[cid] if cid in rep_rows.index else sub.iloc[0]
        records.append({
            "cluster_id": cid,
            "theme": rep["theme"],
            "representative_text": rep["feedback_text"],
            "report_count": len(sub),
            "first_seen": first_seen.date(),
            "last_seen": last_seen.date(),
            "days_open": (as_of - first_seen).days,
            "report_span_days": (last_seen - first_seen).days,
            "max_severity": max(sub["severity"], key=lambda s: SEVERITY_RANK[s]),
            "channels": ", ".join(sorted(sub["channel"].unique())),
            "roles": ", ".join(sorted(sub["user_role"].unique())),
            "firm_sizes": ", ".join(sorted(sub["firm_size"].unique())),
            "firm_tier": max(FIRM_TIER[f] for f in sub["firm_size"].unique()),
            "has_paralegal": (sub["user_role"] == "paralegal").any(),
        })

    clusters = pd.DataFrame.from_records(records)
    clusters["severity_rank"] = clusters["max_severity"].map(SEVERITY_RANK)
    clusters = clusters.sort_values(
        ["firm_tier", "severity_rank", "has_paralegal", "days_open"],
        ascending=[False, False, False, False],
    )
    return clusters


def render_overview_widget(clusters):
    """The 5-second read before anyone starts clicking into anything else:
    how many distinct problems are urgent and how many are corroborated by
    more than one reporter. All four tiles count CLUSTERS (distinct issues),
    not raw report rows, to stay consistent with the deduplicated widgets
    below.
    """
    st.header("Overview")

    urgent = clusters[clusters["max_severity"].isin(["critical", "high"])]
    oldest_urgent = urgent["days_open"].max() if not urgent.empty else None
    corroborated = (clusters["max_severity"] != "praise") & (clusters["report_count"] >= 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critical clusters", int((clusters["max_severity"] == "critical").sum()))
    c2.metric("High clusters", int((clusters["max_severity"] == "high").sum()))
    c3.metric("Oldest urgent issue (days)", int(oldest_urgent) if oldest_urgent is not None else "—")
    c4.metric("Corroborated issues", int(corroborated.sum()))


def _format_priority_line(row):
    theme_name = theme_display_name(row["theme"])
    report_word = "report" if row["report_count"] == 1 else "reports"
    badges = [f"🏢 {FIRM_TIER_LABEL[row['firm_tier']]}"]
    if row["has_paralegal"]:
        badges.append("👤 Paralegal-reported")
    snippet = truncate_at_word(row["representative_text"], 90)
    return (
        f"🔴 **{theme_name}.** "
        f"{row['report_count']} {report_word}, {row['days_open']} days open. "
        f"{' '.join(badges)} — \"{snippet}\""
    )


def render_prioritize_widget(clusters):
    """The verdict, not just the data: a prose callout instead of a table --
    criticals only, no cap. Criticals are rare by construction (~5 in this
    corpus), so listing every one of them *is* the finding; a tiering/overflow
    scheme would just be complexity with nothing left to overflow. No LLM --
    every word is generated from columns already on `clusters`, which
    arrives pre-sorted (firm_tier desc, severity_rank desc, has_paralegal
    desc, days_open desc) from build_cluster_view."""
    st.header("This Week's Priorities")
    st.caption(
        "Ranked by customer size first, then severity — your largest accounts' "
        "critical issues surface first."
    )

    criticals = clusters[clusters["max_severity"] == "critical"]
    if criticals.empty:
        st.write("No critical issues open.")
        return

    lines = [_format_priority_line(row) for _, row in criticals.iterrows()]
    st.markdown("\n\n".join(lines))
    st.markdown("*High and medium issues: see the Issue Clusters table below.*")


def _sentiment_share(df, group_col):
    """Row-normalized (percent-of-group) sentiment counts straight off
    master.csv's sentiment column -- pipeline.py's labels are the single
    source of truth, this never re-derives sentiment."""
    counts = df.groupby([group_col, "sentiment"]).size().unstack(fill_value=0)
    for s in SENTIMENT_ORDER:
        if s not in counts.columns:
            counts[s] = 0
    counts = counts[SENTIMENT_ORDER]
    return counts.div(counts.sum(axis=1), axis=0) * 100


def render_unhappy_widget(df):
    st.header("Who's Unhappy")

    left, right = st.columns(2)
    role_pct = _sentiment_share(df, "user_role")
    firm_pct = _sentiment_share(df, "firm_size")

    with left:
        st.bar_chart(
            role_pct,
            color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
            horizontal=True,
            height=max(180, 40 * len(role_pct)),
        )
    with right:
        st.bar_chart(
            firm_pct,
            color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
            horizontal=True,
            height=max(180, 40 * len(firm_pct)),
        )

    st.caption(
        "Negative users average ~400 queries/month — the same as positive users. "
        "Frustration, not disengagement. Large firms are also the heaviest "
        "users (~588 queries/month vs ~243 at small firms)."
    )


def render_theme_bar_widget(clusters):
    st.header("Open Issues by Theme")
    st.caption("Distinct issues (clusters), not raw reports. Praise excluded -- this is a triage view.")

    issues = clusters[clusters["max_severity"] != "praise"]
    if issues.empty:
        st.write("No open issues.")
        return

    by_theme = issues.groupby(["theme", "max_severity"]).size().unstack(fill_value=0)
    for sev in ISSUE_SEVERITIES:
        if sev not in by_theme.columns:
            by_theme[sev] = 0
    by_theme = by_theme[ISSUE_SEVERITIES]
    # Busiest theme first -- horizontal orientation because theme names
    # (e.g. "calculation_accuracy") are long, per the dataviz form guide's
    # "go horizontal for many/long-named categories" rule for part-to-whole bars.
    by_theme = by_theme.loc[by_theme.sum(axis=1).sort_values(ascending=False).index]
    st.bar_chart(
        by_theme,
        color=[SEVERITY_COLORS[s] for s in ISSUE_SEVERITIES],
        horizontal=True,
        height=max(220, 32 * len(by_theme)),
    )


def render_cluster_widget(df, clusters):
    """Was N stacked st.expanders (too many to scroll to find anything).
    Now: one scannable table (ordered by build_cluster_view's single triage
    sort) plus one focused detail view for whichever single cluster you
    pick -- same information the old expander body showed, just for one
    cluster at a time instead of all of them rendered at once."""
    st.header("Issue Clusters")
    st.caption("Deduplicated reports -- 2+ reports of the same issue collapse into one row.")

    issues = clusters[clusters["max_severity"] != "praise"]
    if issues.empty:
        st.write("No open issues.")
        return

    scan = issues.copy()
    scan["representative_text"] = scan["representative_text"].str.slice(0, 80)
    scan = scan.rename(columns={"max_severity": "severity"})
    st.dataframe(
        scan[["severity", "theme", "report_count", "days_open", "firm_sizes", "roles", "representative_text"]],
        hide_index=True, width="stretch",
    )

    options = {
        f"[{row.max_severity.upper()}] {row.theme} — {row.report_count} reports — {row.representative_text[:60]}":
            row.cluster_id
        for row in issues.itertuples()
    }
    label = st.selectbox("Inspect a cluster", list(options.keys()))
    cid = options[label]
    row = issues[issues["cluster_id"] == cid].iloc[0]

    st.write(row["representative_text"])
    c1, c2 = st.columns(2)
    c1.metric("Reports", row["report_count"])
    c2.metric("Days open", row["days_open"])

    st.write(f"**Channels:** {row['channels']}")
    st.write(f"**Roles:** {row['roles']}")
    st.write(f"**Firm sizes:** {row['firm_sizes']}")
    st.write(f"**First seen:** {row['first_seen']} — **Last seen:** {row['last_seen']}")

    st.write("**All member reports:**")
    members = df[df["cluster_id"] == cid].sort_values("feedback_date")
    st.dataframe(members[DISPLAY_COLUMNS], hide_index=True, width="stretch")


def render_explorer_widget(df):
    st.header("Raw Feedback Explorer")

    c1, c2, c3 = st.columns(3)
    channels = c1.multiselect("Channel", sorted(df["channel"].unique()))
    roles = c2.multiselect("Role", sorted(df["user_role"].unique()))
    firm_sizes = c3.multiselect("Firm size", sorted(df["firm_size"].unique()))

    c4, c5, c6 = st.columns(3)
    sentiments = c4.multiselect("Sentiment", sorted(df["sentiment"].unique()))
    severities = c5.multiselect("Severity", [s for s in SEVERITY_ORDER if s in df["severity"].unique()])
    themes = c6.multiselect("Theme", sorted(df["theme"].unique()))

    min_date, max_date = df["feedback_date"].min().date(), df["feedback_date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    search = st.text_input("Search feedback text")

    filtered = df
    if channels:
        filtered = filtered[filtered["channel"].isin(channels)]
    if roles:
        filtered = filtered[filtered["user_role"].isin(roles)]
    if firm_sizes:
        filtered = filtered[filtered["firm_size"].isin(firm_sizes)]
    if sentiments:
        filtered = filtered[filtered["sentiment"].isin(sentiments)]
    if severities:
        filtered = filtered[filtered["severity"].isin(severities)]
    if themes:
        filtered = filtered[filtered["theme"].isin(themes)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["feedback_date"].dt.date >= start) & (filtered["feedback_date"].dt.date <= end)
        ]
    if search:
        filtered = filtered[filtered["feedback_text"].str.contains(search, case=False, na=False)]

    st.caption(f"{len(filtered)} of {len(df)} rows")
    st.dataframe(
        filtered[DISPLAY_COLUMNS].sort_values("feedback_date", ascending=False),
        hide_index=True, width="stretch",
    )


def main():
    st.set_page_config(page_title="Case Companion Feedback", layout="wide")
    st.title("Case Companion Feedback Analysis")

    df = load_master()
    if df.empty:
        st.warning("No data yet. Run `python pipeline.py` first.")
        return

    clusters = build_cluster_view(df)

    render_overview_widget(clusters)
    st.divider()
    render_prioritize_widget(clusters)
    st.divider()
    render_unhappy_widget(df)
    st.divider()
    render_theme_bar_widget(clusters)
    st.divider()
    render_cluster_widget(df, clusters)
    st.divider()
    render_explorer_widget(df)


if __name__ == "__main__":
    main()
