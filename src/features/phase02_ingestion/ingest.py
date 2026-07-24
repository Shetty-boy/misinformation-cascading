"""
Data ingestion pipeline for CASCADE2VEC.

Loads PHEME and Twitter15/16 datasets, unifies them under a single schema,
deduplicates on tweet_id, and writes the unified Parquet file plus a
data audit report.

Usage:
    python -m src.features.phase02_ingestion.ingest \
        --pheme data/raw/pheme-rnr-dataset \
        --twitter15 data/raw/twitter15 \
        --twitter16 data/raw/twitter16 \
        --out data/processed/02_ingestion \
        --report experiments/logs/02_ingestion/data_audit.md
"""

import argparse
import os
from collections import Counter
from datetime import datetime

import pandas as pd

from .adapters.pheme import load_pheme
from .adapters.twitter15 import load_twitter
from .adapters.twitter16 import load_twitter16
from .schema import SCHEMA_COLUMNS


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Deduplicate on tweet_id, keeping first occurrence. Returns (df, n_dropped)."""
    before = len(df)
    df = df.drop_duplicates(subset=["tweet_id"], keep="first")
    return df, before - len(df)


# ---------------------------------------------------------------------------
# Timestamp normalisation (per cascade)
# ---------------------------------------------------------------------------

def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cascade, ensure the root tweet (parent_id is NaN) has timestamp=0
    and all other tweets are offset in seconds relative to the root.
    If a cascade has no root row with a valid timestamp, leave timestamps as-is.
    """
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    for cascade_id, grp in df.groupby("cascade_id"):
        root_rows = grp[grp["parent_id"].isna()]
        if root_rows.empty or root_rows["timestamp"].isna().all():
            continue
        root_ts = root_rows["timestamp"].dropna().iloc[0]
        if root_ts == 0:
            continue
        df.loc[grp.index, "timestamp"] = df.loc[grp.index, "timestamp"] - root_ts

    return df


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * n / total:.1f}%"


def generate_audit_report(
    df: pd.DataFrame,
    *,
    n_dupes_dropped: int,
    parse_failures_by_source: dict[str, int],
    output_path: str,
) -> str:
    """Generate a Markdown data-audit report and write it to *output_path*."""
    total = len(df)
    lines = []
    lines.append("# CASCADE2VEC — Data Audit Report")
    lines.append(f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    # --- Overview ---
    lines.append("## Overview")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total rows (post-dedup) | {total:,} |")
    lines.append(f"| Duplicates dropped | {n_dupes_dropped:,} |")
    for src, n in parse_failures_by_source.items():
        lines.append(f"| Parse failures ({src}) | {n:,} |")

    # --- Per-source breakdown ---
    lines.append("\n## Rows per Dataset")
    lines.append("| event_id | rows | cascades |")
    lines.append("|----------|------|---------|")
    for event_id, grp in df.groupby("event_id"):
        lines.append(f"| {event_id} | {len(grp):,} | {grp['cascade_id'].nunique():,} |")

    # --- Class balance ---
    lines.append("\n## Class Balance")
    lines.append("| label | count | % of total |")
    lines.append("|-------|-------|-----------|")
    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        lines.append(f"| {label} | {count:,} | {_pct(count, total)} |")

    # --- Missing fields ---
    lines.append("\n## Missing Fields")
    lines.append("| column | missing | % missing |")
    lines.append("|--------|---------|----------|")
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            lines.append(f"| {col} | N/A (column absent) | — |")
            continue
        n_missing = df[col].isna().sum() + (df[col] == "").sum()
        lines.append(f"| {col} | {n_missing:,} | {_pct(n_missing, total)} |")

    # --- Timestamp sanity ---
    lines.append("\n## Timestamp Sanity")
    ts_series = pd.to_numeric(df["timestamp"], errors="coerce")
    n_negative = (ts_series < 0).sum()
    n_null_ts = ts_series.isna().sum()
    lines.append(f"- Rows with negative timestamp (replies before root): **{n_negative:,}** ({_pct(n_negative, total)})")
    lines.append(f"- Rows with null/unparseable timestamp: **{n_null_ts:,}** ({_pct(n_null_ts, total)})")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ingestion(
    pheme_root: str | None,
    twitter15_root: str | None,
    twitter16_root: str | None,
    out_dir: str,
    report_path: str,
) -> pd.DataFrame:
    all_rows: list[dict] = []
    parse_failures: dict[str, int] = {}

    if pheme_root and os.path.isdir(pheme_root):
        print(f"[ingest] Loading PHEME from {pheme_root} ...")
        pheme_rows = load_pheme(pheme_root)
        print(f"[ingest]   → {len(pheme_rows):,} rows")
        all_rows.extend(pheme_rows)
        # Count internal parse failures (printed to stdout by adapter)
        parse_failures["pheme"] = 0  # adapter logs internally

    if twitter15_root and os.path.isdir(twitter15_root):
        print(f"[ingest] Loading Twitter15 from {twitter15_root} ...")
        t15_rows = load_twitter(twitter15_root, event_id="twitter15")
        print(f"[ingest]   → {len(t15_rows):,} rows")
        all_rows.extend(t15_rows)
        parse_failures["twitter15"] = 0

    if twitter16_root and os.path.isdir(twitter16_root):
        print(f"[ingest] Loading Twitter16 from {twitter16_root} ...")
        t16_rows = load_twitter16(twitter16_root)
        print(f"[ingest]   → {len(t16_rows):,} rows")
        all_rows.extend(t16_rows)
        parse_failures["twitter16"] = 0

    if not all_rows:
        print("[ingest] WARNING: No rows loaded. Check dataset paths.")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df = pd.DataFrame(all_rows, columns=SCHEMA_COLUMNS)

    # Deduplication
    print(f"[ingest] Deduplicating on tweet_id ...")
    df, n_dupes = deduplicate(df)
    print(f"[ingest]   → {n_dupes:,} duplicates dropped. {len(df):,} rows remain.")

    # Timestamp normalization
    print(f"[ingest] Normalizing timestamps relative to cascade roots ...")
    df = normalize_timestamps(df)

    # Write output
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "unified.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[ingest] Wrote unified dataset → {out_path}")

    # Audit report
    print(f"[ingest] Generating audit report → {report_path}")
    report = generate_audit_report(
        df,
        n_dupes_dropped=n_dupes,
        parse_failures_by_source=parse_failures,
        output_path=report_path,
    )
    print(report)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="CASCADE2VEC data ingestion pipeline")
    parser.add_argument("--pheme", default=None)
    parser.add_argument("--twitter15", default=None)
    parser.add_argument("--twitter16", default=None)
    parser.add_argument("--out", default="data/processed/02_ingestion")
    parser.add_argument("--report", default="experiments/logs/02_ingestion/data_audit.md")
    args = parser.parse_args()

    run_ingestion(
        pheme_root=args.pheme,
        twitter15_root=args.twitter15,
        twitter16_root=args.twitter16,
        out_dir=args.out,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
