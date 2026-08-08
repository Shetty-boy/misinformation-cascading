"""
build_feature_matrix.py — Phase 6 Feature Matrix Pipeline
==========================================================
Runs the full pipeline:
    Cascade -> Snapshot -> Features -> Feature Matrix

Output
------
data/processed/phase06_07_features/feature_matrix.parquet
    One row per (cascade_id, t_minutes). Columns: all Phase 6A + 6B features,
    plus 'label' (rumour / non-rumour from unified.parquet folder structure).

Label source
------------
The 'label' column is read directly from unified.parquet. It contains only
'rumour' and 'non-rumour' values (folder-based PHEME labeling). No annotation.json
veracity labels (true/false/unverified) are used here — those are out of scope
for Phase 6-7. See docs/phase06_07_features/classification_protocol.md.

Leakage contract
----------------
assert_snapshot_is_clean() is called inside compute_features() for EVERY
(cascade_id, t) pair. If it raises AssertionError the entire run fails immediately.
Never downgrade this to a warning.
"""

import logging
import os
import time

import pandas as pd

from cascade2vec.phase04_05_graph.loader import get_spark, load_unified
from cascade2vec.phase04_05_graph.build_graph import (
    to_vertices, to_edges, build_full_graph, get_cascade_subgraph,
)
from cascade2vec.phase06_07_features.engineering import compute_features

logger = logging.getLogger(__name__)

# Observation time windows (minutes from cascade root)
TIME_WINDOWS_MINUTES = [1, 2, 5, 10, 15, 30, 60, 120]

# Velocity delta (must match engineering.py default)
DELTA_T_MINUTES = 5.0

# Output path
OUT_DIR = "data/processed/phase06_07_features"
OUT_PATH = f"{OUT_DIR}/feature_matrix.parquet"


def _get_cascade_label(cascade_id: str, label_map: dict) -> str:
    """Look up the rumour/non-rumour label for a cascade."""
    return label_map.get(cascade_id, "unknown")


def build_feature_matrix(
    limit_cascades: int | None = None,
    time_windows: list[float] = TIME_WINDOWS_MINUTES,
    skip_disconnected: bool = False,
) -> pd.DataFrame:
    """
    Build the full feature matrix.

    Parameters
    ----------
    limit_cascades : int or None
        If set, process only the first N cascades (for debugging/dry runs).
    time_windows : list of float
        Observation time windows in minutes.
    skip_disconnected : bool
        If True, exclude cascades that are known to be structurally disconnected
        (edge_count < node_count - 1). Default False.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per (cascade_id, t_minutes).
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    spark = get_spark("cascade2vec-phase06-features")
    spark.sparkContext.setLogLevel("WARN")

    logger.info("[build_fm] Loading unified dataset...")
    df = load_unified(spark)

    # Build cascade-level label map (rumour / non-rumour ONLY)
    label_map_pd = (
        df.select("cascade_id", "label")
        .distinct()
        .toPandas()
        .set_index("cascade_id")["label"]
        .to_dict()
    )
    logger.info("[build_fm] Label map built: %d cascades", len(label_map_pd))

    # Build graph
    vertices = to_vertices(df)
    edges = to_edges(df, vertices=vertices)
    full_graph = build_full_graph(vertices, edges)

    # Optionally load disconnected cascade list
    disc_set: set = set()
    if skip_disconnected:
        stats_path = "data/processed/phase04_05_graph/graph_stats.parquet"
        if os.path.exists(stats_path):
            stats_pd = pd.read_parquet(stats_path)
            disc_set = set(
                stats_pd[~stats_pd["is_connected"]]["cascade_id"].tolist()
            )
            logger.info("[build_fm] Skipping %d disconnected cascades", len(disc_set))

    # Cascade IDs to process
    all_cascade_ids = list(label_map_pd.keys())
    if skip_disconnected:
        all_cascade_ids = [c for c in all_cascade_ids if c not in disc_set]
    if limit_cascades:
        all_cascade_ids = all_cascade_ids[:limit_cascades]

    logger.info("[build_fm] Processing %d cascades × %d time windows...",
                len(all_cascade_ids), len(time_windows))

    rows = []
    t_start = time.time()

    for i, cid in enumerate(all_cascade_ids):
        if i % 100 == 0:
            elapsed = time.time() - t_start
            logger.info("[build_fm] Cascade %d / %d (%.1fs elapsed)", i, len(all_cascade_ids), elapsed)

        try:
            subgraph = get_cascade_subgraph(full_graph, cid)
            subgraph.vertices.cache()
            subgraph.edges.cache()

            label = _get_cascade_label(cid, label_map_pd)
            prev_features: dict | None = None

            for t_min in sorted(time_windows):
                try:
                    feats = compute_features(
                        subgraph=subgraph,
                        t_minutes=t_min,
                        cascade_id=cid,
                        prev_features=prev_features,
                        delta_t_minutes=DELTA_T_MINUTES,
                    )
                    feats["label"] = label
                    feats["label_binary"] = 1 if label == "rumour" else 0
                    rows.append(feats)
                    prev_features = feats
                except AssertionError as e:
                    # Leakage check failed — propagate immediately as hard failure
                    raise
                except Exception as e:
                    logger.warning(
                        "[build_fm] Cascade %s at t=%smin failed: %s", cid, t_min, e
                    )
                    prev_features = None

            subgraph.vertices.unpersist()
            subgraph.edges.unpersist()

        except AssertionError:
            raise
        except Exception as e:
            logger.warning("[build_fm] Cascade %s skipped entirely: %s", cid, e)

    total_elapsed = time.time() - t_start
    logger.info("[build_fm] Done. %d rows produced in %.1fs", len(rows), total_elapsed)

    if not rows:
        raise RuntimeError("No rows produced — feature matrix is empty.")

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(OUT_PATH, index=False)
    logger.info("[build_fm] Feature matrix written to %s", OUT_PATH)

    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Build Phase 6 feature matrix")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N cascades (for dry runs)")
    parser.add_argument("--skip-disconnected", action="store_true",
                        help="Exclude structurally disconnected cascades")
    args = parser.parse_args()

    t0 = time.time()
    df_result = build_feature_matrix(
        limit_cascades=args.limit,
        skip_disconnected=args.skip_disconnected,
    )
    t1 = time.time()

    print(f"\n=== FEATURE MATRIX COMPLETE ===")
    print(f"Shape: {df_result.shape}")
    print(f"Columns: {df_result.columns.tolist()}")
    print(f"Label distribution:\n{df_result['label'].value_counts()}")
    print(f"Total runtime: {t1 - t0:.1f}s")
    print(f"Output: {OUT_PATH}")
