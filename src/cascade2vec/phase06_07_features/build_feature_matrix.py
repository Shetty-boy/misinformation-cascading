"""
build_feature_matrix_pandas.py — Phase 6 Feature Matrix Pipeline (Pandas)
=========================================================================
Runs the full pipeline using in-memory Pandas dataframes.
"""

import logging
import os
import time
import pandas as pd

from cascade2vec.phase04_05_graph.loader import get_spark, load_unified
from cascade2vec.phase04_05_graph.build_graph import to_vertices, to_edges
from cascade2vec.phase06_07_features.engineering import compute_features_pandas
from cascade2vec.phase02_ingestion.leakage_audit import flag_suspicious_correlations

logger = logging.getLogger(__name__)

TIME_WINDOWS_MINUTES = [1, 2, 5, 10, 15, 30, 60, 120]
DELTA_T_MINUTES = 5.0

OUT_DIR = "data/processed/phase06_07_features"
OUT_PATH = f"{OUT_DIR}/feature_matrix_pandas.parquet"

def _get_cascade_label(cascade_id: str, label_map: dict) -> str:
    return label_map.get(cascade_id, "unknown")

def build_feature_matrix(
    limit_cascades: int | None = None,
    time_windows: list[float] = TIME_WINDOWS_MINUTES,
    skip_disconnected: bool = False,
) -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)

    spark = get_spark("cascade2vec-phase06-features-pandas")
    spark.sparkContext.setLogLevel("WARN")

    logger.info("[build_fm] Loading unified dataset...")
    df = load_unified(spark)

    label_map_pd = (
        df.select("cascade_id", "label")
        .distinct()
        .toPandas()
        .set_index("cascade_id")["label"]
        .to_dict()
    )
    logger.info("[build_fm] Label map built: %d cascades", len(label_map_pd))

    # 1. Use Spark only once upfront
    vertices_spark = to_vertices(df)
    edges_spark = to_edges(df, vertices=vertices_spark)
    
    # Collect entirely to Pandas
    logger.info("[build_fm] Collecting entire graph to Pandas...")
    vertices_pd = vertices_spark.toPandas()
    edges_pd = edges_spark.toPandas()
    logger.info(f"[build_fm] Collected {len(vertices_pd)} vertices and {len(edges_pd)} edges.")

    disc_set: set = set()
    if skip_disconnected:
        stats_path = "data/processed/phase04_05_graph/graph_stats.parquet"
        if os.path.exists(stats_path):
            stats_pd = pd.read_parquet(stats_path)
            disc_set = set(stats_pd[~stats_pd["is_connected"]]["cascade_id"].tolist())
            logger.info("[build_fm] Skipping %d disconnected cascades", len(disc_set))

    all_cascade_ids = list(label_map_pd.keys())
    if skip_disconnected:
        all_cascade_ids = [c for c in all_cascade_ids if c not in disc_set]
    if limit_cascades:
        all_cascade_ids = all_cascade_ids[:limit_cascades]

    logger.info("[build_fm] Processing %d cascades × %d time windows...",
                len(all_cascade_ids), len(time_windows))

    rows = []
    t_start = time.time()

    # Create GroupBy objects for faster extraction
    v_grouped = vertices_pd.groupby('cascade_id')
    e_grouped = edges_pd.groupby('cascade_id')

    for i, cid in enumerate(all_cascade_ids):
        if i % 100 == 0:
            elapsed = time.time() - t_start
            logger.info("[build_fm] Cascade %d / %d (%.1fs elapsed)", i, len(all_cascade_ids), elapsed)

        try:
            # Extract subgraph for this cascade
            try:
                c_vertices = v_grouped.get_group(cid).copy()
            except KeyError:
                continue
                
            try:
                c_edges = e_grouped.get_group(cid).copy()
            except KeyError:
                c_edges = pd.DataFrame(columns=edges_pd.columns)

            label = _get_cascade_label(cid, label_map_pd)
            prev_features: dict | None = None

            for t_min in sorted(time_windows):
                try:
                    feats = compute_features_pandas(
                        vertices=c_vertices,
                        edges=c_edges,
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
                    raise
                except Exception as e:
                    logger.warning("[build_fm] Cascade %s at t=%smin failed: %s", cid, t_min, e)
                    prev_features = None

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

    logger.info("[build_fm] Generating feature correlation report...")
    flag_suspicious_correlations(result_df)

    return result_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-disconnected", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    df_result = build_feature_matrix(
        limit_cascades=args.limit,
        skip_disconnected=args.skip_disconnected,
    )
    t1 = time.time()

    print(f"\\n=== FEATURE MATRIX COMPLETE ===")
    print(f"Shape: {df_result.shape}")
    print(f"Total runtime: {t1 - t0:.1f}s")
