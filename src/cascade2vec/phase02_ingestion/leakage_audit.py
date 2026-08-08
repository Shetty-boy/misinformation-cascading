"""
leakage_audit.py — Temporal Leakage Guardrails
================================================
Provides assertion utilities to detect temporal leakage in snapshot graphs.

The canonical guardrail function is assert_snapshot_is_clean(), which MUST be
called before any feature computation on a snapshot graph.  It is a HARD FAILURE
(raises AssertionError, never just warns), because silent leakage is the most
dangerous failure mode in early-detection research — it produces inflated metrics
that only appear later in deployment or ablation.

Usage:
    from cascade2vec.phase02_ingestion.leakage_audit import assert_snapshot_is_clean
    assert_snapshot_is_clean(snapshot_graph, t_seconds)
"""

import logging
from graphframes import GraphFrame

logger = logging.getLogger(__name__)


def assert_snapshot_is_clean(snapshot: GraphFrame, t_seconds: float) -> None:
    """
    Assert that a temporal snapshot graph contains NO future information.

    Checks:
      1. No vertex has timestamp > t_seconds
      2. No edge has timestamp > t_seconds (if edges carry a timestamp column)

    This is a HARD FAILURE — raises AssertionError if any violation is found.
    Never use this as a warn-only check.  Temporal leakage produces inflated
    evaluation metrics that can persist undetected for entire experiment cycles.

    Parameters
    ----------
    snapshot : GraphFrame
        The temporally-sliced graph to validate.
    t_seconds : float
        The observation cutoff in seconds from cascade root.

    Raises
    ------
    AssertionError
        If any vertex or edge timestamp exceeds t_seconds.
    """
    from pyspark.sql import functions as F

    # Check vertices
    if "timestamp" in snapshot.vertices.columns:
        max_v_ts = snapshot.vertices.select(F.max("timestamp")).collect()[0][0]
        if max_v_ts is not None:
            assert max_v_ts <= t_seconds, (
                f"TEMPORAL LEAKAGE DETECTED: Vertex timestamp {max_v_ts}s "
                f"exceeds snapshot cutoff {t_seconds}s. "
                f"All features computed from this snapshot are invalid."
            )

    # Check edges (if timestamped — edges typically inherit via vertex join)
    if "timestamp" in snapshot.edges.columns:
        max_e_ts = snapshot.edges.select(F.max("timestamp")).collect()[0][0]
        if max_e_ts is not None:
            assert max_e_ts <= t_seconds, (
                f"TEMPORAL LEAKAGE DETECTED: Edge timestamp {max_e_ts}s "
                f"exceeds snapshot cutoff {t_seconds}s."
            )


def flag_suspicious_correlations(
    feature_df,
    threshold: float = 0.95,
    method: str = "pearson",
) -> list[tuple[str, str, float]]:
    """
    Identify pairs of features with suspiciously high correlation.

    This function does NOT prune anything.  It only flags pairs whose absolute
    correlation exceeds `threshold`.  Pruning decisions are deferred to Phase 18
    ablations — see logs/phase06_07_features/feature_correlation.md.

    Parameters
    ----------
    feature_df : pd.DataFrame
        A pandas DataFrame of numeric features (one row per sample).
    threshold : float
        Absolute correlation threshold for flagging. Default 0.95.
    method : str
        'pearson' or 'spearman'.

    Returns
    -------
    list of (feature_a, feature_b, correlation_value)
        Sorted by descending |correlation|.
    """
    import pandas as pd

    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    corr_matrix = feature_df[numeric_cols].corr(method=method)

    flagged = []
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            val = corr_matrix.loc[col_a, col_b]
            if abs(val) >= threshold:
                flagged.append((col_a, col_b, round(float(val), 4)))

    flagged.sort(key=lambda x: abs(x[2]), reverse=True)
    return flagged
