"""
engineering.py — Phase 6A/6B Feature Engineering
=================================================
Computes structural and temporal features from temporal cascade snapshots.

Feature groups
--------------
Phase 6A — Core Features:
  Structural: node_count, edge_count, max_depth, avg_depth, leaf_count,
              leaf_ratio, branching_factor, root_degree, reachable_ratio,
              is_connected
  Temporal:   tweets_per_minute, growth_velocity, mean_interarrival,
              std_interarrival, burstiness, cascade_age

Phase 6B — Hybrid velocity features (configurable delta_t, default 5 min):
  depth_velocity, breadth_velocity, branching_velocity
  Formula: (feature(t) - feature(t - delta_t)) / delta_t
  Only pre-t snapshots are used in the delta computation.

Leakage contract
----------------
- assert_snapshot_is_clean() is called before EVERY feature computation.
  It is a HARD FAILURE (raises AssertionError) if violated.
- The denominator for reachable_ratio is the node count of the snapshot, NOT
  the final cascade size. See inline comment for explanation.

Depth dependency
----------------
All depth-dependent features (max_depth, avg_depth, reachable_ratio) use
compute_depths() imported from cascade2vec.phase04_05_graph.depth.
Do NOT write a second BFS/traversal here — this is the single highest-risk
duplication in the codebase because it could silently re-introduce the
max_depth=0.0 bug class at per-snapshot granularity.

Numerical stability conventions (all documented in feature_dictionary.md)
--------------------------------------------------------------------------
- std_interarrival:  n_intervals < 2  → 0.0  (logged at INFO)
- burstiness:        n_intervals < 2  → 0.0  (logged at INFO)
- branching_factor:  no internal nodes → 0.0
  (Convention: a tree of 0 or 1 nodes has no branching; consistent with
   treating the root-only case as a star with 0 branches rather than undef.)
- depth_velocity, breadth_velocity, branching_velocity:
  t - delta_t < 0 → 0.0  (no prior snapshot available)
"""

import logging
import math
from typing import Optional

import pandas as pd
from graphframes import GraphFrame
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from cascade2vec.phase02_ingestion.leakage_audit import assert_snapshot_is_clean
from cascade2vec.phase04_05_graph.depth import compute_depths
from cascade2vec.phase04_05_graph.snapshots import get_snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reachable_from_root(
    depth_pd: pd.DataFrame,
    cascade_root_id: str,
    edges_pd: pd.DataFrame,
) -> "pd.Series":
    """
    Return a boolean mask of rows in depth_pd reachable from cascade_root_id.

    compute_depths() BFS seeds from ALL null-parent nodes (including orphaned
    sub-tree roots in disconnected cascades). This function does a single-root
    pandas BFS from cascade_root_id using the snapshot's edge list to correctly
    determine which nodes are reachable from the actual cascade root.

    NOTE: This is NOT a second implementation of the main depth computation.
    The main distributed BFS (compute_depths) runs on the full PySpark graph.
    This is a per-cascade pandas BFS on a small already-collected DataFrame,
    used only to determine which nodes belong to the cascade-root component
    for reachable_ratio and is_connected. It runs purely in memory.
    """
    import collections
    # Build adjacency: src -> list of dst
    adj: dict[str, list[str]] = {}
    for _, row in edges_pd.iterrows():
        adj.setdefault(row["src"], []).append(row["dst"])

    # BFS from cascade root
    visited: set[str] = set()
    queue: collections.deque = collections.deque([cascade_root_id])
    visited.add(cascade_root_id)
    while queue:
        node = queue.popleft()
        for child in adj.get(node, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)

    return depth_pd["tweet_id"].isin(visited)


def _depth_df_to_pandas(depth_df: DataFrame) -> pd.DataFrame:
    """Collect a PySpark depth DataFrame to pandas for per-cascade math."""
    return depth_df.select("tweet_id", "cascade_id", "depth", "reachable").toPandas()


def _compute_snapshot_structural(
    snap: GraphFrame,
    t_seconds: float,
    cascade_id: str,
) -> dict:
    """
    Compute Phase 6A structural features from a validated temporal snapshot.

    The snapshot MUST already have been validated with assert_snapshot_is_clean().

    Parameters
    ----------
    snap : GraphFrame
        Temporally filtered graph (only nodes/edges at or before t_seconds).
    t_seconds : float
        Observation cutoff in seconds. Used as denominator guard.
    cascade_id : str
        Identifier of the cascade being processed.

    Returns
    -------
    dict with keys: node_count, edge_count, max_depth, avg_depth, leaf_count,
                    leaf_ratio, branching_factor, root_degree, reachable_ratio,
                    is_connected
    """
    vertices = snap.vertices
    edges = snap.edges

    node_count = vertices.count()
    edge_count = edges.count()

    # --- Depth features via validated BFS (MUST use compute_depths, not reimplemented) ---
    depth_df = compute_depths(vertices, edges)
    depth_pd = _depth_df_to_pandas(depth_df)

    # Reachability is anchored to the CASCADE root only (the node with parent_id IS NULL
    # and the minimum timestamp). Orphaned sub-tree roots in disconnected cascades should
    # NOT count as reachable — they are unreachable from the actual cascade root.
    # This ensures is_connected=False for genuinely fractured cascades.
    root_rows = vertices.filter(F.col("parent_id").isNull()).orderBy("timestamp").select("id").collect()
    edges_pd = edges.select("src", "dst").toPandas()
    if root_rows:
        cascade_root_id = root_rows[0]["id"]
        reachable_mask = _reachable_from_root(depth_pd, cascade_root_id, edges_pd)
    else:
        reachable_mask = depth_pd["reachable"]  # fallback: no root found
    reachable_count = reachable_mask.sum()
    reachable_depths = depth_pd.loc[reachable_mask, "depth"]

    max_depth = float(reachable_depths.max()) if reachable_count > 0 else 0.0
    avg_depth = float(reachable_depths.mean()) if reachable_count > 0 else 0.0

    # reachable_ratio: Denominator is total nodes in THIS snapshot, NOT final
    # cascade size, to avoid leakage (final cascade size is future information).
    reachable_ratio = float(reachable_count) / node_count if node_count > 0 else 0.0

    is_connected = bool(reachable_count == node_count)

    # --- Leaf count / ratio ---
    # A leaf is a node that is never used as a src in the snapshot's edge set.
    if edge_count == 0:
        # All nodes are leaves if there are no edges (singletons)
        leaf_count = node_count
    else:
        src_ids = edges.select(F.col("src").alias("id")).distinct()
        leaf_df = vertices.select("id").join(src_ids, on="id", how="left_anti")
        leaf_count = leaf_df.count()

    leaf_ratio = float(leaf_count) / node_count if node_count > 0 else 0.0

    # --- Branching factor ---
    # Convention: average out-degree of all non-leaf (internal) nodes.
    # If there are no internal nodes (singleton or all-leaf), branching_factor = 0.0.
    # See feature_dictionary.md for the full stability convention.
    internal_node_count = node_count - leaf_count
    if internal_node_count > 0:
        branching_factor = float(edge_count) / float(internal_node_count)
    else:
        branching_factor = 0.0

    # --- Root degree (out-degree of the root node in the snapshot) ---
    root_rows = vertices.filter(F.col("parent_id").isNull()).select("id").collect()
    if root_rows:
        root_id = root_rows[0]["id"]
        root_degree = edges.filter(F.col("src") == root_id).count()
    else:
        root_degree = 0

    return {
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "max_depth": max_depth,
        "avg_depth": avg_depth,
        "leaf_count": int(leaf_count),
        "leaf_ratio": leaf_ratio,
        "branching_factor": branching_factor,
        "root_degree": int(root_degree),
        "reachable_ratio": reachable_ratio,
        "is_connected": is_connected,
    }


def _compute_snapshot_temporal(
    vertices: DataFrame,
    t_seconds: float,
) -> dict:
    """
    Compute Phase 6A temporal features from snapshot vertices.

    Parameters
    ----------
    vertices : DataFrame
        Snapshot vertices (already filtered to <= t_seconds).
    t_seconds : float
        Observation cutoff in seconds from cascade root.

    Returns
    -------
    dict with keys: tweets_per_minute, growth_velocity, mean_interarrival,
                    std_interarrival, burstiness, cascade_age
    """
    ts_pd = vertices.select("timestamp").toPandas()["timestamp"].dropna().sort_values().tolist()
    n = len(ts_pd)

    # cascade_age: elapsed time from root (t=0) to latest observed tweet,
    # capped at t_seconds
    cascade_age = float(ts_pd[-1]) if n > 0 else 0.0

    # tweets_per_minute: node_count / elapsed_minutes
    elapsed_minutes = cascade_age / 60.0
    tweets_per_minute = float(n) / elapsed_minutes if elapsed_minutes > 0 else 0.0

    # growth_velocity: same as tweets_per_minute in seconds
    growth_velocity = float(n) / cascade_age if cascade_age > 0 else 0.0

    # interarrival times: differences between consecutive tweet timestamps
    if n >= 2:
        intervals = [ts_pd[i + 1] - ts_pd[i] for i in range(n - 1)]
        mean_interarrival = float(sum(intervals)) / len(intervals)
        if len(intervals) >= 2:
            variance = sum((x - mean_interarrival) ** 2 for x in intervals) / len(intervals)
            std_interarrival = math.sqrt(variance)
            # Burstiness: (std - mean) / (std + mean), range [-1, 1]
            denom = std_interarrival + mean_interarrival
            burstiness = (std_interarrival - mean_interarrival) / denom if denom > 0 else 0.0
        else:
            # Only 1 interval — std and burstiness are undefined; convention = 0.0
            logger.info(
                "Insufficient temporal observations (n_intervals=1) for "
                "std_interarrival/burstiness; setting to 0.0"
            )
            std_interarrival = 0.0
            burstiness = 0.0
    else:
        # Insufficient observations — numerical stability convention
        logger.info(
            "Insufficient temporal observations (n=%d intervals) for "
            "std_interarrival/burstiness; setting to 0.0", n - 1 if n > 0 else 0
        )
        mean_interarrival = 0.0
        std_interarrival = 0.0
        burstiness = 0.0

    return {
        "tweets_per_minute": tweets_per_minute,
        "growth_velocity": growth_velocity,
        "mean_interarrival": mean_interarrival,
        "std_interarrival": std_interarrival,
        "burstiness": burstiness,
        "cascade_age": cascade_age,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_features(
    subgraph: GraphFrame,
    t_minutes: float,
    cascade_id: str,
    prev_features: Optional[dict] = None,
    delta_t_minutes: float = 5.0,
) -> dict:
    """
    Compute all Phase 6A and 6B features for a single (cascade_id, t) pair.

    Parameters
    ----------
    subgraph : GraphFrame
        Full cascade subgraph (from get_cascade_subgraph). Must NOT be
        pre-filtered — filtering to t is done internally.
    t_minutes : float
        Observation time in minutes from cascade root.
    cascade_id : str
        Cascade identifier, forwarded to output for join convenience.
    prev_features : dict, optional
        Feature dict from t - delta_t_minutes snapshot.
        If None, velocity features are set to 0.0.
    delta_t_minutes : float
        Time delta for velocity feature computation (default 5 minutes).

    Returns
    -------
    dict
        All features for this (cascade_id, t) pair, including label=None
        (label is joined in build_feature_matrix.py, not here).
    """
    t_seconds = t_minutes * 60.0

    # 1. Build temporal snapshot
    snap = get_snapshot(subgraph, t_minutes)

    # 2. HARD LEAKAGE CHECK — raises AssertionError if any future data present
    assert_snapshot_is_clean(snap, t_seconds)

    # 3. Structural features
    structural = _compute_snapshot_structural(snap, t_seconds, cascade_id)

    # 4. Temporal features
    temporal = _compute_snapshot_temporal(snap.vertices, t_seconds)

    # 5. Phase 6B: velocity features
    # Formula: (feature(t) - feature(t - delta_t)) / delta_t
    # If no prior snapshot (t <= delta_t), velocity = 0.0
    if prev_features is not None and t_minutes > delta_t_minutes:
        delta_t_sec = delta_t_minutes * 60.0
        depth_velocity = (
            structural["max_depth"] - prev_features.get("max_depth", 0.0)
        ) / delta_t_sec
        breadth_velocity = (
            structural["node_count"] - prev_features.get("node_count", 0)
        ) / delta_t_sec
        branching_velocity = (
            structural["branching_factor"] - prev_features.get("branching_factor", 0.0)
        ) / delta_t_sec
    else:
        depth_velocity = 0.0
        breadth_velocity = 0.0
        branching_velocity = 0.0

    return {
        "cascade_id": cascade_id,
        "t_minutes": t_minutes,
        # Phase 6A: structural
        **structural,
        # Phase 6A: temporal
        **temporal,
        # Phase 6B: velocity
        "depth_velocity": depth_velocity,
        "breadth_velocity": breadth_velocity,
        "branching_velocity": branching_velocity,
    }
