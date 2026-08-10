"""
engineering_pandas.py — Phase 6A/6B Feature Engineering (Pandas Port)
=====================================================================
Computes structural and temporal features from temporal cascade snapshots
using in-memory Pandas dataframes.
"""

import logging
import math
from typing import Optional
import pandas as pd

from cascade2vec.phase02_ingestion.leakage_audit import assert_snapshot_is_clean
from cascade2vec.phase04_05_graph.depth_pandas import compute_depths_pandas

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
    Uses in-memory Pandas BFS from the actual cascade root.
    """
    import collections
    adj: dict[str, list[str]] = {}
    for _, row in edges_pd.iterrows():
        adj.setdefault(row["src"], []).append(row["dst"])

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


def _compute_snapshot_structural(
    snap: dict,
    t_seconds: float,
    cascade_id: str,
) -> dict:
    """
    Compute Phase 6A structural features from a validated temporal snapshot.
    """
    vertices = snap["vertices"]
    edges = snap["edges"]

    node_count = len(vertices)
    edge_count = len(edges)

    # --- Depth features via validated BFS ---
    depth_pd = compute_depths_pandas(vertices, edges)

    # Find the actual cascade root (parent_id is null, min timestamp)
    root_rows = vertices[vertices["parent_id"].isna()].sort_values("timestamp")
    if not root_rows.empty:
        cascade_root_id = root_rows.iloc[0]["id"]
        reachable_mask = _reachable_from_root(depth_pd, cascade_root_id, edges)
    else:
        reachable_mask = depth_pd["reachable"]
        
    reachable_count = int(reachable_mask.sum())
    reachable_depths = depth_pd.loc[reachable_mask, "depth"]

    max_depth = float(reachable_depths.max()) if reachable_count > 0 else 0.0
    avg_depth = float(reachable_depths.mean()) if reachable_count > 0 else 0.0

    reachable_ratio = float(reachable_count) / node_count if node_count > 0 else 0.0
    is_connected = bool(reachable_count == node_count)

    # --- Leaf count / ratio ---
    if edge_count == 0:
        leaf_count = node_count
    else:
        src_ids = set(edges["src"].unique())
        leaf_count = len(vertices[~vertices["id"].isin(src_ids)])

    leaf_ratio = float(leaf_count) / node_count if node_count > 0 else 0.0

    # --- Branching factor ---
    internal_node_count = node_count - leaf_count
    if internal_node_count > 0:
        branching_factor = float(edge_count) / float(internal_node_count)
    else:
        branching_factor = 0.0

    # --- Root degree ---
    if not root_rows.empty:
        root_id = root_rows.iloc[0]["id"]
        root_degree = len(edges[edges["src"] == root_id])
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
    vertices: pd.DataFrame,
    t_seconds: float,
) -> dict:
    """
    Compute Phase 6A temporal features from snapshot vertices.
    """
    ts_pd = vertices["timestamp"].dropna().sort_values().tolist()
    n = len(ts_pd)

    cascade_age = float(ts_pd[-1]) if n > 0 else 0.0

    elapsed_minutes = cascade_age / 60.0
    tweets_per_minute = float(n) / elapsed_minutes if elapsed_minutes > 0 else 0.0

    growth_velocity = float(n) / cascade_age if cascade_age > 0 else 0.0

    if n >= 2:
        intervals = [ts_pd[i + 1] - ts_pd[i] for i in range(n - 1)]
        mean_interarrival = float(sum(intervals)) / len(intervals)
        if len(intervals) >= 2:
            variance = sum((x - mean_interarrival) ** 2 for x in intervals) / len(intervals)
            std_interarrival = math.sqrt(variance)
            denom = std_interarrival + mean_interarrival
            burstiness = (std_interarrival - mean_interarrival) / denom if denom > 0 else 0.0
        else:
            logger.info("Insufficient temporal observations (n_intervals=1) for std_interarrival/burstiness; setting to 0.0")
            std_interarrival = 0.0
            burstiness = 0.0
    else:
        logger.info(f"Insufficient temporal observations (n={n - 1 if n > 0 else 0} intervals) for std_interarrival/burstiness; setting to 0.0")
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

def compute_features_pandas(
    vertices: pd.DataFrame,
    edges: pd.DataFrame,
    t_minutes: float,
    cascade_id: str,
    prev_features: Optional[dict] = None,
    delta_t_minutes: float = 5.0,
) -> dict:
    """
    Compute all Phase 6A and 6B features for a single (cascade_id, t) pair using Pandas.
    """
    t_seconds = t_minutes * 60.0

    # 1. Build temporal snapshot (filter to t_seconds)
    # The PySpark `get_snapshot` did this exactly:
    v_snap = vertices[vertices["timestamp"] <= t_seconds].copy()
    e_snap = edges.copy()
    # Edges should only be kept if their src and dst are in the vertices subset. 
    # But since all children arrive after their parents, if a node is before t, 
    # its parent must be before t. So we can just join on dst to filter edges.
    e_snap = e_snap[e_snap["dst"].isin(v_snap["id"])]
    
    snap = {"vertices": v_snap, "edges": e_snap}

    # 2. HARD LEAKAGE CHECK
    assert_snapshot_is_clean(snap, t_seconds)

    # 3. Structural features
    structural = _compute_snapshot_structural(snap, t_seconds, cascade_id)

    # 4. Temporal features
    temporal = _compute_snapshot_temporal(v_snap, t_seconds)

    # 5. Phase 6B: velocity features
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
        **structural,
        **temporal,
        "depth_velocity": depth_velocity,
        "breadth_velocity": breadth_velocity,
        "branching_velocity": branching_velocity,
    }
