"""
stats.py — Phase 4 Graph Construction
======================================
Per-cascade graph summary statistics using Spark SQL recursive CTE (BFS).

Public API:
    graph_summary_stats(graph) -> Spark DataFrame

Output columns:
    cascade_id   (str)  — cascade identifier
    node_count   (int)  — number of vertices in this cascade
    edge_count   (int)  — number of edges (replies) in this cascade
    is_singleton (bool) — True if edge_count == 0
    is_connected (bool) — True if all nodes were reached during BFS expansion
    max_depth    (int)  — longest path from root. Disconnected nodes get NULL
                          depth; is_connected=False for those cascades.

Implementation:
    Uses Spark SQL WITH RECURSIVE CTE (Spark 3.5+) for depth computation.
    This is the correct scalable approach — no Python driver loop, no lineage
    blowup, no manual checkpointing. Spark parallelises across cascades natively.

Edge direction:
    Edges are stored as src=parent_id → dst=reply_id (root → leaf direction).
    BFS walks src→dst, i.e. from root downward — correct.
"""

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from graphframes import GraphFrame

logger = logging.getLogger(__name__)


def graph_summary_stats(graph: GraphFrame) -> DataFrame:
    """
    Compute per-cascade summary statistics for the full GraphFrame.
    Delegates depth computation to depth.py.
    """
    from cascade2vec.phase04_05_graph.depth import compute_depths

    vertices = graph.vertices
    edges = graph.edges

    # --- Node count per cascade ---
    node_counts = (
        vertices
        .groupBy("cascade_id")
        .agg(F.count("id").alias("node_count"))
    )

    # --- Edge count per cascade ---
    edge_counts = (
        edges
        .groupBy("cascade_id")
        .agg(F.count("src").alias("edge_count"))
    )

    # --- Depth computation ---
    logger.info("[stats] Computing depths using distributed BFS...")
    all_depths = compute_depths(vertices, edges)

    # Max depth, mean depth, reachable_ratio per cascade
    # Count NULLs (unreachable) vs total
    depth_stats = (
        all_depths
        .groupBy("cascade_id")
        .agg(
            F.max("depth").alias("max_depth"),
            F.mean("depth").alias("mean_depth"),
            F.sum(F.when(F.col("reachable"), 1).otherwise(0)).alias("reachable_count"),
            F.count("*").alias("total_nodes")
        )
        .withColumn("reachable_ratio", F.col("reachable_count") / F.col("total_nodes"))
        .withColumn("is_connected", F.col("reachable_count") == F.col("total_nodes"))
    )

    # Log disconnected cascades
    n_unreachable = depth_stats.filter(F.col("is_connected") == False).count()
    if n_unreachable > 0:
        logger.warning(
            "[stats] %d cascade(s) have unreachable nodes "
            "(orphaned subtrees — is_connected=False).",
            n_unreachable,
        )

    # --- Assemble final stats DataFrame ---
    stats = (
        node_counts
        .join(edge_counts, on="cascade_id", how="left")
        .join(
            depth_stats.select("cascade_id", "max_depth", "mean_depth", "reachable_ratio", "is_connected"),
            on="cascade_id", how="left"
        )
        .fillna({"edge_count": 0, "max_depth": 0, "mean_depth": 0.0, "reachable_ratio": 0.0})
        .withColumn("is_singleton", F.col("edge_count") == 0)
        .orderBy("cascade_id")
    )

    logger.info("[stats] graph_summary_stats complete for %d cascades.", stats.count())
    return stats
