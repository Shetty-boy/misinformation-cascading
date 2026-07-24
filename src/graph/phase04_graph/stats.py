"""
stats.py — Phase 4 Graph Construction
======================================
Per-cascade graph summary statistics using GraphFrames built-in algorithms.

Public API:
    graph_summary_stats(graph) -> Spark DataFrame

Output columns:
    cascade_id   (str)  — cascade identifier
    node_count   (int)  — number of vertices in this cascade
    edge_count   (int)  — number of edges (replies) in this cascade
    is_singleton (bool) — True if edge_count == 0
    is_connected (bool) — True if all nodes reachable from root
    max_depth    (int)  — longest path from root; -1 if unreachable nodes exist
                          (indicates a broken tree / orphan remnant)
"""

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from graphframes import GraphFrame

logger = logging.getLogger(__name__)


def graph_summary_stats(graph: GraphFrame) -> DataFrame:
    """
    Compute per-cascade summary statistics for the full GraphFrame.

    Uses GraphFrames' connectedComponents for connectivity and
    shortestPaths (from each cascade root) for max depth.

    Parameters
    ----------
    graph : GraphFrame
        The full combined GraphFrame from build_full_graph().

    Returns
    -------
    Spark DataFrame with columns:
        cascade_id, node_count, edge_count, is_singleton,
        is_connected, max_depth
    """
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

    # --- Connected components ---
    # connectedComponents assigns each node a component_id.
    # A cascade is "connected" if all its nodes share one component_id.
    logger.info("[stats] Running connectedComponents...")
    cc = graph.connectedComponents()

    # Count distinct components per cascade
    components_per_cascade = (
        cc
        .groupBy("cascade_id")
        .agg(F.countDistinct("component").alias("num_components"))
    )

    # --- Max depth via shortestPaths from cascade roots ---
    # Roots are vertices with no incoming edges (parent_id IS NULL)
    roots = (
        vertices
        .filter(F.col("parent_id").isNull())
        .select(F.col("id").alias("root_id"), "cascade_id")
    )

    logger.info("[stats] Running shortestPaths from %d roots...", roots.count())

    # Collect root ids for shortestPaths landmarks
    root_ids = [row["root_id"] for row in roots.select("root_id").collect()]

    sp = graph.shortestPaths(landmarks=root_ids)

    # For each node, find the shortest path distance to its own cascade root
    roots_map = {row["root_id"]: row["cascade_id"] for row in roots.collect()}

    # Extract the distance to own-cascade root from the distances map column
    # distances is a MapType(StringType, IntegerType)
    # We pick the distance to the root that shares this node's cascade_id
    @F.udf("int")
    def extract_own_root_distance(cascade_id, distances):
        if distances is None:
            return -1
        # Find the root that belongs to this cascade
        for root_id, cid in roots_map.items():
            if cid == cascade_id and root_id in distances:
                return int(distances[root_id])
        return -1

    sp_with_depth = sp.withColumn(
        "depth_from_root",
        extract_own_root_distance(F.col("cascade_id"), F.col("distances")),
    )

    # Max depth per cascade (if any node has depth -1, cascade may have a cycle)
    max_depths = (
        sp_with_depth
        .groupBy("cascade_id")
        .agg(
            F.max("depth_from_root").alias("max_depth"),
            F.min("depth_from_root").alias("min_depth"),
        )
    )

    # Log cascades with unreachable nodes (min_depth == -1 indicates broken paths)
    unreachable = max_depths.filter(F.col("min_depth") == -1)
    n_unreachable = unreachable.count()
    if n_unreachable > 0:
        logger.warning(
            "[stats] %d cascade(s) have nodes unreachable from root "
            "(possible non-trivial cycles or broken tree structure). "
            "max_depth is set to -1 for these cascades.",
            n_unreachable,
        )

    # --- Assemble final stats DataFrame ---
    stats = (
        node_counts
        .join(edge_counts, on="cascade_id", how="left")
        .join(components_per_cascade, on="cascade_id", how="left")
        .join(max_depths.select("cascade_id", "max_depth"), on="cascade_id", how="left")
        .fillna({"edge_count": 0, "num_components": 1, "max_depth": 0})
        .withColumn("is_singleton", F.col("edge_count") == 0)
        .withColumn("is_connected", F.col("num_components") == 1)
        .drop("num_components")
        .orderBy("cascade_id")
    )

    logger.info("[stats] graph_summary_stats complete for %d cascades.", stats.count())
    return stats
