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

    Uses a Spark SQL recursive CTE to BFS from each cascade root, computing
    depth for every reachable node. Unreachable nodes get NULL depth and their
    cascade is flagged is_connected=False.

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
    spark = graph.vertices.sparkSession

    # Raise recursion limit above our known max depth (~231 + safety margin)
    spark.conf.set("spark.sql.recursion.limit", "500")

    vertices = graph.vertices
    edges = graph.edges

    # Register as temp views for SQL
    vertices.createOrReplaceTempView("_stats_vertices")
    edges.createOrReplaceTempView("_stats_edges")

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

    # --- BFS depth via Spark SQL recursive CTE ---
    # Edges: src=parent → dst=reply (root→leaf direction), so walk src→dst.
    # Base case: root nodes (parent_id IS NULL) get depth=0.
    # Recursive step: children get parent_depth + 1.
    logger.info("[stats] Computing BFS depths via recursive SQL CTE...")

    depths_df = spark.sql("""
        WITH RECURSIVE bfs (id, cascade_id, depth) AS (
            -- Base case: roots (no parent)
            SELECT id, cascade_id, 0 AS depth
            FROM _stats_vertices
            WHERE parent_id IS NULL

            UNION ALL

            -- Recursive: expand one level from current frontier
            SELECT e.dst AS id,
                   e.cascade_id,
                   b.depth + 1 AS depth
            FROM bfs b
            INNER JOIN _stats_edges e
                ON b.id = e.src
                AND b.cascade_id = e.cascade_id
        )
        SELECT id, cascade_id, MIN(depth) AS depth_from_root
        FROM bfs
        GROUP BY id, cascade_id
    """)

    # Left-join all vertices to depths — unreachable nodes get NULL depth
    all_depths = (
        vertices.select("id", "cascade_id")
        .join(depths_df, on=["id", "cascade_id"], how="left")
    )

    # Max depth per cascade; count NULLs to flag disconnected cascades
    max_depths = (
        all_depths
        .groupBy("cascade_id")
        .agg(
            F.max("depth_from_root").alias("max_depth"),
            F.sum(
                F.when(F.col("depth_from_root").isNull(), 1).otherwise(0)
            ).alias("unreachable_count")
        )
    )

    # Log disconnected cascades
    n_unreachable = max_depths.filter(F.col("unreachable_count") > 0).count()
    if n_unreachable > 0:
        logger.warning(
            "[stats] %d cascade(s) have unreachable nodes "
            "(orphaned subtrees — is_connected=False, depth=NULL for those nodes).",
            n_unreachable,
        )

    # --- Assemble final stats DataFrame ---
    stats = (
        node_counts
        .join(edge_counts, on="cascade_id", how="left")
        .join(
            max_depths.select("cascade_id", "max_depth", "unreachable_count"),
            on="cascade_id", how="left"
        )
        .fillna({"edge_count": 0, "max_depth": 0, "unreachable_count": 0})
        .withColumn("is_singleton", F.col("edge_count") == 0)
        .withColumn("is_connected", F.col("unreachable_count") == 0)
        .drop("unreachable_count")
        .orderBy("cascade_id")
    )

    logger.info("[stats] graph_summary_stats complete for %d cascades.", stats.count())
    return stats
