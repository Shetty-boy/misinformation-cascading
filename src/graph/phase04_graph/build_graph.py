"""
build_graph.py — Phase 4 Graph Construction
============================================
Core module for building GraphFrames from the unified CASCADE2VEC dataset.

Public API (fixed contract — teammate's snapshot code depends on these):
    to_vertices(df)                          -> Spark DataFrame
    to_edges(df)                             -> Spark DataFrame
    build_full_graph(vertices, edges)        -> GraphFrame
    get_cascade_subgraph(graph, cascade_id) -> GraphFrame
    flag_singletons(vertices, edges)         -> Spark DataFrame

See contract.md for the exact output spec of get_cascade_subgraph.

Edge-case policy (all cases log at WARNING level and continue):
    - Orphaned edges:  parent_id not in the same cascade's vertex set → dropped.
    - Duplicate edges: (src, dst) pairs that appear more than once → dropped.
    - Singleton cascades: single node, zero edges → kept in graph, flagged separately.
    - Cycles / self-loops: logged as WARNING. Full cycle detection is too expensive
      on Spark; self-loops (tweet_id == parent_id) are detected and dropped.
      Non-trivial cycles in the reply tree are flagged via a log warning per cascade
      when max_depth == -1 from shortestPaths (unreachable nodes).
"""

import logging
import os
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from graphframes import GraphFrame

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = "data/processed/04_graph"


# ---------------------------------------------------------------------------
# 1. Vertex construction
# ---------------------------------------------------------------------------

def to_vertices(df: DataFrame) -> DataFrame:
    """
    Build the vertex DataFrame from the unified dataset.

    GraphFrames requires a column named exactly ``id``.
    All original columns (timestamp, label, text, etc.) are preserved
    as vertex attributes.

    Parameters
    ----------
    df : DataFrame
        The unified dataset loaded by loader.load_unified().

    Returns
    -------
    DataFrame with columns:
        id (str), user_id, timestamp, text, parent_id,
        cascade_id, event_id, label
    """
    vertices = df.withColumnRenamed("tweet_id", "id")
    logger.info("[build_graph] to_vertices: %d vertices total.", vertices.count())
    return vertices


# ---------------------------------------------------------------------------
# 2. Edge construction
# ---------------------------------------------------------------------------

def to_edges(df: DataFrame, vertices: Optional[DataFrame] = None) -> DataFrame:
    """
    Build the edge DataFrame from the unified dataset.

    GraphFrames requires columns named exactly ``src`` and ``dst``.
    src = parent_id (the replying-to tweet)
    dst = tweet_id  (the reply itself)

    Processing steps:
    1. Drop root tweets (parent_id IS NULL) — they have no incoming edge.
    2. Drop self-loops (tweet_id == parent_id) — log as potential cycle.
    3. Deduplicate (src, dst) pairs.
    4. If ``vertices`` is provided, detect and drop orphaned edges
       (parent not in the same cascade's vertex set).

    Parameters
    ----------
    df : DataFrame
        The unified dataset loaded by loader.load_unified().
    vertices : DataFrame, optional
        If provided, used to validate that each src exists within
        the same cascade (orphan detection).

    Returns
    -------
    DataFrame with columns: src (str), dst (str), cascade_id (str)
        Only cascade_id is carried through; all vertex attributes live
        on the vertex DataFrame.
    """
    # Step 1: Drop roots (no parent)
    edges = df.filter(F.col("parent_id").isNotNull())
    n_after_root_drop = edges.count()
    logger.info("[build_graph] to_edges: %d rows after dropping roots.", n_after_root_drop)

    # Step 2: Detect and drop self-loops
    self_loops = edges.filter(F.col("tweet_id") == F.col("parent_id"))
    n_self_loops = self_loops.count()
    if n_self_loops > 0:
        logger.warning(
            "[build_graph] Detected %d self-loop(s) (tweet_id == parent_id). "
            "These indicate potential cycles and are dropped.",
            n_self_loops,
        )
    edges = edges.filter(F.col("tweet_id") != F.col("parent_id"))

    # Step 3: Rename to GraphFrames-required column names, keep cascade_id
    edges = (
        edges
        .withColumnRenamed("parent_id", "src")
        .withColumnRenamed("tweet_id", "dst")
        .select("src", "dst", "cascade_id")
    )

    # Step 4: Drop duplicate (src, dst) pairs
    n_before_dedup = edges.count()
    edges = edges.dropDuplicates(["src", "dst"])
    n_dupes = n_before_dedup - edges.count()
    if n_dupes > 0:
        logger.warning("[build_graph] Dropped %d duplicate edge(s).", n_dupes)

    # Step 5: Orphan detection (requires vertices)
    if vertices is not None:
        # Build a set of valid (id, cascade_id) pairs
        valid_parents = (
            vertices
            .select(F.col("id").alias("src"), F.col("cascade_id").alias("v_cascade_id"))
        )
        edges_checked = edges.join(
            valid_parents,
            on="src",
            how="left",
        )
        # Orphan = src exists but belongs to a different cascade, OR src not in vertices at all
        orphans = edges_checked.filter(
            F.col("v_cascade_id").isNull() |
            (F.col("v_cascade_id") != F.col("cascade_id"))
        )
        n_orphans = orphans.count()
        if n_orphans > 0:
            logger.warning(
                "[build_graph] Dropped %d orphaned edge(s) "
                "(parent_id not found in the same cascade's vertex set).",
                n_orphans,
            )
        edges = edges_checked.filter(
            F.col("v_cascade_id").isNotNull() &
            (F.col("v_cascade_id") == F.col("cascade_id"))
        ).select("src", "dst", "cascade_id")

    n_final = edges.count()
    logger.info("[build_graph] to_edges: %d clean edges.", n_final)
    return edges


# ---------------------------------------------------------------------------
# 3. Full graph construction
# ---------------------------------------------------------------------------

def build_full_graph(vertices: DataFrame, edges: DataFrame) -> GraphFrame:
    """
    Build a single GraphFrame containing all cascades together.

    Keeping one combined GraphFrame is cheaper than constructing one
    per cascade — per-cascade subgraphs are materialized on demand
    via get_cascade_subgraph().

    Parameters
    ----------
    vertices : DataFrame
        Output of to_vertices(). Must have column ``id``.
    edges : DataFrame
        Output of to_edges(). Must have columns ``src``, ``dst``.

    Returns
    -------
    GraphFrame
        Combined graph; vertex attributes include cascade_id,
        enabling later filtering per cascade.
    """
    graph = GraphFrame(vertices, edges)
    logger.info(
        "[build_graph] Built full GraphFrame: %d vertices, %d edges.",
        vertices.count(), edges.count(),
    )
    return graph


# ---------------------------------------------------------------------------
# 4. Per-cascade subgraph  ← FIXED CONTRACT (teammate depends on this)
# ---------------------------------------------------------------------------

def get_cascade_subgraph(graph: GraphFrame, cascade_id: str) -> GraphFrame:
    """
    Filter the full GraphFrame down to a single cascade.

    CONTRACT (fixed — do not change signature or output columns):
    ─────────────────────────────────────────────────────────────
    Input:
        graph       — The full GraphFrame built by build_full_graph().
        cascade_id  — The cascade_id string to isolate (same value
                      stored in the ``cascade_id`` column of vertices
                      and edges).

    Output:
        A new GraphFrame where:
        • vertices contain only rows where cascade_id == cascade_id arg.
        • edges contain only rows where both src and dst belong to the
          filtered vertex set (i.e. both endpoints are in this cascade).
        • ALL original vertex columns are preserved:
            id, user_id, timestamp, text, parent_id,
            cascade_id, event_id, label
        • Edge columns: src, dst, cascade_id

    Notes:
    • The returned GraphFrame is not cached — caller should .cache()
      if it will be reused across multiple operations.
    • Singleton cascades (no edges) return a GraphFrame with vertices
      but an empty edge DataFrame — this is valid and expected.
    • See contract.md for the full interface spec.

    Parameters
    ----------
    graph : GraphFrame
        The full combined GraphFrame from build_full_graph().
    cascade_id : str
        The cascade identifier to extract.

    Returns
    -------
    GraphFrame
        Subgraph restricted to the given cascade.
    """
    sub_v = graph.vertices.filter(F.col("cascade_id") == cascade_id)
    sub_e = graph.edges.filter(F.col("cascade_id") == cascade_id)

    # Belt-and-suspenders: ensure no dangling edges after vertex filter
    valid_ids = sub_v.select(F.col("id")).rdd.flatMap(lambda r: [r[0]]).collect()
    valid_ids_set = set(valid_ids)

    # For large cascades, broadcast join is more efficient than collect
    valid_ids_bc = graph.vertices.sparkSession.sparkContext.broadcast(valid_ids_set)

    @F.udf("boolean")
    def in_valid(node_id):
        return node_id in valid_ids_bc.value

    sub_e = sub_e.filter(in_valid(F.col("src")) & in_valid(F.col("dst")))

    return GraphFrame(sub_v, sub_e)


# ---------------------------------------------------------------------------
# 5. Singleton detection
# ---------------------------------------------------------------------------

def flag_singletons(vertices: DataFrame, edges: DataFrame) -> DataFrame:
    """
    Return a DataFrame of cascade_ids that have no edges (singletons).

    Singletons are cascades with exactly one post and zero replies.
    They are kept in the full graph but carry no structural information
    for graph learning — the model should exclude them from training
    or handle them with a fallback feature vector.

    Parameters
    ----------
    vertices : DataFrame
        Output of to_vertices().
    edges : DataFrame
        Output of to_edges().

    Returns
    -------
    DataFrame with columns: cascade_id (str), node_count (int)
        One row per singleton cascade.
    """
    # Cascades that appear in edges (have at least one reply)
    cascades_with_edges = edges.select("cascade_id").distinct()

    # All cascades from vertices
    all_cascades = (
        vertices
        .groupBy("cascade_id")
        .agg(F.count("id").alias("node_count"))
    )

    singletons = all_cascades.join(
        cascades_with_edges,
        on="cascade_id",
        how="left_anti",  # cascades NOT in edges
    )

    n = singletons.count()
    logger.info("[build_graph] Flagged %d singleton cascade(s).", n)
    return singletons


# ---------------------------------------------------------------------------
# 6. Main: build and persist graphs
# ---------------------------------------------------------------------------

def main():
    """
    End-to-end pipeline: load data → build graph → write vertices/edges
    → write singleton list → write summary stats.

    Outputs (all in data/processed/04_graph/):
        vertices.parquet
        edges.parquet
        singletons.parquet
    """
    import os
    from src.graph.phase04_graph.loader import get_spark, load_unified
    from src.graph.phase04_graph.stats import graph_summary_stats

    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    os.makedirs("experiments/logs/04_graph", exist_ok=True)

    spark = get_spark()
    df = load_unified(spark)

    vertices = to_vertices(df)
    edges = to_edges(df, vertices=vertices)
    graph = build_full_graph(vertices, edges)
    singletons = flag_singletons(vertices, edges)

    # Persist outputs
    vertices.write.mode("overwrite").parquet(f"{DEFAULT_OUT_DIR}/vertices.parquet")
    edges.write.mode("overwrite").parquet(f"{DEFAULT_OUT_DIR}/edges.parquet")
    singletons.write.mode("overwrite").parquet(f"{DEFAULT_OUT_DIR}/singletons.parquet")
    logger.info("[build_graph] Vertices/edges/singletons written to %s.", DEFAULT_OUT_DIR)

    # Generate and write summary stats
    stats_df = graph_summary_stats(graph)
    stats_df.write.mode("overwrite").parquet(f"{DEFAULT_OUT_DIR}/graph_stats.parquet")

    # Write markdown report
    stats_pd = stats_df.toPandas()
    report_path = "experiments/logs/04_graph/graph_stats.md"
    with open(report_path, "w") as f:
        f.write("# CASCADE2VEC — Phase 4 Graph Stats\n\n")
        f.write(f"_Generated by build_graph.main()_\n\n")
        f.write("## Summary\n")
        f.write(f"- Total cascades: {len(stats_pd)}\n")
        f.write(f"- Singleton cascades: {(stats_pd['edge_count'] == 0).sum()}\n")
        f.write(f"- Fully connected cascades: {stats_pd['is_connected'].sum()}\n")
        f.write(f"- Mean node count: {stats_pd['node_count'].mean():.1f}\n")
        f.write(f"- Mean edge count: {stats_pd['edge_count'].mean():.1f}\n")
        f.write(f"- Mean max depth: {stats_pd[stats_pd['max_depth'] >= 0]['max_depth'].mean():.1f}\n\n")
        f.write("## Per-Cascade Stats (first 20)\n\n")
        f.write(stats_pd.head(20).to_markdown(index=False))
        f.write("\n")

    logger.info("[build_graph] Stats report written to %s.", report_path)
    print(f"[build_graph] Done. Stats written to {report_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
