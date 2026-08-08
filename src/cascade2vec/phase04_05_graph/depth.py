"""
depth.py — Distributed Graph Traversal Engine
=============================================
Iterative frontier expansion (BFS) to compute graph depth.

# DAG GUARANTEE
# This BFS does NOT need to track visited nodes (no left_anti join or deduplication).
# That is safe for two combined reasons:
#
#   1. TEMPORAL ORDERING: Twitter's reply semantics guarantee that a reply is always
#      chronologically *after* its parent tweet. No edge can point backwards in time,
#      so no cycle (which would require traversing backwards against time) is possible.
#
#   2. IN-DEGREE <= 1: Each tweet has at most one parent_id, so each node has at
#      most one incoming edge. Together with (1), this proves the graph is a strict
#      forest (a set of disjoint trees).
#
# NOTE: In-degree <= 1 alone is INSUFFICIENT. A 2-node cycle A -> B -> A has
# in-degree 1 at both A and B. Temporal ordering is the argument that breaks cycles.
"""

import logging
from functools import reduce
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 500

def compute_depths(vertices_df: DataFrame, edges_df: DataFrame) -> DataFrame:
    """
    Computes the depth of each node from the root(s) in a forest of independent cascades.
    Uses iterative frontier expansion (BFS) with distributed DataFrame joins.
    
    Returns:
        DataFrame: [tweet_id, cascade_id, depth, reachable]
    """
    # 1. Find Roots -> Depth = 0 -> Frontier
    roots = vertices_df.filter(F.col("parent_id").isNull()).select(
        F.col("id").alias("tweet_id"),
        F.col("cascade_id"),
        F.lit(0).alias("depth")
    )
    
    frontier = roots
    depths_list = [roots]
    
    iteration = 0
    frontier_count = frontier.count()
    
    logger.info("------------------")
    logger.info(f"Iteration {iteration}")
    logger.info(f"Frontier = {frontier_count} nodes")
    
    edges_sub = edges_df.select("src", "dst", "cascade_id")
    
    while frontier_count > 0:
        iteration += 1
        
        if iteration > MAX_ITERATIONS:
            raise RuntimeError(f"Graph contains cycle or exceeded max depth of {MAX_ITERATIONS}")
            
        # Join -> Children
        children = frontier.join(
            edges_sub,
            (frontier.tweet_id == edges_sub.src) & (frontier.cascade_id == edges_sub.cascade_id),
            "inner"
        ).select(
            F.col("dst").alias("tweet_id"),
            frontier.cascade_id,
            (F.col("depth") + 1).alias("depth")
        )
        
        # Temporal ordering + in-degree <= 1 => forest, no cycles possible.
        # No deduplication or visited-set tracking needed (see module docstring).
        # Use persist() instead of localCheckpoint() to break query plan lineage
        # without triggering the SIGTERM signal-handler re-entry that corrupts
        # the Py4J connection in test environments (localCheckpoint calls
        # sc.cancelAllJobs() on teardown, which is a reentrant JVM call).
        new_frontier = children.persist()
        frontier_count = new_frontier.count()
        
        if frontier_count == 0:
            break
            
        logger.info("------------------")
        logger.info(f"Iteration {iteration}")
        logger.info(f"Frontier = {frontier_count}")
        
        depths_list.append(new_frontier)
        frontier = new_frontier

    all_depths = reduce(DataFrame.unionAll, depths_list)

    # Join Back -> NULL -> unreachable
    result = vertices_df.select(
        F.col("id").alias("tweet_id"),
        F.col("cascade_id")
    ).join(
        all_depths,
        on=["tweet_id", "cascade_id"],
        how="left"
    ).withColumn(
        "reachable",
        F.col("depth").isNotNull()
    )
    
    return result
