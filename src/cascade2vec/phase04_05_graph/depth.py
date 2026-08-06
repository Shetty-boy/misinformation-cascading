"""
depth.py — Distributed Graph Traversal Engine
=============================================
Iterative frontier expansion (BFS) to compute graph depth.
"""

import logging
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
    all_depths = roots
    
    frontier.cache()
    all_depths.cache()
    
    iteration = 0
    visited_count = all_depths.count()
    frontier_count = frontier.count()
    
    logger.info("------------------")
    logger.info(f"Iteration {iteration}")
    logger.info(f"Frontier = {frontier_count} nodes")
    logger.info(f"Visited = {visited_count}")
    
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
        
        # Deduplicate children
        children = children.groupBy("tweet_id", "cascade_id").agg(F.min("depth").alias("depth"))
        
        # left_anti -> New Frontier
        new_frontier = children.join(
            all_depths,
            on=["tweet_id", "cascade_id"],
            how="left_anti"
        ).localCheckpoint()
        
        frontier_count = new_frontier.count()
        
        if frontier_count == 0:
            break
            
        # Append Depths
        new_all_depths = all_depths.unionByName(new_frontier).localCheckpoint()
        visited_count = new_all_depths.count()
        
        logger.info("------------------")
        logger.info(f"Iteration {iteration}")
        logger.info(f"Frontier = {frontier_count}")
        logger.info(f"Visited = {visited_count}")
        
        frontier = new_frontier
        all_depths = new_all_depths

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
