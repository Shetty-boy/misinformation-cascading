"""
depth_pandas.py — In-memory Graph Traversal Engine
==================================================
Iterative frontier expansion (BFS) to compute graph depth using Pandas.

This is a pure-Pandas port of the PySpark logic in depth.py, designed to match
its output semantics exactly (including handling of unreachable nodes).
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 500

def compute_depths_pandas(vertices_pd: pd.DataFrame, edges_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the depth of each node from the root(s) in a forest of independent cascades.
    Uses iterative frontier expansion (BFS) with Pandas joins.
    
    Returns:
        pd.DataFrame: [tweet_id, cascade_id, depth, reachable]
    """
    # 1. Find Roots -> Depth = 0 -> Frontier
    # A root is any node with parent_id == None / NaN / Null
    roots_mask = vertices_pd['parent_id'].isna()
    roots = vertices_pd.loc[roots_mask, ['id', 'cascade_id']].copy()
    roots.rename(columns={'id': 'tweet_id'}, inplace=True)
    roots['depth'] = 0.0

    frontier = roots
    depths_list = [roots]
    
    iteration = 0
    frontier_count = len(frontier)
    
    edges_sub = edges_pd[['src', 'dst', 'cascade_id']].copy()
    
    while frontier_count > 0:
        iteration += 1
        
        if iteration > MAX_ITERATIONS:
            raise RuntimeError(f"Graph contains cycle or exceeded max depth of {MAX_ITERATIONS}")
            
        # Join -> Children
        children = pd.merge(
            frontier,
            edges_sub,
            left_on=['tweet_id', 'cascade_id'],
            right_on=['src', 'cascade_id'],
            how='inner'
        )
        
        if len(children) == 0:
            break
            
        children = children[['dst', 'cascade_id', 'depth']].copy()
        children.rename(columns={'dst': 'tweet_id'}, inplace=True)
        children['depth'] = children['depth'] + 1.0
        
        frontier = children
        frontier_count = len(frontier)
        
        depths_list.append(frontier)

    all_depths = pd.concat(depths_list, ignore_index=True)

    # Join Back -> NULL -> unreachable
    result = vertices_pd[['id', 'cascade_id']].copy()
    result.rename(columns={'id': 'tweet_id'}, inplace=True)
    
    result = pd.merge(
        result,
        all_depths,
        on=['tweet_id', 'cascade_id'],
        how='left'
    )
    
    result['reachable'] = result['depth'].notna()
    
    return result
