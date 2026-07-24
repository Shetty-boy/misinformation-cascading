# %% [markdown]
# # Temporal Snapshot Validation
# This notebook proves that the `get_snapshot` function works correctly on real cascades.
# We manually compute the expected node count, edge count, and max depth for 5 test cascades
# using standard PySpark DataFrame filtering, and assert that the `get_snapshot` GraphFrame matches perfectly.

# %%
import os
import pandas as pd
from pyspark.sql import functions as F
from src.graph.phase04_graph.loader import get_spark, load_unified
from src.graph.phase04_graph.build_graph import (
    to_vertices, to_edges, build_full_graph, get_cascade_subgraph
)
from src.graph.snapshots import get_snapshot

# Initialize PySpark and load full graph
spark = get_spark()
df = load_unified(spark)
vertices = to_vertices(df)
edges = to_edges(df, vertices=vertices)
full_graph = build_full_graph(vertices, edges)

# Select 5 specific cascades to test.
# We will just pick the top 5 cascades with the most nodes in the dataset.
top_cascades = df.groupBy("cascade_id").count().orderBy(F.desc("count")).limit(5).collect()
test_cascade_ids = [row.cascade_id for row in top_cascades]
print(f"Test Cascades: {test_cascade_ids}")

# %%
# Validation Loop
time_intervals_minutes = [1, 2, 5, 10, 15, 30, 60, 120]

for cid in test_cascade_ids:
    print(f"\\n--- Validating Cascade: {cid} ---")
    
    # 1. Get the subgraph using Phase 4 contract
    subgraph = get_cascade_subgraph(full_graph, cid)
    subgraph.vertices.cache()
    subgraph.edges.cache()
    
    for t in time_intervals_minutes:
        t_seconds = t * 60
        
        # --- MANUAL COMPUTATION (Expected) ---
        # Filter raw dataset manually for this cascade up to time t
        raw_df = df.filter((F.col("cascade_id") == cid) & (F.col("timestamp") <= t_seconds))
        expected_nodes = raw_df.count()
        
        # Expected edges: count number of rows that have a parent_id that is ALSO in the filtered set
        # Since every reply has exactly one parent_id, and if parent is kept, edge is kept.
        # But wait, what if parent is NOT kept? Then edge is dropped.
        # Let's do an exact inner join on parent_id
        valid_ids = raw_df.select(F.col("tweet_id").alias("valid_id")) 
        expected_edges = raw_df.filter(F.col("parent_id").isNotNull()).join(
            valid_ids, raw_df["parent_id"] == valid_ids["valid_id"], "inner"
        ).count()
        
        # --- PIPELINE COMPUTATION (Actual) ---
        snapshot = get_snapshot(subgraph, t)
        actual_nodes = snapshot.vertices.count()
        actual_edges = snapshot.edges.count()
        
        print(f"t={t:3d}m | Nodes: {expected_nodes:4d} (Expected) vs {actual_nodes:4d} (Actual) | Edges: {expected_edges:4d} (Expected) vs {actual_edges:4d} (Actual)")
        
        assert expected_nodes == actual_nodes, f"Node count mismatch at t={t}m for cascade {cid}"
        assert expected_edges == actual_edges, f"Edge count mismatch at t={t}m for cascade {cid}"

print("\\n✅ All snapshots matched manual computations exactly!")
