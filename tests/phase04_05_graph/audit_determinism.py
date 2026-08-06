import sys
import logging
from pyspark.sql import functions as F

sys.path.append('src')

from cascade2vec.phase04_05_graph.loader import get_spark, load_unified
from cascade2vec.phase04_05_graph.build_graph import to_vertices, to_edges
from cascade2vec.phase04_05_graph.depth import compute_depths

def run_audit():
    print("Initializing Spark...")
    spark = get_spark()
    
    print("Loading graph...")
    raw_df = load_unified(spark)
    vertices_df = to_vertices(raw_df)
    edges_df = to_edges(raw_df, vertices_df)
    
    print("Computing depths (Run A)...")
    depths_A = compute_depths(vertices_df, edges_df).withColumnRenamed("depth", "depth_A").withColumnRenamed("reachable", "reachable_A")
    
    print("Computing depths (Run B)...")
    depths_B = compute_depths(vertices_df, edges_df).withColumnRenamed("depth", "depth_B").withColumnRenamed("reachable", "reachable_B")
    
    print("Validating Determinism...")
    joined = depths_A.join(depths_B, on=["tweet_id", "cascade_id"], how="full")
    
    # Check for mismatches
    mismatches = joined.filter(
        (F.col("depth_A") != F.col("depth_B")) |
        (F.col("reachable_A") != F.col("reachable_B")) |
        (F.col("depth_A").isNull() & F.col("depth_B").isNotNull()) |
        (F.col("depth_A").isNotNull() & F.col("depth_B").isNull())
    )
    
    mismatch_count = mismatches.count()
    if mismatch_count > 0:
        print(f"DETERMINISM FAILED: Found {mismatch_count} mismatches between Run A and Run B")
        mismatches.show(20)
        sys.exit(1)
        
    print("SUCCESS: Determinism Verified.")

if __name__ == "__main__":
    run_audit()
