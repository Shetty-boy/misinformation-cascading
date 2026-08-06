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
    
    print("Computing depths...")
    depths_df = compute_depths(vertices_df, edges_df)
    
    print("Validating Edge Integrity (depth(child) == depth(parent) + 1)...")
    
    # Filter reachable nodes only
    reachable_depths = depths_df.filter(F.col("reachable") == True)
    
    # Join parent depth
    edges_with_src_depth = edges_df.join(
        reachable_depths.select(
            F.col("tweet_id").alias("src"),
            F.col("cascade_id"),
            F.col("depth").alias("src_depth")
        ),
        on=["src", "cascade_id"],
        how="inner"
    )
    
    # Join child depth
    edges_with_full_depth = edges_with_src_depth.join(
        reachable_depths.select(
            F.col("tweet_id").alias("dst"),
            F.col("cascade_id"),
            F.col("depth").alias("dst_depth")
        ),
        on=["dst", "cascade_id"],
        how="inner"
    )
    
    invalid_edges = edges_with_full_depth.filter(F.col("dst_depth") != F.col("src_depth") + 1)
    
    invalid_count = invalid_edges.count()
    if invalid_count > 0:
        print(f"INTEGRITY FAILED: Found {invalid_count} edges violating depth(child) = depth(parent) + 1")
        invalid_edges.show(20)
        sys.exit(1)
        
    print("SUCCESS: Graph Edge Integrity Verified.")

if __name__ == "__main__":
    run_audit()
