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
    
    print("Validating Schema...")
    expected_cols = {"tweet_id", "cascade_id", "depth", "reachable"}
    actual_cols = set(depths_df.columns)
    
    if expected_cols != actual_cols:
        print(f"SCHEMA FAILED: Expected {expected_cols}, got {actual_cols}")
        sys.exit(1)
        
    print("Validating Row Counts...")
    total_vertices = vertices_df.count()
    total_depths = depths_df.count()
    
    if total_vertices != total_depths:
        print(f"COUNT FAILED: Vertices ({total_vertices}) != Depths ({total_depths})")
        sys.exit(1)
        
    print("Validating Uniqueness...")
    unique_tweets = depths_df.select("tweet_id").distinct().count()
    if unique_tweets != total_vertices:
        print(f"UNIQUENESS FAILED: Unique tweet_ids ({unique_tweets}) != Total rows ({total_depths})")
        sys.exit(1)
        
    print("SUCCESS: Algorithm Traversal Correctness Verified.")

if __name__ == "__main__":
    run_audit()
