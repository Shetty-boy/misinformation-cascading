import sys
import os
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
    
    print("Analyzing reachability...")
    # Calculate stats per cascade
    cascade_stats = depths_df.groupBy("cascade_id").agg(
        F.count("*").alias("total_nodes"),
        F.sum(F.when(F.col("reachable") == True, 1).otherwise(0)).alias("reachable_nodes"),
        F.sum(F.when(F.col("reachable") == False, 1).otherwise(0)).alias("unreachable_nodes"),
        F.max("depth").alias("max_depth")
    ).withColumn(
        "reachable_ratio",
        F.col("reachable_nodes") / F.col("total_nodes")
    )
    
    disconnected = cascade_stats.filter(F.col("unreachable_nodes") > 0).collect()
    
    report_path = "logs/phase04_05_graph/disconnected_cascades_report.md"
    os.makedirs("logs/phase04_05_graph", exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Disconnected Cascades Report\n\n")
        f.write(f"**Total Disconnected Cascades Found:** {len(disconnected)}\n\n")
        
        f.write("| Cascade ID | Total Nodes | Reachable | Unreachable | Reachable Ratio | Max Reachable Depth |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in disconnected:
            f.write(f"| {r.cascade_id} | {r.total_nodes} | {r.reachable_nodes} | {r.unreachable_nodes} | {r.reachable_ratio:.4f} | {r.max_depth} |\n")
            
    print(f"SUCCESS: Reachability audit complete. Found {len(disconnected)} disconnected cascades.")

if __name__ == "__main__":
    run_audit()
