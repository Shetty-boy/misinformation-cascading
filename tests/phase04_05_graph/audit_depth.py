import sys
import os
import random
import logging
import networkx as nx
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
    
    print("Computing depths (Spark)...")
    depths_df = compute_depths(vertices_df, edges_df)
    
    print("Selecting cascades...")
    # Seed randomness
    random.seed(42)
    
    # Get all distinct cascades
    all_cascades = [r["cascade_id"] for r in vertices_df.select("cascade_id").distinct().collect()]
    
    # 5 "manual" (just pick first 5 sorted to be deterministic "manual" choice)
    all_cascades_sorted = sorted(all_cascades)
    manual_cascades = all_cascades_sorted[:5]
    
    # 50 random cascades (excluding the 5 manual)
    remaining_cascades = all_cascades_sorted[5:]
    random_cascades = random.sample(remaining_cascades, min(50, len(remaining_cascades)))
    
    target_cascades = manual_cascades + random_cascades
    
    print(f"Targeting {len(target_cascades)} cascades for NetworkX validation...")
    
    # Filter dfs
    v_local = vertices_df.filter(F.col("cascade_id").isin(target_cascades)).collect()
    e_local = edges_df.filter(F.col("cascade_id").isin(target_cascades)).collect()
    d_local = depths_df.filter(F.col("cascade_id").isin(target_cascades)).collect()
    
    # Organize by cascade
    from collections import defaultdict
    nodes_by_cascade = defaultdict(list)
    edges_by_cascade = defaultdict(list)
    spark_depths_by_cascade = defaultdict(dict)
    
    for r in v_local:
        nodes_by_cascade[r.cascade_id].append(r)
    for r in e_local:
        edges_by_cascade[r.cascade_id].append(r)
    for r in d_local:
        spark_depths_by_cascade[r.cascade_id][r.tweet_id] = r
        
    mismatches = []
    
    print("Validating with NetworkX BFS...")
    for cid in target_cascades:
        G = nx.DiGraph()
        roots = []
        
        for n in nodes_by_cascade[cid]:
            G.add_node(n.id)
            if n.parent_id is None:
                roots.append(n.id)
                
        for e in edges_by_cascade[cid]:
            G.add_edge(e.src, e.dst)
            
        # BFS
        nx_depths = {}
        for node in G.nodes():
            nx_depths[node] = None
            
        for root in roots:
            nx_depths[root] = 0
            queue = [root]
            while queue:
                current = queue.pop(0)
                d = nx_depths[current]
                for child in G.successors(current):
                    if nx_depths[child] is None or nx_depths[child] > d + 1:
                        nx_depths[child] = d + 1
                        queue.append(child)
                        
        # Compare
        for node in G.nodes():
            spark_res = spark_depths_by_cascade[cid][node]
            s_depth = spark_res.depth
            s_reachable = spark_res.reachable
            
            n_depth = nx_depths[node]
            n_reachable = n_depth is not None
            
            if s_depth != n_depth or s_reachable != n_reachable:
                mismatches.append({
                    "cascade_id": cid,
                    "tweet_id": node,
                    "spark_depth": s_depth,
                    "nx_depth": n_depth,
                    "spark_reachable": s_reachable,
                    "nx_reachable": n_reachable
                })
                
    # Generate Report
    report_path = "logs/phase04_05_graph/depth_validation_report.md"
    os.makedirs("logs/phase04_05_graph", exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Depth Validation Report\n\n")
        f.write(f"**Target Cascades:** {len(target_cascades)} (5 manual, 50 random)\n")
        f.write(f"**Mismatches Found:** {len(mismatches)}\n\n")
        
        if len(mismatches) == 0:
            f.write("✅ **SUCCESS**: Spark distributed BFS matches NetworkX manual BFS exactly for all tested nodes.\n")
        else:
            f.write("❌ **FAILURE**: Mismatches detected.\n\n")
            f.write("| Cascade ID | Tweet ID | Spark Depth | NX Depth | Spark Reachable | NX Reachable |\n")
            f.write("|---|---|---|---|---|---|\n")
            for m in mismatches[:100]:
                f.write(f"| {m['cascade_id']} | {m['tweet_id']} | {m['spark_depth']} | {m['nx_depth']} | {m['spark_reachable']} | {m['nx_reachable']} |\n")
                
    if len(mismatches) > 0:
        print(f"DEPTH VALIDATION FAILED: {len(mismatches)} mismatches. See report.")
        sys.exit(1)
        
    print("SUCCESS: Depth Validation matches NetworkX perfectly.")

if __name__ == "__main__":
    run_audit()
