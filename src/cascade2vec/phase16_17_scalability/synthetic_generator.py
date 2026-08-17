import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id, rand
import os
import argparse
import time

def generate_faithful_sample(n_nodes=50000, m_edges=2, beta=0.05, gamma=0.01, fraction_infected=0.001):
    """
    Generate a faithful SIR cascade using NetworkX and NDlib to extract structural statistics.
    """
    g = nx.barabasi_albert_graph(n_nodes, m_edges)
    
    # Configure SIR model
    model = ep.SIRModel(g)
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)
    config.add_model_parameter("fraction_infected", fraction_infected)
    model.set_initial_status(config)
    
    iterations = model.iteration_bunch(100)
    
    # Extract the new infections per time step
    new_infections_per_step = []
    for it in iterations:
        if 'status_delta' in it:
            new_infected = sum(1 for node, state in it['status_delta'].items() if state == 1)
            new_infections_per_step.append(new_infected)
            
    total_infected = sum(new_infections_per_step)
    active_steps = [x for x in new_infections_per_step if x > 0]
    avg_burstiness = np.mean(active_steps) if active_steps else 1.0
    
    return {
        "avg_burstiness": avg_burstiness,
        "total_infected": total_infected
    }

def _build_trees(iterator):
    """
    Pandas UDF-like function to construct trees for each cascade partition.
    Takes an iterator of DataFrames.
    """
    vocab = ["breaking", "news", "rumour", "confirmed", "fake", "truth", "viral", "update"]
    
    for pdf in iterator:
        # We need to assign parent_id. 
        # For each cascade_id, sort by generated sequence (simulating time).
        # First node is root (parent_id = None). Subsequent nodes attach to a random previous node.
        
        # Sort by cascade_id and node_seq to ensure causal ordering
        pdf = pdf.sort_values(['cascade_id', 'node_seq'])
        
        parent_ids = []
        labels = []
        texts = []
        timestamps = []
        
        for cascade_id, group in pdf.groupby('cascade_id'):
            n = len(group)
            tweet_ids = group['tweet_id'].tolist()
            
            # Root node
            parent_ids.append(None)
            
            # Random attachment for the rest
            for i in range(1, n):
                # Pick a random parent from nodes that arrived before
                parent_idx = np.random.randint(0, i)
                parent_ids.append(tweet_ids[parent_idx])
                
            # Cascade-level label (same for all nodes in cascade)
            is_rumour = np.random.rand() > 0.5
            labels.extend([int(is_rumour)] * n)
            
            # Text and timestamps
            base_time = int(np.random.rand() * 86400)
            for i in range(n):
                texts.append(" ".join(np.random.choice(vocab, 5)))
                timestamps.append(base_time + int(np.random.exponential(300) + (i * 60)))
                
        pdf['parent_id'] = parent_ids
        pdf['label'] = labels
        pdf['label_binary'] = labels
        pdf['text'] = texts
        pdf['timestamp'] = timestamps
        
        yield pdf

def generate_scalable_volume(output_dir="data/processed/phase16_17_scalability/", num_cascades=1000, avg_burstiness=50):
    """
    Generate massive volume using PySpark.
    """
    t0 = time.time()
    spark = SparkSession.builder.appName("SyntheticCascadeGenerator").getOrCreate()
    
    cascade_df = spark.range(0, num_cascades).withColumnRenamed("id", "cascade_id")
    
    max_nodes = int(avg_burstiness * 10) 
    
    # Generate nodes
    node_df = cascade_df.selectExpr(
        "cast(cascade_id as string) as cascade_id", 
        f"explode(sequence(0, CAST(rand() * {max_nodes} + 5 AS INT))) as node_seq"
    )
    
    # Assign basic IDs
    df = node_df.withColumn("tweet_id", monotonically_increasing_id().cast("string")) \
                .withColumn("user_id", (rand() * 1000000).cast("long").cast("string")) \
                .withColumn("event_id", col("cascade_id"))
                
    # Now we need to assign parent_ids to form trees. We'll use mapInPandas.
    schema = "cascade_id string, node_seq int, tweet_id string, user_id string, event_id string, parent_id string, label int, label_binary int, text string, timestamp long"
    
    df_tree = df.repartition("cascade_id").mapInPandas(_build_trees, schema)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "synthetic_cascades.parquet")
    
    df_tree.write.mode("overwrite").parquet(out_path)
    
    total_nodes = spark.read.parquet(out_path).count()
    t1 = time.time()
    
    stats = {
        "num_cascades": num_cascades,
        "total_nodes": total_nodes,
        "time_seconds": t1 - t0
    }
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cascades", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="data/processed/phase16_17_scalability/")
    args = parser.parse_args()
    
    stats = generate_faithful_sample()
    gen_stats = generate_scalable_volume(
        output_dir=args.output_dir, 
        num_cascades=args.num_cascades,
        avg_burstiness=stats.get("avg_burstiness", 50)
    )
    print(f"Generated {gen_stats['total_nodes']} nodes across {gen_stats['num_cascades']} cascades in {gen_stats['time_seconds']:.1f}s.")
