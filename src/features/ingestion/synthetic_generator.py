import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id, rand
import os

def generate_faithful_sample(n_nodes=50000, m_edges=2, beta=0.05, gamma=0.01, fraction_infected=0.001):
    """
    Generate a faithful SIR cascade using NetworkX and NDlib to extract structural statistics.
    """
    print(f"Generating Barabasi-Albert graph with {n_nodes} nodes...")
    g = nx.barabasi_albert_graph(n_nodes, m_edges)
    
    # Configure SIR model
    model = ep.SIRModel(g)
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)
    config.add_model_parameter("fraction_infected", fraction_infected)
    model.set_initial_status(config)
    
    print("Running SIR simulation for 100 iterations...")
    iterations = model.iteration_bunch(100)
    
    # Extract the new infections per time step
    new_infections_per_step = []
    for it in iterations:
        if 'status_delta' in it:
            # state 1 is infected (0=S, 1=I, 2=R)
            new_infected = sum(1 for node, state in it['status_delta'].items() if state == 1)
            new_infections_per_step.append(new_infected)
            
    total_infected = sum(new_infections_per_step)
    
    # Calculate average burstiness (infections per timestep during the active spread phase)
    active_steps = [x for x in new_infections_per_step if x > 0]
    avg_burstiness = np.mean(active_steps) if active_steps else 1.0
    
    print(f"Faithful sample generated. Total infected over 100 steps: {total_infected}")
    print(f"Average new infections per active timestep (burstiness): {avg_burstiness:.2f}")
    
    return {
        "avg_burstiness": avg_burstiness,
        "total_infected": total_infected
    }

def generate_scalable_volume(output_dir="data/processed/synthetic/", num_cascades=1000, avg_burstiness=50):
    """
    Generate massive volume using PySpark based on faithful parameters.
    """
    print("\nInitializing PySpark session for volume generation...")
    spark = SparkSession.builder.appName("SyntheticCascadeGenerator").master("local[*]").getOrCreate()
    
    # Generate cascade_ids
    cascade_df = spark.range(0, num_cascades).withColumnRenamed("id", "cascade_id")
    
    # For each cascade, generate a random number of nodes (parameterized by the burstiness)
    # This ensures our volume graphs have similar scale properties to the SIR simulation
    max_nodes = int(avg_burstiness * 10) 
    
    print(f"Generating up to {max_nodes} nodes per cascade across {num_cascades} cascades...")
    node_df = cascade_df.selectExpr("cascade_id", f"explode(sequence(0, CAST(rand() * {max_nodes} + 5 AS INT))) as node_seq")
    
    # Construct the fields mapping to the unified schema
    df = node_df.withColumn("tweet_id", (monotonically_increasing_id()).cast("string")) \
                .withColumn("user_id", (rand() * 1000000).cast("long").cast("string")) \
                .withColumn("timestamp", (rand() * 86400).cast("long")) \
                .withColumn("text", lit("Synthetic tweet text")) \
                .withColumn("parent_id", lit(None).cast("string")) \
                .withColumn("event_id", col("cascade_id").cast("string")) \
                .withColumn("label", (rand() > 0.5).cast("int")) # 50/50 rumor/non-rumor
                
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "synthetic_cascades.parquet")
    
    print(f"Writing massive synthetic dataset to {out_path}...")
    df.write.mode("overwrite").parquet(out_path)
    
    # Print the total count to prove scalability
    total_generated = spark.read.parquet(out_path).count()
    print(f"Scalable volume generation complete. Successfully generated {total_generated} nodes!")

if __name__ == "__main__":
    print("--- 1. Generating Faithful Sample (NDlib) ---")
    stats = generate_faithful_sample()
    
    print("--- 2. Generating Scalable Volume (PySpark) ---")
    # We pass the empirical burstiness from the SIR model into the PySpark generator
    generate_scalable_volume(avg_burstiness=stats.get("avg_burstiness", 50))
