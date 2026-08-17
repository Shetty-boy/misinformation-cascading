import pytest
import os
import pandas as pd
from cascade2vec.phase16_17_scalability.synthetic_generator import generate_scalable_volume

def test_synthetic_generator(tmp_path):
    out_dir = str(tmp_path)
    
    stats = generate_scalable_volume(output_dir=out_dir, num_cascades=5, avg_burstiness=5)
    
    assert stats["num_cascades"] == 5
    assert stats["total_nodes"] > 5
    
    df = pd.read_parquet(os.path.join(out_dir, "synthetic_cascades.parquet"))
    
    # Check schema
    expected_cols = ["cascade_id", "node_seq", "tweet_id", "user_id", "event_id", "parent_id", "label", "label_binary", "text", "timestamp"]
    for col in expected_cols:
        assert col in df.columns
        
    # Check parents
    root_nodes = df[df["parent_id"].isna()]
    assert len(root_nodes) == 5 # One root per cascade
    
    # Verify temporal ordering (parent timestamp <= child timestamp)
    joined = df.merge(df, left_on="parent_id", right_on="tweet_id", suffixes=("_child", "_parent"))
    assert all(joined["timestamp_parent"] <= joined["timestamp_child"])
