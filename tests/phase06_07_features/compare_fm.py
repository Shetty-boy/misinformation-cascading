import pandas as pd
import numpy as pd_np

def compare_dataframes():
    df_old = pd.read_parquet("data/processed/phase06_07_features/fm_old_50.parquet")
    df_new = pd.read_parquet("data/processed/phase06_07_features/feature_matrix_pandas.parquet")

    # Sort to ensure aligned comparison
    df_old = df_old.sort_values(["cascade_id", "t_minutes"]).reset_index(drop=True)
    df_new = df_new.sort_values(["cascade_id", "t_minutes"]).reset_index(drop=True)

    if len(df_old) != len(df_new):
        print(f"Row count mismatch: {len(df_old)} != {len(df_new)}")
        return

    features = [
        "node_count", "edge_count", "max_depth", "avg_depth", "leaf_count", "leaf_ratio",
        "branching_factor", "root_degree", "reachable_ratio", "is_connected",
        "tweets_per_minute", "growth_velocity", "mean_interarrival", "std_interarrival",
        "burstiness", "cascade_age", "depth_velocity", "breadth_velocity", "branching_velocity"
    ]

    all_match = True
    for col in features:
        try:
            pd.testing.assert_series_equal(df_old[col], df_new[col], check_exact=False, rtol=1e-5, atol=1e-5)
            print(f"✅ {col}: match")
        except AssertionError as e:
            all_match = False
            print(f"❌ {col}: MISMATCH")
            print(e)
            
            # Show a sample mismatch
            mask = ~pd_np.isclose(df_old[col].astype(float), df_new[col].astype(float), rtol=1e-5, atol=1e-5)
            diffs = pd.DataFrame({
                "cascade_id": df_old.loc[mask, "cascade_id"],
                "t": df_old.loc[mask, "t_minutes"],
                "old": df_old.loc[mask, col],
                "new": df_new.loc[mask, col]
            })
            print(diffs.head())
            print("-" * 40)

    if all_match:
        print("\nAll 19 features match exactly (or within tolerance) between Spark and Pandas pipelines!")
    else:
        print("\nSome features had mismatches.")

if __name__ == "__main__":
    compare_dataframes()
