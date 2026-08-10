import pandas as pd

def check_disconnected():
    df = pd.read_parquet("data/processed/phase06_07_features/feature_matrix_pandas.parquet")
    
    # We want to check how many unique cascade_ids are disconnected (is_connected=False) 
    # or singleton (node_count=1) in any of their snapshots, typically the final one (t=120)
    
    df_t120 = df[df["t_minutes"] == 120]
    
    disconnected = df_t120[df_t120["is_connected"] == False]
    singletons = df_t120[df_t120["node_count"] <= 1]
    
    print(f"Total cascades in sample: {len(df_t120)}")
    print(f"Disconnected cascades: {len(disconnected)}")
    if len(disconnected) > 0:
        print(f"Disconnected IDs: {disconnected['cascade_id'].tolist()}")
        
    print(f"Singleton cascades: {len(singletons)}")
    if len(singletons) > 0:
        print(f"Singleton IDs: {singletons['cascade_id'].tolist()}")
        
if __name__ == "__main__":
    check_disconnected()
