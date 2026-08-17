import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def run_detection_loop(confidence_df: pd.DataFrame, split_name: str, 
                       threshold_model, feature_cols: list, 
                       best_fixed: float, model_type: str = "c2v") -> pd.DataFrame:
    """
    Simulates real-time detection on the given split.
    """
    df = confidence_df[confidence_df["split"] == split_name].copy()
    conf_col = f"confidence_{model_type}"
    
    # Sort logically
    df = df.sort_values(["cascade_id", "t_minutes"])
    
    results = []
    
    # Pre-predict adaptive thresholds for the whole dataframe
    X = df[feature_cols]
    df["adaptive_thresh"] = threshold_model.predict(X)
    
    for cascade_id, group in df.groupby("cascade_id"):
        label = group["label_binary"].iloc[0]
        
        # We need data up to t=120
        # Check if cascade reached threshold early for Adaptive and Fixed
        adaptive_time = 120
        adaptive_pred = 0
        fixed_time = 120
        fixed_pred = 0
        
        # Iterate over time windows
        for _, row in group.iterrows():
            t = row["t_minutes"]
            conf = row[conf_col]
            ada_t = row["adaptive_thresh"]
            
            # Adaptive stopping logic
            if adaptive_time == 120 and conf >= ada_t:
                adaptive_time = t
                adaptive_pred = 1  # We stopped early because we confidently flagged a rumour
                
            # Fixed scalar threshold stopping logic
            if fixed_time == 120 and conf >= best_fixed:
                fixed_time = t
                fixed_pred = 1
                
        # If we reached 120 without stopping, use the t=120 confidence
        row_120 = group[group["t_minutes"] == 120]
        if row_120.empty:
            row_120 = group.iloc[-1:]  # Fallback to last available
        
        conf_120 = row_120[conf_col].values[0]
        
        if adaptive_time == 120:
            adaptive_pred = 1 if conf_120 >= best_fixed else 0
            
        if fixed_time == 120:
            fixed_pred = 1 if conf_120 >= best_fixed else 0
            
        # Fixed t=120 baseline
        fixed_120_pred = 1 if conf_120 >= best_fixed else 0
        
        # Fixed t=30 baseline
        row_30 = group[group["t_minutes"] <= 30]
        if row_30.empty:
            row_30 = group.iloc[:1]
        else:
            row_30 = row_30.iloc[-1:]
        conf_30 = row_30[conf_col].values[0]
        fixed_30_pred = 1 if conf_30 >= best_fixed else 0
        
        results.append({
            "cascade_id": cascade_id,
            "label_binary": label,
            "adaptive_time": adaptive_time,
            "adaptive_pred": adaptive_pred,
            "fixed_thresh_time": fixed_time,
            "fixed_thresh_pred": fixed_pred,
            "fixed_120_time": 120,
            "fixed_120_pred": fixed_120_pred,
            "fixed_30_time": 30,
            "fixed_30_pred": fixed_30_pred
        })
        
    return pd.DataFrame(results)


def compute_detection_metrics(results_df: pd.DataFrame) -> dict:
    """
    Computes MDT, Median MDT, F1, and early stop % for all 4 strategies.
    """
    metrics = {}
    y_true = results_df["label_binary"].values
    
    strategies = [
        ("adaptive", "adaptive_time", "adaptive_pred"),
        ("fixed_thresh", "fixed_thresh_time", "fixed_thresh_pred"),
        ("fixed_120", "fixed_120_time", "fixed_120_pred"),
        ("fixed_30", "fixed_30_time", "fixed_30_pred"),
    ]
    
    for name, t_col, p_col in strategies:
        y_pred = results_df[p_col].values
        f1 = f1_score(y_true, y_pred, average="macro")
        
        t_vals = results_df[t_col].values
        mdt = np.mean(t_vals)
        median_mdt = np.median(t_vals)
        
        early_stop_pct = np.mean(t_vals < 120) * 100.0
        
        rumour_t = results_df[results_df["label_binary"] == 1][t_col].values
        non_rumour_t = results_df[results_df["label_binary"] == 0][t_col].values
        
        metrics[name] = {
            "macro_f1": float(f1),
            "mdt": float(mdt),
            "median_mdt": float(median_mdt),
            "early_stop_pct": float(early_stop_pct),
            "mdt_rumour": float(np.mean(rumour_t)) if len(rumour_t) > 0 else 0.0,
            "mdt_non_rumour": float(np.mean(non_rumour_t)) if len(non_rumour_t) > 0 else 0.0
        }
        
    return metrics
