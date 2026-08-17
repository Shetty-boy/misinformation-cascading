import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc

OUT_DIR = "data/processed/phase13_14_adaptive_stopping/checkpoints"


def _find_optimal_static_thresholds(df: pd.DataFrame, conf_col: str, target_precision=0.85):
    """
    Finds T_opt(t) for each time window that achieves at least target_precision
    on the training set. If unable to reach target, takes the max threshold.
    """
    t_opt = {}
    for t_min, group in df.groupby("t_minutes"):
        y_true = group["label_binary"].values
        y_score = group[conf_col].values
        
        if sum(y_true) == 0:
            t_opt[t_min] = 1.0
            continue
            
        prec, rec, thresh = precision_recall_curve(y_true, y_score)
        # Find lowest threshold that gives precision >= target_precision
        valid_idx = np.where(prec[:-1] >= target_precision)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[0]
            t_opt[t_min] = thresh[best_idx]
        else:
            t_opt[t_min] = 1.0  # default to very strict if we can't get good precision
            
    return t_opt


def fit_adaptive_threshold(confidence_df: pd.DataFrame, split_df: pd.DataFrame, model_type: str = "c2v") -> dict:
    """
    Trains XGBoost to predict the optimal threshold.
    Returns a dict with model and best fixed thresholds.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    
    conf_col = f"confidence_{model_type}"
    features_cols = [
        "t_minutes", "node_count", "max_depth", "growth_velocity", 
        "burstiness", "branching_factor", "mean_interarrival", conf_col
    ]
    
    # Isolate training set
    train_df = confidence_df[confidence_df["split"] == "train"].copy()
    val_df = confidence_df[confidence_df["split"] == "val"].copy()
    
    # 1. Find T_opt(t)
    t_opt = _find_optimal_static_thresholds(train_df, conf_col, target_precision=0.85)
    
    # Also find a single best fixed threshold across all t
    y_true_val = val_df["label_binary"].values
    y_score_val = val_df[conf_col].values
    best_fixed = 1.0
    if sum(y_true_val) > 0:
        prec, rec, thresh = precision_recall_curve(y_true_val, y_score_val)
        valid_idx = np.where(prec[:-1] >= 0.85)[0]
        if len(valid_idx) > 0:
            best_fixed = float(thresh[valid_idx[0]])
            
    # 2. Construct Oracle target for XGBoost
    # y = T_opt(t) if label=1 and confidence >= T_opt(t), else 1.0
    def make_target(row):
        t = row["t_minutes"]
        t_thresh = t_opt.get(t, 1.0)
        if row["label_binary"] == 1 and row[conf_col] >= t_thresh:
            return t_thresh
        return 1.0
        
    train_df["target_thresh"] = train_df.apply(make_target, axis=1)
    val_df["target_thresh"] = val_df.apply(make_target, axis=1)
    
    X_train = train_df[features_cols]
    y_train = train_df["target_thresh"]
    
    X_val = val_df[features_cols]
    y_val = val_df["target_thresh"]
    
    # 3. Train XGBoost (hyperparameter sweep logic built in)
    best_model = None
    best_val_loss = float('inf')
    best_params = {}
    
    for max_depth in [3, 5, 7]:
        for n_est in [100, 200]:
            for lr in [0.05, 0.1]:
                model = xgb.XGBRegressor(
                    max_depth=max_depth,
                    n_estimators=n_est,
                    learning_rate=lr,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                val_loss = np.mean((y_val - preds) ** 2)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_params = {"max_depth": max_depth, "n_estimators": n_est, "learning_rate": lr}
                    
    # Save model
    model_path = os.path.join(OUT_DIR, f"threshold_model_{model_type}.json")
    best_model.save_model(model_path)
    
    return {
        "model_path": model_path,
        "model": best_model,
        "best_fixed_threshold": best_fixed,
        "t_opt": t_opt,
        "best_params": best_params,
        "features": features_cols
    }


def predict_threshold(model, features_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    X = features_df[feature_cols]
    return model.predict(X)
