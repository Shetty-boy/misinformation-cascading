import os
import time
import json
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

from cascade2vec.phase11_12_cascade2vec.cascade2vec import CASCADE2VEC, C2VClassifier, SnapshotDataset, DEVICE, SEED, train_c2v, evaluate_c2v
from cascade2vec.phase11_12_cascade2vec.run_c2v import _load_best_config
from cascade2vec.phase11_12_cascade2vec.sweep import _build_tfidf, SWEEP_FIXED
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

logger = logging.getLogger(__name__)

OUT_DIR = "logs/phase18_eval"
ABLATION_RESULTS = os.path.join(OUT_DIR, "ablation_results.json")

UNIFIED_FILE   = "data/processed/phase02_ingestion/unified.parquet"
SPLIT_FILE     = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"

# Custom Ablation Models
class C2V_MeanPool(CASCADE2VEC):
    def forward(self, x, edge_index, edge_weight, batch):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight)
            if i < self.n_layers - 1:
                h = self.dropout(h)
        # Mean pool directly instead of attention
        pooled = global_mean_pool(h, batch)
        embed = self.proj(pooled)
        return torch.nn.functional.normalize(embed, p=2, dim=-1)

def run_cascade2vec_ablation(name: str, best_cfg: dict, unified: pd.DataFrame, split_df: pd.DataFrame, tfidf) -> dict:
    logger.info(f"Running Ablation: {name}")
    
    cfg = best_cfg.copy()
    model_class = CASCADE2VEC
    
    # Apply ablation changes
    if name == "A1_no_decay":
        cfg["lam"] = 0.0
    elif name == "A2_mean_pool":
        model_class = C2V_MeanPool
    elif name == "A3_alpha_0":
        cfg["alpha"] = 0.0
    elif name == "A4_alpha_1":
        cfg["alpha"] = 1.0
    elif name == "A5_1_layer":
        cfg["n_layers"] = 1
    elif name == "A6_embed_dim":
        cfg["embed_dim"] = max(1, cfg["embed_dim"] // 2)
        
    # Datasets
    batch_size = SWEEP_FIXED["batch_size"]
    train_ds = SnapshotDataset(unified, split_df, "train", tfidf, cfg["lam"])
    val_ds   = SnapshotDataset(unified, split_df, "val",   tfidf, cfg["lam"])
    test_ds  = SnapshotDataset(unified, split_df, "test",  tfidf, cfg["lam"])

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  generator=g, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    encoder = model_class(
        in_dim=5000,
        hidden_dim=SWEEP_FIXED["hidden_dim"],
        embed_dim=cfg["embed_dim"],
        n_layers=cfg["n_layers"],
        dropout=SWEEP_FIXED["dropout"],
    )
    classifier = C2VClassifier(embed_dim=cfg["embed_dim"], num_classes=2)

    # Class weights
    train_sub = split_df[split_df["split"] == "train"]
    counts = train_sub["label"].value_counts()
    total = len(train_sub)
    cw = torch.tensor(
        [total / (2 * counts.get("non-rumour", 1)),
         total / (2 * counts.get("rumour", 1))],
        dtype=torch.float32, device=DEVICE,
    )
    
    train_result = train_c2v(
        encoder, classifier, train_loader, val_loader,
        n_epochs=20, # reduced epochs for ablation speed
        lr=SWEEP_FIXED["lr"],
        weight_decay=SWEEP_FIXED["weight_decay"],
        alpha=cfg["alpha"],
        temperature=SWEEP_FIXED["temperature"],
        patience=5,
        class_weights=cw,
        device=DEVICE,
        checkpoint_path=None,
    )
    
    test_metrics = evaluate_c2v(encoder, classifier, test_loader, device=DEVICE, return_embeddings=False)
    
    # Get predictions for McNemar's test later
    encoder.eval()
    classifier.eval()
    
    cascade_preds = {}
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            embeds = encoder(batch.x, batch.edge_index, batch.edge_attr if hasattr(batch, 'edge_attr') and batch.edge_attr.numel() > 0 else None, batch.batch)
            logits = classifier(embeds)
            
            for i in range(batch.num_graphs):
                cid = batch.cascade_id[i] if hasattr(batch.cascade_id, '__getitem__') else batch.cascade_id
                t_min = int(batch.t_minutes[i].item()) if hasattr(batch.t_minutes, '__getitem__') else int(batch.t_minutes)
                pred_y = int(logits[i].argmax().item())
                
                if cid not in cascade_preds or t_min > cascade_preds[cid][1]:
                    cascade_preds[cid] = (pred_y, t_min)
                    
    # Only store predictions in sorted order of cascade_id
    sorted_cids = sorted(list(cascade_preds.keys()))
    preds_ordered = [cascade_preds[c][0] for c in sorted_cids]
    
    test_metrics["predictions"] = preds_ordered
    test_metrics["cascade_ids"] = sorted_cids
    
    return test_metrics

def run_feature_ablations() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from cascade2vec.phase06_07_features.baselines_simple import run_baselines
    
    logger.info("Running Feature Subset Ablations on Logistic Regression...")
    
    results = {}
    
    # 1. Structural features only
    structural_cols = ["node_count", "edge_count", "max_depth", "avg_depth", "leaf_count", "leaf_ratio", "branching_factor", "root_degree", "reachable_ratio", "is_connected"]
    res_str = run_baselines(selected_features=structural_cols)
    lr_str = res_str[(res_str["model"] == "Logistic Regression") & (res_str["weighted"] == True)].iloc[0]
    results["B1_structural_only"] = lr_str["macro_f1"]
    
    # 2. Temporal features only
    temporal_cols = ["tweets_per_minute", "growth_velocity", "mean_interarrival", "std_interarrival", "burstiness", "cascade_age"]
    res_temp = run_baselines(selected_features=temporal_cols)
    lr_temp = res_temp[(res_temp["model"] == "Logistic Regression") & (res_temp["weighted"] == True)].iloc[0]
    results["B2_temporal_only"] = lr_temp["macro_f1"]
    
    # 3. Top-5 features (using an approximation based on prior runs)
    top_5 = ["max_depth", "burstiness", "leaf_ratio", "growth_velocity", "node_count"]
    res_top5 = run_baselines(selected_features=top_5)
    lr_top5 = res_top5[(res_top5["model"] == "Logistic Regression") & (res_top5["weighted"] == True)].iloc[0]
    results["B3_top5_only"] = lr_top5["macro_f1"]
    
    return results

def run_all_ablations():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    logger.info("Loading unified data and best config...")
    unified = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)
    best_cfg = _load_best_config()
    tfidf = _build_tfidf(unified, split_df)
    
    ablations = [
        "A1_no_decay",
        "A2_mean_pool",
        "A3_alpha_0",
        "A4_alpha_1",
        "A5_1_layer",
        "A6_embed_dim"
    ]
    
    results = {}
    for ab in ablations:
        results[ab] = run_cascade2vec_ablation(ab, best_cfg, unified, split_df, tfidf)
        
    feat_results = run_feature_ablations()
    results["feature_ablations"] = feat_results
    
    with open(ABLATION_RESULTS, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info("Ablations complete.")
