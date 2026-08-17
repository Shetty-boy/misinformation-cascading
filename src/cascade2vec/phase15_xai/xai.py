import os
import json
import logging
import pandas as pd
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

from cascade2vec.phase11_12_cascade2vec.cascade2vec import CASCADE2VEC, C2VClassifier, SnapshotDataset, DEVICE
from cascade2vec.phase11_12_cascade2vec.run_c2v import _load_best_config
from cascade2vec.phase11_12_cascade2vec.sweep import _build_tfidf
from cascade2vec.phase15_xai.gnn_explain import filter_explainable_cascades, build_explainer, explain_cascade, plot_cascade_explanation, aggregate_explanations
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

UNIFIED_FILE = "data/processed/phase02_ingestion/unified.parquet"
SPLIT_FILE = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"
FM_FILE = "data/processed/phase06_07_features/feature_matrix.parquet"
CONF_FILE = "data/processed/phase13_14_adaptive_stopping/confidence_features.parquet"
THRESHOLD_MODEL = "data/processed/phase13_14_adaptive_stopping/checkpoints/threshold_model_c2v.json"
C2V_CKPT = "data/processed/phase11_12_cascade2vec/checkpoints/final_model.pt"
OUT_DIR = "logs/phase15_xai"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

def generate_shap_explanations():
    """Generate SHAP explanations for the XGBoost adaptive threshold model."""
    logger.info("Generating SHAP explanations for tabular models...")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    if not os.path.exists(THRESHOLD_MODEL):
        logger.error(f"Threshold model not found: {THRESHOLD_MODEL}")
        return
        
    df = pd.read_parquet(CONF_FILE)
    split_df = df[["cascade_id", "split"]].drop_duplicates()
    
    # We use the same features as phase 13-14
    model_type = "c2v"
    conf_col = f"confidence_{model_type}"
    features_cols = [
        "t_minutes", "node_count", "max_depth", "growth_velocity", 
        "burstiness", "branching_factor", "mean_interarrival", conf_col
    ]
    
    # Load model
    model = xgb.XGBRegressor()
    model.load_model(THRESHOLD_MODEL)
    
    # Use test set for explanations
    test_df = df[df["split"] == "test"].copy()
    
    if test_df.empty:
        logger.error("Test dataframe is empty. Cannot compute SHAP.")
        return
        
    X_test = test_df[features_cols]
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Global feature importance bar plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.savefig(os.path.join(PLOTS_DIR, "shap_bar.png"), bbox_inches='tight')
    plt.close()
    
    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(os.path.join(PLOTS_DIR, "shap_beeswarm.png"), bbox_inches='tight')
    plt.close()
    
    # Dependence plots for top features
    top_features = ["burstiness", conf_col, "max_depth"]
    for feat in top_features:
        if feat in features_cols:
            plt.figure(figsize=(8, 6))
            shap.plots.scatter(shap_values[:, feat], color=shap_values, show=False)
            plt.savefig(os.path.join(PLOTS_DIR, f"shap_dependence_{feat}.png"), bbox_inches='tight')
            plt.close()
            
    # Force plots for illustrative cascades
    # Select 3 cascades: true positive (caught rumour), true negative, and false positive/negative
    # We will just pick random cascades for illustration here since we don't have predictions directly available
    test_df_sorted = test_df.sort_values(by=conf_col, ascending=False)
    
    # Rumour with high confidence
    rumours = test_df_sorted[(test_df_sorted["label_binary"] == 1) & (test_df_sorted[conf_col] > 0.8)]
    if not rumours.empty:
        idx = rumours.index[0]
        iloc_idx = X_test.index.get_loc(idx)
        plt.figure(figsize=(12, 4))
        shap.plots.waterfall(shap_values[iloc_idx], show=False)
        plt.savefig(os.path.join(PLOTS_DIR, "shap_waterfall_true_rumour.png"), bbox_inches='tight')
        plt.close()
        
    # Non-rumour with low confidence
    non_rumours = test_df_sorted[(test_df_sorted["label_binary"] == 0) & (test_df_sorted[conf_col] < 0.2)]
    if not non_rumours.empty:
        idx = non_rumours.index[0]
        iloc_idx = X_test.index.get_loc(idx)
        plt.figure(figsize=(12, 4))
        shap.plots.waterfall(shap_values[iloc_idx], show=False)
        plt.savefig(os.path.join(PLOTS_DIR, "shap_waterfall_true_non_rumour.png"), bbox_inches='tight')
        plt.close()

def generate_gnn_explanations(n_cascades=20):
    """Generate GNNExplainer explanations for CASCADE2VEC."""
    logger.info("Generating GNNExplainer explanations...")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    if not os.path.exists(C2V_CKPT):
        logger.error(f"CASCADE2VEC model not found: {C2V_CKPT}")
        return
        
    unified = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)
    
    best_cfg = _load_best_config()
    tfidf = _build_tfidf(unified, split_df)
    
    encoder = CASCADE2VEC(
        in_dim=5000, hidden_dim=128, embed_dim=best_cfg["embed_dim"],
        n_layers=best_cfg["n_layers"], dropout=0.5
    ).to(DEVICE)
    classifier = C2VClassifier(embed_dim=best_cfg["embed_dim"], num_classes=2).to(DEVICE)
    
    ckpt = torch.load(C2V_CKPT, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    classifier.load_state_dict(ckpt["classifier"])
    encoder.eval()
    classifier.eval()
    
    # We need a unified model to pass to explainer
    class UnifiedModel(torch.nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
            
        def forward(self, x, edge_index, edge_weight=None, batch=None):
            z = self.encoder(x, edge_index, edge_weight, batch)
            return self.classifier(z)
            
    model = UnifiedModel(encoder, classifier).to(DEVICE)
    explainer = build_explainer(model)
    
    # Get test dataset
    ds = SnapshotDataset(unified, split_df, "test", tfidf, best_cfg["lam"])
    
    # Filter for connected, non-singleton cascades
    valid_data = filter_explainable_cascades(ds.data_list)
    logger.info(f"Filtered to {len(valid_data)} explainable cascades (out of {len(ds)} total).")
    
    # We want a mix of rumours and non-rumours
    rumours = [d for d in valid_data if d.y.item() == 1]
    non_rumours = [d for d in valid_data if d.y.item() == 0]
    
    n_per_class = min(n_cascades // 2, len(rumours), len(non_rumours))
    if n_per_class == 0:
        logger.warning("Not enough valid cascades to explain.")
        return
        
    # Pick the largest cascades for better visualizations
    rumours.sort(key=lambda d: d.num_nodes, reverse=True)
    non_rumours.sort(key=lambda d: d.num_nodes, reverse=True)
    
    selected_data = rumours[:n_per_class] + non_rumours[:n_per_class]
    
    results = []
    for i, data in enumerate(selected_data):
        data = data.to(DEVICE)
        node_mask, edge_mask = explain_cascade(explainer, data)
        
        save_path = os.path.join(PLOTS_DIR, f"gnn_explain_cascade_{data.cascade_id}_{data.t_minutes}m.png")
        plot_cascade_explanation(data, edge_mask, node_mask, save_path)
        
        results.append({
            "cascade_id": getattr(data, 'cascade_id', f"unknown_{i}"),
            "t_minutes": getattr(data, 't_minutes', 0),
            "label": "rumour" if data.y.item() == 1 else "non-rumour",
            "node_mask": node_mask.cpu(),
            "edge_mask": edge_mask.cpu(),
        })
        
    df_res = aggregate_explanations(results)
    df_res.to_csv(os.path.join(OUT_DIR, "gnn_explain_stats.csv"), index=False)
    logger.info("GNN explanations complete.")

def generate_xai_summary():
    """Generate the markdown summary for Phase 15 XAI."""
    summary_path = os.path.join(OUT_DIR, "xai_summary.md")
    
    content = f"""# Phase 15: Explainability (XAI)

## 1. Tabular Explainability (SHAP on Adaptive Threshold)
We computed SHAP values for the XGBoost adaptive threshold model. This shows which structural and temporal features most heavily influence the dynamic stopping threshold.

### Global Feature Importance
![SHAP Bar Plot](plots/shap_bar.png)

### Feature Distribution (Beeswarm)
![SHAP Beeswarm Plot](plots/shap_beeswarm.png)

### Dependence Plots
![Dependence - Burstiness](plots/shap_dependence_burstiness.png)
![Dependence - Confidence](plots/shap_dependence_confidence_c2v.png)

---

## 2. Graph Explainability (GNNExplainer on CASCADE2VEC)
We applied PyG's `GNNExplainer` to attribute importance to specific edges and nodes within the CASCADE2VEC model's predictions.

> [!WARNING]
> **Limitation: Singletons & Disconnected Cascades**
> GNNExplainer attributes importance over edges. The dataset contains ~358 singleton cascades and ~606 disconnected cascades. These were explicitly excluded from the explanation sample, as the explainer cannot produce meaningful structural attribution without connected edges.

### Sample Explanations
*Review `logs/phase15_xai/plots/` for individual cascade topology plots with edge importance highlighted.*

**Aggregate Statistics:**
The mean edge importance and node importance were computed across the sampled explainable cascades. See `logs/phase15_xai/gnn_explain_stats.csv` for detailed metrics.
"""
    with open(summary_path, "w") as f:
        f.write(content)
        
    # Update master index
    index_file = "docs/all_phases_results_index.md"
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            idx_content = f.read()
            
        if "Phase 15: Explainability (XAI)" not in idx_content:
            section = """
## Phase 15: Explainability (XAI)
- **Summary:** [xai_summary.md](file:///home/dr_shetty/misinformation-cascading/logs/phase15_xai/xai_summary.md)
"""
            with open(index_file, "a") as f:
                f.write(section)
                
    logger.info(f"Summary written to {summary_path}")

def run_all(n_cascades=20):
    generate_shap_explanations()
    generate_gnn_explanations(n_cascades)
    generate_xai_summary()
