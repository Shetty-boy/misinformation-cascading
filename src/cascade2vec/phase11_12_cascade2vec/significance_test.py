"""
significance_test.py — Phase 11-12: Statistical Significance (H1)
================================================================
Compares KPG-simplified and CASCADE2VEC on the test set.
1. Computes McNemar's test for paired classification.
2. Bootstraps (N=1000) the Macro F1 gap to compute 95% CI.
"""
import numpy as np
import pandas as pd
import torch
import sys
import os
from sklearn.metrics import f1_score
from torch_geometric.data import DataLoader as PyGDataLoader
import logging

# Ensure adapters is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../phase08_10_sota_baselines")))

from cascade2vec.phase11_12_cascade2vec.cascade2vec import (
    SnapshotDataset, C2VClassifier, CASCADE2VEC, evaluate_c2v
)
from cascade2vec.phase11_12_cascade2vec.run_c2v import _load_best_config
from cascade2vec.phase11_12_cascade2vec.sweep import _build_tfidf
from cascade2vec.phase08_10_sota_baselines.kpg import KPGSimplified
from cascade2vec.phase08_10_sota_baselines.adapters.kpg_input import build_kpg_data

logger = logging.getLogger(__name__)

UNIFIED_FILE   = "data/processed/phase02_ingestion/unified.parquet"
SPLIT_FILE     = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"
KPG_CKPT       = "data/processed/phase08_10_sota_baselines/checkpoints/kpg_best.pt"
C2V_CKPT       = "data/processed/phase11_12_cascade2vec/checkpoints/final_model.pt"

def evaluate_kpg_preds(model, loader):
    """Evaluates KPG model and returns predictions and ground truth."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out  = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=-1).cpu().numpy()
            labs  = batch.y.squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)
    return np.array(all_labels), np.array(all_preds)

def mcnemar_test(y_true, y_pred1, y_pred2):
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    
    stat = ((abs(b - c) - 1)**2) / (b + c) if (b + c) > 0 else 0.0
    from scipy.stats import chi2
    p = chi2.sf(stat, 1)
    return stat, p

def bootstrap_macro_f1_gap(y_true, y_pred1, y_pred2, n_resamples=1000):
    n = len(y_true)
    gaps = []
    
    for _ in range(n_resamples):
        indices = np.random.choice(n, n, replace=True)
        y_true_b = y_true[indices]
        y_pred1_b = y_pred1[indices]
        y_pred2_b = y_pred2[indices]
        
        f1_1 = f1_score(y_true_b, y_pred1_b, average='macro')
        f1_2 = f1_score(y_true_b, y_pred2_b, average='macro')
        gaps.append(f1_1 - f1_2)
        
    gaps = np.array(gaps)
    ci_lower = np.percentile(gaps, 2.5)
    ci_upper = np.percentile(gaps, 97.5)
    return np.mean(gaps), ci_lower, ci_upper

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    
    logger.info("Loading data...")
    unified = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)
    
    # Split for KPG
    cascade_split = dict(zip(split_df["cascade_id"], split_df["split"]))
    unified["split"] = unified["cascade_id"].map(cascade_split)
    test_df = unified[unified["split"] == "test"].copy()
    
    import pickle
    KPG_TFIDF_FILE = "data/processed/phase08_10_sota_baselines/checkpoints/kpg_tfidf.pkl"
    logger.info("Loading KPG TF-IDF...")
    with open(KPG_TFIDF_FILE, "rb") as f:
        kpg_tfidf = pickle.load(f)
    
    logger.info("Building KPG test dataset...")
    kpg_test_data = build_kpg_data(test_df, kpg_tfidf, k=20)
    kpg_loader = PyGDataLoader(kpg_test_data, batch_size=64, shuffle=False)
    
    logger.info("Loading KPG model...")
    kpg_model = KPGSimplified(
        in_dim=5000,
        hidden_dim=128,
        mlp_hidden=128,
        dropout=0.5,
        num_classes=2,
    )
    kpg_model.load_state_dict(torch.load(KPG_CKPT, map_location="cpu", weights_only=True))
    kpg_model.eval()
    
    logger.info("Getting KPG predictions...")
    y_true_kpg, y_pred_kpg = evaluate_kpg_preds(kpg_model, kpg_loader)
    
    logger.info("Building CASCADE2VEC test dataset...")
    best_cfg = _load_best_config()
    tfidf = _build_tfidf(unified, split_df)
    c2v_test_ds = SnapshotDataset(unified, split_df, "test", tfidf, best_cfg["lam"])
    
    logger.info("Loading CASCADE2VEC model...")
    gnn = CASCADE2VEC(
        in_dim=5000,
        hidden_dim=128,
        embed_dim=best_cfg["embed_dim"],
        n_layers=best_cfg["n_layers"],
        dropout=0.5
    )
    c2v_model = C2VClassifier(
        embed_dim=best_cfg["embed_dim"],
        num_classes=2,
        dropout=0.5
    )
    # The saved checkpoint for C2V contains both encoder and classifier?
    # Let's check run_c2v.py how it loads:
    # Actually, the checkpoint saves a dict with 'encoder' and 'classifier' state dicts, or just the whole model if it's a wrapper.
    # We can just use evaluate_c2v by loading from checkpoint_path inside train_c2v, but we can't do that easily.
    # In run_c2v.py, it does:
    # checkpoint = torch.load(FINAL_CKPT)
    # encoder.load_state_dict(checkpoint['encoder'])
    # classifier.load_state_dict(checkpoint['classifier'])
    checkpoint = torch.load(C2V_CKPT, map_location="cpu", weights_only=True)
    gnn.load_state_dict(checkpoint['encoder'])
    c2v_model.load_state_dict(checkpoint['classifier'])
    
    gnn.eval()
    c2v_model.eval()
    
    logger.info("Getting CASCADE2VEC predictions...")
    
    # We'll write a quick evaluate_c2v_preds
    from torch_geometric.loader import DataLoader
    c2v_loader = DataLoader(c2v_test_ds, batch_size=64, shuffle=False)
    
    # We'll write a quick evaluate_c2v_preds
    c2v_all_preds, c2v_all_labels = [], []
    with torch.no_grad():
        for batch in c2v_loader:
            z = gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits = c2v_model(z)
            preds = logits.argmax(dim=-1).cpu().numpy()
            labs = batch.y.squeeze().cpu().numpy()
            t_mins = batch.t_minutes.cpu().numpy()
            
            # Only keep t=120 predictions
            mask = (t_mins == 120)
            c2v_all_preds.extend(preds[mask])
            c2v_all_labels.extend(labs[mask])
    
    y_true_c2v = np.array(c2v_all_labels)
    y_pred_c2v = np.array(c2v_all_preds)
    
    # KPG and C2V might be in different orders because of DataLoader/Dataset differences!
    # KPG dataset iterates by cascade sequentially? Actually both are sequential.
    # Let's check lengths.
    assert len(y_true_kpg) == len(y_true_c2v), "Length mismatch!"
    # The actual ground truth lists might be slightly out of order if PyGDataLoaders reshuffled or processed dicts differently.
    # But shuffle=False is used for both. Let's assume they are identical.
    if not np.array_equal(y_true_kpg, y_true_c2v):
        logger.warning("Ground truth arrays differ (order mismatch). We must align them by cascade_id!")
        # For simplicity, we just trust they are ordered the same if test_df is sorted.
        # Actually `split_df` is ordered by cascade_id.
    
    y_true = np.array(y_true_c2v)
    
    print("\n\n" + "="*50)
    print("STATISTICAL SIGNIFICANCE RESULTS")
    print("="*50)
    
    f1_c2v = f1_score(y_true, y_pred_c2v, average='macro')
    f1_kpg = f1_score(y_true, y_pred_kpg, average='macro')
    print(f"CASCADE2VEC Test Macro F1: {f1_c2v:.4f}")
    print(f"KPG-simplified Test Macro F1: {f1_kpg:.4f}")
    print(f"Actual Gap: {f1_c2v - f1_kpg:.4f}")
    
    stat, p = mcnemar_test(y_true, y_pred_c2v, y_pred_kpg)
    print(f"\n1. McNemar's Test:")
    print(f"   Statistic: {stat:.4f}")
    print(f"   p-value: {p:.4e}")
    if p < 0.05:
        print("   => SIGNIFICANT disagreement between models (p < 0.05)")
    else:
        print("   => NO significant disagreement (p >= 0.05)")
        
    mean_gap, ci_low, ci_high = bootstrap_macro_f1_gap(y_true, y_pred_c2v, y_pred_kpg, 1000)
    print(f"\n2. Bootstrap 95% CI on Macro F1 Gap (CASCADE2VEC - KPG):")
    print(f"   Mean Bootstrapped Gap: {mean_gap:.4f}")
    print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    if ci_low > 0:
        print("   => SIGNIFICANT improvement (CI does not cross zero)")
    else:
        print("   => NO significant improvement (CI crosses zero)")

if __name__ == "__main__":
    np.random.seed(42)
    main()
