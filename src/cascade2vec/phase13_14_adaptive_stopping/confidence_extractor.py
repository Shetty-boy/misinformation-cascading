import os
import pickle
import logging
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from cascade2vec.phase11_12_cascade2vec.cascade2vec import (
    CASCADE2VEC, C2VClassifier, SnapshotDataset, DEVICE
)
from cascade2vec.phase11_12_cascade2vec.run_c2v import _load_best_config
from cascade2vec.phase11_12_cascade2vec.sweep import _build_tfidf

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../phase08_10_sota_baselines"))
from cascade2vec.phase08_10_sota_baselines.kpg import KPGSimplified
from cascade2vec.phase08_10_sota_baselines.adapters.kpg_input import build_kpg_data
from cascade2vec.phase02_ingestion.leakage_audit import assert_snapshot_is_clean

logger = logging.getLogger(__name__)

UNIFIED_FILE   = "data/processed/phase02_ingestion/unified.parquet"
SPLIT_FILE     = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"
FM_FILE        = "data/processed/phase06_07_features/feature_matrix.parquet"

C2V_CKPT       = "data/processed/phase11_12_cascade2vec/checkpoints/final_model.pt"
KPG_CKPT       = "data/processed/phase08_10_sota_baselines/checkpoints/kpg_best.pt"
KPG_TFIDF      = "data/processed/phase08_10_sota_baselines/checkpoints/kpg_tfidf.pkl"

OUT_DIR        = "data/processed/phase13_14_adaptive_stopping"
OUT_FILE       = os.path.join(OUT_DIR, "confidence_features.parquet")
TIME_WINDOWS   = [1, 2, 5, 10, 15, 30, 60, 120]


def extract_c2v_confidences(unified: pd.DataFrame, split_df: pd.DataFrame, batch_size=64) -> pd.DataFrame:
    best_cfg = _load_best_config()
    tfidf = _build_tfidf(unified, split_df)
    
    # Load model
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
    
    rows = []
    for split_name in ["train", "val", "test"]:
        ds = SnapshotDataset(unified, split_df, split_name, tfidf, best_cfg["lam"])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                z = encoder(batch.x, batch.edge_index, batch.edge_attr if batch.edge_attr.numel() > 0 else None, batch.batch)
                logits = classifier(z)
                probs = F.softmax(logits, dim=-1)
                
                # prob_rumour is the 1th index (assuming non-rumour=0, rumour=1)
                for i in range(batch.num_graphs):
                    cid = batch.cascade_id[i] if hasattr(batch.cascade_id, '__getitem__') else batch.cascade_id
                    t_min = int(batch.t_minutes[i].item()) if hasattr(batch.t_minutes, '__getitem__') else int(batch.t_minutes)
                    prob_rumour = float(probs[i, 1].item())
                    rows.append({
                        "cascade_id": cid,
                        "t_minutes": t_min,
                        "confidence_c2v": prob_rumour
                    })
    return pd.DataFrame(rows)


def extract_kpg_confidences(unified: pd.DataFrame, split_df: pd.DataFrame, batch_size=64) -> pd.DataFrame:
    with open(KPG_TFIDF, "rb") as f:
        kpg_tfidf = pickle.load(f)
        
    model = KPGSimplified(in_dim=5000, hidden_dim=128, mlp_hidden=128, dropout=0.5, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(KPG_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    
    cascade_split = dict(zip(split_df["cascade_id"], split_df["split"]))
    valid_cids = set(cascade_split.keys())
    
    rows = []
    
    for t_min in TIME_WINDOWS:
        t_sec = t_min * 60.0
        # Subset to time window
        snap_df = unified[unified["timestamp"] <= t_sec].copy()
        snap_df = snap_df[snap_df["cascade_id"].isin(valid_cids)]
        
        if snap_df.empty:
            continue
            
        # Build KPG data
        data_list = build_kpg_data(snap_df, kpg_tfidf, k=20)
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(logits, dim=-1)
                
                for i in range(batch.num_graphs):
                    cid = batch.cascade_id[i] if hasattr(batch.cascade_id, '__getitem__') else batch.cascade_id
                    prob_rumour = float(probs[i, 1].item())
                    rows.append({
                        "cascade_id": cid,
                        "t_minutes": t_min,
                        "confidence_kpg": prob_rumour
                    })
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    logger.info("Loading unified and split data...")
    unified = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)
    fm_df = pd.read_parquet(FM_FILE)
    
    logger.info("Extracting CASCADE2VEC confidences...")
    c2v_df = extract_c2v_confidences(unified, split_df)
    
    logger.info("Extracting KPG-simplified confidences...")
    kpg_df = extract_kpg_confidences(unified, split_df)
    
    logger.info("Merging confidences with feature matrix...")
    # Base feature matrix already has all (cascade_id, t_minutes) combinations
    res_df = fm_df.merge(c2v_df, on=["cascade_id", "t_minutes"], how="inner")
    res_df = res_df.merge(kpg_df, on=["cascade_id", "t_minutes"], how="inner")
    
    # Also attach the split info for downstream threshold training
    # drop split if already exists in fm_df to avoid _x _y suffixes
    if "split" in res_df.columns:
        res_df.drop(columns=["split"], inplace=True)
    res_df = res_df.merge(split_df[["cascade_id", "split"]], on="cascade_id", how="inner")
    
    logger.info(f"Final confidence features shape: {res_df.shape}")
    res_df.to_parquet(OUT_FILE, index=False)
    logger.info(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
