"""
variance_check.py — Phase 11-12: Variance Check for Time Decay
============================================================
Runs the top 3 lambda configurations (0.0, 0.0001, 0.0005) across 5 random seeds (42, 43, 44, 45, 46).
Evaluates strictly on the val split.
"""
import numpy as np
import pandas as pd
import torch
import logging

from cascade2vec.phase11_12_cascade2vec.cascade2vec import (
    CASCADE2VEC, C2VClassifier, SnapshotDataset,
    train_c2v, evaluate_c2v
)
from cascade2vec.phase11_12_cascade2vec.sweep import (
    _build_tfidf, SWEEP_FIXED
)

logger = logging.getLogger(__name__)

UNIFIED_FILE   = "data/processed/phase02_ingestion/unified.parquet"
SPLIT_FILE     = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"

TOP_LAMBDAS = [0.0, 0.0001, 0.0005]
SEEDS = [42, 43, 44, 45, 46]

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    logger.info("Loading data...")
    unified = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)
    tfidf = _build_tfidf(unified, split_df)

    results = {l: [] for l in TOP_LAMBDAS}

    for lam in TOP_LAMBDAS:
        logger.info(f"=== Starting lambda={lam} ===")
        # Shared dataset (train/val) for this lambda
        train_ds = SnapshotDataset(unified, split_df, "train", tfidf, lam)
        val_ds   = SnapshotDataset(unified, split_df, "val",   tfidf, lam)

        for seed in SEEDS:
            logger.info(f"--- seed={seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            cfg = {
                "embed_dim": 128,  # Best embed_dim
                "n_layers": 1,     # Best n_layers
                "alpha": 0.5,      # Best alpha
                "lam": lam
            }
            
            from torch_geometric.loader import DataLoader
            
            # Create DataLoaders with a small generator for reproducibility
            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(train_ds, batch_size=SWEEP_FIXED["batch_size"], shuffle=True, generator=g, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=SWEEP_FIXED["batch_size"], shuffle=False, num_workers=0)
            
            encoder = CASCADE2VEC(
                in_dim=train_ds[0].x.shape[1],
                hidden_dim=SWEEP_FIXED["hidden_dim"],
                embed_dim=cfg["embed_dim"],
                n_layers=cfg["n_layers"],
                dropout=SWEEP_FIXED["dropout"],
            )
            classifier = C2VClassifier(
                embed_dim=cfg["embed_dim"], num_classes=2,
                dropout=SWEEP_FIXED["dropout"],
            )
            
            # Compute class weights
            train_sub = split_df[split_df["split"] == "train"]
            counts = train_sub["label"].value_counts()
            total = len(train_sub)
            cw = torch.tensor(
                [total / (2 * counts.get("non-rumour", 1)),
                 total / (2 * counts.get("rumour", 1))],
                dtype=torch.float32
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            train_result = train_c2v(
                encoder, classifier, train_loader, val_loader,
                lr=SWEEP_FIXED["lr"],
                weight_decay=SWEEP_FIXED["weight_decay"],
                n_epochs=SWEEP_FIXED["n_epochs"],
                patience=SWEEP_FIXED["patience"],
                alpha=cfg["alpha"],
                temperature=SWEEP_FIXED["temperature"],
                class_weights=cw,
                checkpoint_path=None
            )
            best_val_f1 = train_result["best_val_macro_f1"]
            
            results[lam].append(best_val_f1)
            logger.info(f"lam={lam} seed={seed} val_macro_f1={best_val_f1:.4f}")
    
    print("\n\n" + "="*50)
    print("VARIANCE CHECK RESULTS:")
    print("="*50)
    
    for lam in TOP_LAMBDAS:
        scores = results[lam]
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        print(f"Lambda = {lam}: Mean = {mean_val:.4f}, Std = {std_val:.4f} (Scores: {[round(x, 4) for x in scores]})")

    gap = np.mean(results[0.0]) - np.mean(results[0.0001])
    std_0 = np.std(results[0.0])
    print(f"\nMean gap (λ=0.0 - λ=0.0001): {gap:.4f}")
    
    if abs(gap) < std_0:
        print("DECISION: No significant difference detected between decay settings in this range.")
    else:
        print("DECISION: Significant difference detected.")

if __name__ == "__main__":
    main()
