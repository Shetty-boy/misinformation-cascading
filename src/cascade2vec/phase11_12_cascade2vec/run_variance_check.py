import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch

from cascade2vec.phase11_12_cascade2vec.cascade2vec import (
    CASCADE2VEC,
    C2VClassifier,
    DEVICE,
    train_c2v,
)
from cascade2vec.phase11_12_cascade2vec.sweep import (
    _build_tfidf,
    _make_loaders,
    SPLIT_FILE,
    UNIFIED_FILE,
    SWEEP_FIXED,
)

logger = logging.getLogger(__name__)

# The configs to check
CONFIGS = [
    {"name": "best_overall", "embed_dim": 32, "lam": 0.0005, "n_layers": 2, "alpha": 0.5},
    {"name": "best_zero_decay", "embed_dim": 128, "lam": 0.0, "n_layers": 2, "alpha": 0.3},
    {"name": "third_overall", "embed_dim": 64, "lam": 0.001, "n_layers": 2, "alpha": 0.7},
]

SEEDS = [42, 43, 44, 45, 46]

OUT_DIR = "logs/phase11_12_cascade2vec"
OUT_FILE = os.path.join(OUT_DIR, "variance_check.json")


def run_variance_check():
    logging.basicConfig(level=logging.INFO)
    logger.info("[variance_check] Loading data...")
    unified = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)

    logger.info("[variance_check] Building TF-IDF (train only)...")
    tfidf = _build_tfidf(unified, split_df)

    train_sub = split_df[split_df["split"] == "train"]
    counts = train_sub["label"].value_counts()
    total = len(train_sub)
    cw = torch.tensor(
        [total / (2 * counts.get("non-rumour", 1)),
         total / (2 * counts.get("rumour", 1))],
        dtype=torch.float32, device=DEVICE,
    )

    results = []

    for cfg in CONFIGS:
        logger.info(f"=== Config: {cfg['name']} ===")
        cfg_f1s = []

        # Note: lam affects data loading, so we must load per lam
        train_loader, val_loader = _make_loaders(
            unified, split_df, tfidf, cfg["lam"], SWEEP_FIXED["batch_size"]
        )
        
        for seed in SEEDS:
            logger.info(f"  Seed {seed}...")
            # Set all seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            in_dim = 5000  # TF-IDF features
            encoder = CASCADE2VEC(
                in_dim=in_dim,
                hidden_dim=SWEEP_FIXED["hidden_dim"],
                embed_dim=cfg["embed_dim"],
                n_layers=cfg["n_layers"],
                dropout=SWEEP_FIXED["dropout"],
            ).to(DEVICE)
            classifier = C2VClassifier(
                embed_dim=cfg["embed_dim"], num_classes=2,
                dropout=SWEEP_FIXED["dropout"],
            ).to(DEVICE)

            train_result = train_c2v(
                encoder, classifier, train_loader, val_loader,
                n_epochs=SWEEP_FIXED["n_epochs"],
                lr=SWEEP_FIXED["lr"],
                weight_decay=SWEEP_FIXED["weight_decay"],
                alpha=cfg["alpha"],
                temperature=SWEEP_FIXED["temperature"],
                patience=SWEEP_FIXED["patience"],
                class_weights=cw,
                device=DEVICE,
            )
            val_f1 = train_result["best_val_macro_f1"]
            cfg_f1s.append(val_f1)
            logger.info(f"    -> Val F1: {val_f1:.4f}")

        mean_f1 = np.mean(cfg_f1s)
        std_f1 = np.std(cfg_f1s)
        logger.info(f"  => Mean: {mean_f1:.4f}, Std: {std_f1:.4f}")

        results.append({
            "config": cfg,
            "seeds": SEEDS,
            "f1s": cfg_f1s,
            "mean": float(mean_f1),
            "std": float(std_f1),
        })

        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    logger.info("Done. Results saved to %s", OUT_FILE)


if __name__ == "__main__":
    run_variance_check()
