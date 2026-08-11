"""
sweep.py — Phase 11-12: Hyperparameter Sweep for CASCADE2VEC
=============================================================
Sweeps: embedding_dim, lambda, n_layers, alpha (contrastive weight)

Uses val split exclusively. Test split NOT touched.
Logs all results to logs/phase11_12_cascade2vec/hyperparameter_sweep.md.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.loader import DataLoader

from cascade2vec.phase11_12_cascade2vec.cascade2vec import (
    CASCADE2VEC,
    C2VClassifier,
    SnapshotDataset,
    DEVICE,
    SEED,
    train_c2v,
    evaluate_c2v,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

SWEEP_GRID = {
    "embed_dim":  [32, 64, 128],
    "lam":        [0.0, 0.0001, 0.0005, 0.001],
    "n_layers":   [1, 2],
    "alpha":      [0.3, 0.5, 0.7],
}

# Fixed hyperparameters for sweep (not swept)
SWEEP_FIXED = {
    "hidden_dim":  128,
    "lr":          5e-4,
    "weight_decay": 1e-4,
    "n_epochs":    30,    # faster during sweep; full training uses 50
    "patience":    8,
    "batch_size":  64,
    "temperature": 0.07,
    "dropout":     0.5,
}

OUT_DIR   = "logs/phase11_12_cascade2vec"
CKPT_DIR  = "data/processed/phase11_12_cascade2vec/checkpoints"
SPLIT_FILE   = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"
UNIFIED_FILE = "data/processed/phase02_ingestion/unified.parquet"
SWEEP_LOG    = os.path.join(OUT_DIR, "hyperparameter_sweep.md")


def _build_tfidf(unified: pd.DataFrame, split_df: pd.DataFrame) -> TfidfVectorizer:
    """Fit TF-IDF on train cascade root tweets ONLY (no val/test leakage)."""
    train_ids = set(split_df[split_df["split"] == "train"]["cascade_id"])
    train_data = unified[unified["cascade_id"].isin(train_ids)]
    roots = (
        train_data[train_data["parent_id"].isna()]
        .groupby("cascade_id")["text"]
        .first()
        .fillna("")
        .tolist()
    )
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    vec.fit(roots)
    return vec


def _make_loaders(
    unified: pd.DataFrame,
    split_df: pd.DataFrame,
    tfidf: TfidfVectorizer,
    lam: float,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SnapshotDataset(unified, split_df, "train", tfidf, lam)
    val_ds   = SnapshotDataset(unified, split_df, "val",   tfidf, lam)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=g, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader


def run_sweep(force: bool = False) -> dict:
    """
    Run the full hyperparameter sweep.

    Returns the best config dict.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    if os.path.exists(SWEEP_LOG) and not force:
        raise RuntimeError(
            f"{SWEEP_LOG} already exists. Use --force to re-run the sweep."
        )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logging.basicConfig(level=logging.INFO)
    logger.info("[sweep] Loading data...")
    unified  = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)

    logger.info("[sweep] Building TF-IDF (train only)...")
    tfidf = _build_tfidf(unified, split_df)

    # Generate all configs
    keys   = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    logger.info("[sweep] Total configs: %d", len(configs))

    results: list[dict] = []
    best_val_f1 = -1.0
    best_config: dict[str, Any] = {}

    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        logger.info("[sweep] Config %d/%d: %s", i, len(configs), cfg)

        train_loader, val_loader = _make_loaders(
            unified, split_df, tfidf, cfg["lam"], SWEEP_FIXED["batch_size"]
        )

        in_dim = 5000  # TF-IDF features
        encoder = CASCADE2VEC(
            in_dim=in_dim,
            hidden_dim=SWEEP_FIXED["hidden_dim"],
            embed_dim=cfg["embed_dim"],
            n_layers=cfg["n_layers"],
            dropout=SWEEP_FIXED["dropout"],
        )
        classifier = C2VClassifier(
            embed_dim=cfg["embed_dim"], num_classes=2,
            dropout=SWEEP_FIXED["dropout"],
        )

        # Compute class weights from train split
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
        elapsed = round(time.time() - t0, 1)

        row = {**cfg, "val_macro_f1": round(val_f1, 4),
               "best_epoch": train_result["best_epoch"],
               "runtime_s": elapsed}
        results.append(row)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = dict(cfg)
            logger.info("[sweep] *** New best: val_macro_f1=%.4f ***", val_f1)

    # Write sweep log
    _write_sweep_log(results, best_config, SWEEP_LOG)
    return best_config


def _write_sweep_log(results: list[dict], best_config: dict, path: str):
    lines = [
        "# Phase 11-12: Hyperparameter Sweep Results",
        "",
        f"Total configs evaluated: {len(results)}",
        f"Fixed hyperparameters: {json.dumps(SWEEP_FIXED, indent=2)}",
        "",
        "## Best Config",
        "",
        f"```json\n{json.dumps(best_config, indent=2)}\n```",
        "",
        "## Full Results (sorted by val Macro F1 desc)",
        "",
        "| embed_dim | lam | n_layers | alpha | val_macro_f1 | best_epoch | runtime_s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: x["val_macro_f1"], reverse=True):
        lines.append(
            f"| {r['embed_dim']} | {r['lam']} | {r['n_layers']} | {r['alpha']} "
            f"| **{r['val_macro_f1']}** | {r['best_epoch']} | {r['runtime_s']}s |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("[sweep] Sweep log written to %s", path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing sweep log")
    args = parser.parse_args()
    run_sweep(force=args.force)
