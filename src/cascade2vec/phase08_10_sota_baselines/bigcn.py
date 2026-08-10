"""
bigcn.py — Bi-Directional Graph Convolutional Network for Rumour Detection.

Reference: Bian et al., AAAI 2020
  "Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks"
  
Source: Adapted from safe-graph/GNN-FakeNews (MIT license, updated Dec 2025).
  Deviations documented in docs/phase08_10_sota_baselines/implementation_notes.md.

Training protocol:
  - Fixed seed: SEED = 42
  - Test split accessed EXACTLY ONCE after training is complete
  - All early stopping and model selection uses VAL split exclusively
  - Best checkpoint selected by val Macro F1 (not val loss)
  
Run:
    python src/cascade2vec/phase08_10_sota_baselines/bigcn.py
    
Outputs:
    data/processed/phase08_10_sota_baselines/checkpoints/bigcn_best.pt
    logs/phase08_10_sota_baselines/bigcn_training.log
"""

from __future__ import annotations

import os
import random
import time
import json
import logging
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from adapters.bigcn_input import build_bigcn_data, fit_tfidf

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Paths ────────────────────────────────────────────────────────────────────
SPLIT_FILE   = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"
UNIFIED_FILE = "data/processed/phase02_ingestion/unified.parquet"
CKPT_DIR     = "data/processed/phase08_10_sota_baselines/checkpoints"
LOG_DIR      = "logs/phase08_10_sota_baselines"
CKPT_FILE    = os.path.join(CKPT_DIR, "bigcn_best.pt")
TFIDF_FILE   = os.path.join(CKPT_DIR, "bigcn_tfidf.pkl")

# ── Hyperparameters ──────────────────────────────────────────────────────────
HP = {
    "model": "BiGCN",
    "seed": SEED,
    "tfidf_max_features": 5000,
    "hidden_dim": 128,
    "dropout": 0.5,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 50,
    "batch_size": 64,
    "patience": 10,          # early stopping on val Macro F1
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss (class-weighted)",
    "pooling": "global_mean_pool",
    "num_gcn_layers": 2,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {"non-rumour": 0, "rumour": 1}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bigcn")


# ── Model ────────────────────────────────────────────────────────────────────
class BiGCN(nn.Module):
    """
    Two-branch GCN: one on the top-down (TD) graph, one on bottom-up (BU) graph.
    Concatenate pooled representations → linear classifier.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        # Top-down branch
        self.td_conv1 = GCNConv(in_dim, hidden_dim)
        self.td_conv2 = GCNConv(hidden_dim, hidden_dim)
        # Bottom-up branch
        self.bu_conv1 = GCNConv(in_dim, hidden_dim)
        self.bu_conv2 = GCNConv(hidden_dim, hidden_dim)
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, edge_index_td, edge_index_bu, batch):
        # TD branch
        h_td = F.relu(self.td_conv1(x, edge_index_td))
        h_td = self.dropout(h_td)
        h_td = F.relu(self.td_conv2(h_td, edge_index_td))
        h_td = global_mean_pool(h_td, batch)  # [B, hidden_dim]

        # BU branch
        h_bu = F.relu(self.bu_conv1(x, edge_index_bu))
        h_bu = self.dropout(h_bu)
        h_bu = F.relu(self.bu_conv2(h_bu, edge_index_bu))
        h_bu = global_mean_pool(h_bu, batch)  # [B, hidden_dim]

        out = torch.cat([h_td, h_bu], dim=-1)  # [B, 2*hidden_dim]
        out = self.dropout(out)
        return self.fc(out)


# ── Custom collate for BiGCN (handles edge_index_bu extra field) ─────────────
from torch_geometric.data import Batch as PyGBatch


def bigcn_collate(data_list: list[Data]):
    """Batch Data objects, stacking edge_index_bu with corrected offsets."""
    batch = PyGBatch.from_data_list(data_list)
    # Manually stack edge_index_bu with the same offsets as edge_index
    bu_edges = []
    offset = 0
    for d in data_list:
        bu_edges.append(d.edge_index_bu + offset)
        offset += d.num_nodes
    if bu_edges:
        batch.edge_index_bu = torch.cat(bu_edges, dim=1)
    else:
        batch.edge_index_bu = torch.zeros((2, 0), dtype=torch.long)
    return batch


# ── Training helpers ─────────────────────────────────────────────────────────

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse frequency class weighting."""
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / (2.0 * counts + 1e-9)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model: BiGCN, loader, criterion, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_index_bu, batch.batch)
            loss = criterion(out, batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs
            probs = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=-1).cpu().numpy()
            labs  = batch.y.squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)
            all_probs.extend(probs)

    macro_f1    = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc         = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    avg_loss = total_loss / max(len(all_labels), 1)
    return {"loss": avg_loss, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "accuracy": acc, "roc_auc": auc}


def train_epoch(model: BiGCN, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_index_bu, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / max(len(loader.dataset), 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log.info(f"[BiGCN] Device: {DEVICE}")
    log.info(f"[BiGCN] Hyperparameters: {HP}")

    # ── Load data ────────────────────────────────────────────────────────────
    log.info("[BiGCN] Loading unified dataset...")
    df = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)

    # Merge split info
    cascade_split = dict(zip(split_df["cascade_id"], split_df["split"]))
    df["split"] = df["cascade_id"].map(cascade_split)

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    log.info(f"[BiGCN] Train cascades: {train_df['cascade_id'].nunique()} | "
             f"Val: {val_df['cascade_id'].nunique()} | Test: {test_df['cascade_id'].nunique()}")

    # ── Fit TF-IDF on TRAIN only ─────────────────────────────────────────────
    log.info("[BiGCN] Fitting TF-IDF on training text...")
    tfidf = fit_tfidf(train_df, max_features=HP["tfidf_max_features"])
    with open(TFIDF_FILE, "wb") as f:
        pickle.dump(tfidf, f)
    log.info(f"[BiGCN] TF-IDF vocab size: {len(tfidf.vocabulary_)}")

    # ── Build PyG datasets ────────────────────────────────────────────────────
    log.info("[BiGCN] Building PyG datasets...")
    train_data = build_bigcn_data(train_df, tfidf)
    val_data   = build_bigcn_data(val_df,   tfidf)
    test_data  = build_bigcn_data(test_df,  tfidf)

    log.info(f"[BiGCN] Graphs — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def make_loader(data, shuffle):
        return PyGDataLoader(
            data, batch_size=HP["batch_size"], shuffle=shuffle,
            collate_fn=bigcn_collate,
            worker_init_fn=lambda wid: np.random.seed(SEED + wid),
        )

    train_loader = make_loader(train_data, shuffle=True)
    val_loader   = make_loader(val_data,   shuffle=False)
    test_loader  = make_loader(test_data,  shuffle=False)

    # ── Class weights ─────────────────────────────────────────────────────────
    train_labels = [d.y.item() for d in train_data]
    class_weights = compute_class_weights(train_labels).to(DEVICE)
    log.info(f"[BiGCN] Class weights: {class_weights.tolist()}")

    # ── Model, optimizer, criterion ───────────────────────────────────────────
    in_dim = train_data[0].x.shape[1]
    model = BiGCN(
        in_dim=in_dim,
        hidden_dim=HP["hidden_dim"],
        num_classes=2,
        dropout=HP["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_macro_f1 = 0.0
    patience_counter  = 0
    log_rows          = []
    t_start           = time.time()

    log.info("[BiGCN] Starting training...")
    for epoch in range(1, HP["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        }
        log_rows.append(row)
        log.info(
            f"[BiGCN] Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        # Best checkpoint on val Macro F1
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), CKPT_FILE)
            patience_counter = 0
            log.info(f"[BiGCN] *** New best val Macro F1: {best_val_macro_f1:.4f} — checkpoint saved ***")
        else:
            patience_counter += 1
            if patience_counter >= HP["patience"]:
                log.info(f"[BiGCN] Early stopping at epoch {epoch} (patience={HP['patience']})")
                break

    elapsed = time.time() - t_start
    log.info(f"[BiGCN] Training complete in {elapsed/60:.1f} min. Best val Macro F1: {best_val_macro_f1:.4f}")

    # Save training log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(LOG_DIR, "bigcn_training.csv"), index=False)

    # ── Final test evaluation (ONCE only, after training complete) ────────────
    # HARD RULE: test split is used EXACTLY ONCE, after training is complete.
    # All early stopping and model selection above used val split exclusively.
    log.info("[BiGCN] Loading best checkpoint for final test evaluation...")
    model.load_state_dict(torch.load(CKPT_FILE, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    log.info("[BiGCN] === FINAL TEST RESULTS ===")
    for k, v in test_metrics.items():
        log.info(f"  {k}: {v:.4f}")

    # Check Macro F1 > 0.40 threshold
    if test_metrics["macro_f1"] < 0.40:
        log.warning(
            f"[BiGCN] WARNING: Test Macro F1 ({test_metrics['macro_f1']:.4f}) is below "
            f"the Phase 6-7 simple baseline threshold of 0.40. Review results before "
            f"adding to comparison table."
        )

    # Save results summary
    results = {
        "model": "BiGCN",
        "hyperparameters": HP,
        "best_val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics,
        "runtime_minutes": round(elapsed / 60, 2),
        "seed": SEED,
    }
    results_path = os.path.join(LOG_DIR, "bigcn_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"[BiGCN] Results saved to {results_path}")


if __name__ == "__main__":
    main()
