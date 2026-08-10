"""
pgnn.py — Propagation Graph Neural Network for Rumour Detection.

Reference: Wu et al., 2021
  "Rumor Detection Based On Propagation Graph Neural Network With Attention Mechanism"

Implementation: Built from scratch using PyG MessagePassing (no official repo).
Deviations documented in docs/phase08_10_sota_baselines/phase08_10_implementation_notes.md.

Architecture:
  - 2-layer Gated Graph Propagation Network (GCN with LeakyReLU activation)
  - Attention-based graph pooling (learned attention scores over node embeddings)
  - 2-layer MLP classifier

Training protocol:
  - Fixed seed: SEED = 42
  - Test split accessed EXACTLY ONCE after training is complete
  - All early stopping uses VAL split exclusively

Run:
    PYTHONPATH=src python src/cascade2vec/phase08_10_sota_baselines/pgnn.py
"""

from __future__ import annotations

import os
import random
import time
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from adapters.pgnn_input import build_pgnn_data
from adapters.bigcn_input import fit_tfidf  # shared TF-IDF fitting util

import pickle

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
CKPT_FILE    = os.path.join(CKPT_DIR, "pgnn_best.pt")
TFIDF_FILE   = os.path.join(CKPT_DIR, "pgnn_tfidf.pkl")

# ── Hyperparameters ──────────────────────────────────────────────────────────
HP = {
    "model": "PGNN",
    "seed": SEED,
    "tfidf_max_features": 5000,
    "hidden_dim": 128,
    "attn_dim": 64,
    "mlp_hidden": 128,
    "dropout": 0.5,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 50,
    "batch_size": 64,
    "patience": 10,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss (class-weighted)",
    "num_gcn_layers": 2,
    "pooling": "attention",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pgnn")


# ── Model ────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Soft attention pooling over node embeddings within a graph."""

    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x, batch):
        # x: [total_nodes, hidden_dim], batch: [total_nodes]
        scores = self.attn(x)              # [total_nodes, 1]
        # Softmax per graph
        from torch_geometric.utils import softmax as pyg_softmax
        attn_weights = pyg_softmax(scores, batch)  # [total_nodes, 1]
        # Weighted sum per graph
        out = torch.zeros(batch.max().item() + 1, x.size(1), device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), attn_weights * x)
        return out


class PGNN(nn.Module):
    """
    2-layer GCN with attention pooling -> MLP classifier.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        attn_dim: int,
        mlp_hidden: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.attn_pool = AttentionPool(hidden_dim, attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, num_classes)

    def forward(self, x, edge_index, batch):
        h = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.2)
        h = self.dropout(h)
        h = F.leaky_relu(self.conv2(h, edge_index), negative_slope=0.2)
        h = self.attn_pool(h, batch)      # [B, hidden_dim]
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / (2.0 * counts + 1e-9)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out  = model(batch.x, batch.edge_index, batch.batch)
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
    return {"loss": total_loss / max(len(all_labels), 1),
            "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "accuracy": acc, "roc_auc": auc}


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
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
    log.info(f"[PGNN] Device: {DEVICE}")
    log.info(f"[PGNN] Hyperparameters: {HP}")

    df = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)
    cascade_split = dict(zip(split_df["cascade_id"], split_df["split"]))
    df["split"] = df["cascade_id"].map(cascade_split)

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    log.info("[PGNN] Fitting TF-IDF on training text only...")
    tfidf = fit_tfidf(train_df, max_features=HP["tfidf_max_features"])
    with open(TFIDF_FILE, "wb") as f:
        pickle.dump(tfidf, f)

    log.info("[PGNN] Building PyG datasets...")
    train_data = build_pgnn_data(train_df, tfidf)
    val_data   = build_pgnn_data(val_df,   tfidf)
    test_data  = build_pgnn_data(test_df,  tfidf)

    train_loader = PyGDataLoader(train_data, batch_size=HP["batch_size"], shuffle=True)
    val_loader   = PyGDataLoader(val_data,   batch_size=HP["batch_size"])
    test_loader  = PyGDataLoader(test_data,  batch_size=HP["batch_size"])

    train_labels  = [d.y.item() for d in train_data]
    class_weights = compute_class_weights(train_labels).to(DEVICE)

    in_dim = train_data[0].x.shape[1]
    model = PGNN(
        in_dim=in_dim, hidden_dim=HP["hidden_dim"], attn_dim=HP["attn_dim"],
        mlp_hidden=HP["mlp_hidden"], num_classes=2, dropout=HP["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=HP["lr"],
                                 weight_decay=HP["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_macro_f1, patience_counter = 0.0, 0
    log_rows = []
    t_start = time.time()

    for epoch in range(1, HP["epochs"] + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        row = {"epoch": epoch, "train_loss": round(train_loss, 4),
               **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()}}
        log_rows.append(row)
        log.info(f"[PGNN] Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                 f"val_macro_f1={val_metrics['macro_f1']:.4f} | val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), CKPT_FILE)
            patience_counter = 0
            log.info(f"[PGNN] *** Best val Macro F1: {best_val_macro_f1:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= HP["patience"]:
                log.info(f"[PGNN] Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t_start
    pd.DataFrame(log_rows).to_csv(os.path.join(LOG_DIR, "pgnn_training.csv"), index=False)

    # HARD RULE: test split accessed EXACTLY ONCE here
    model.load_state_dict(torch.load(CKPT_FILE, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    log.info("[PGNN] === FINAL TEST RESULTS ===")
    for k, v in test_metrics.items():
        log.info(f"  {k}: {v:.4f}")

    if test_metrics["macro_f1"] < 0.40:
        log.warning("[PGNN] WARNING: Test Macro F1 below 0.40 threshold.")

    results = {"model": "PGNN", "hyperparameters": HP,
               "best_val_macro_f1": best_val_macro_f1,
               "test_metrics": test_metrics, "runtime_minutes": round(elapsed / 60, 2), "seed": SEED}
    with open(os.path.join(LOG_DIR, "pgnn_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"[PGNN] Done in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
