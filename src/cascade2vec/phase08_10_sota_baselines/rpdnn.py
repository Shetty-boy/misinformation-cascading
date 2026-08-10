"""
rpdnn.py — RP-DNN: Rumor Propagation Deep Neural Network for Rumour Detection.

Reference: Bian et al., 2020
  "RP-DNN: A Tweet Level Propagation Context Based Deep Neural Networks
   for Early Rumor Detection in Social Media"

Implementation: Built from scratch (no official PyTorch repo available).
Deviations documented in docs/phase08_10_sota_baselines/implementation_notes.md.

Architecture:
  - Branch 1: BiGRU over root tweet token sequence -> pooled text embedding
  - Branch 2: GRU over BFS-depth structural feature sequence -> pooled struct embedding
  - Concatenate -> 2-layer MLP -> Binary classifier

Training protocol:
  - Fixed seed: SEED = 42
  - Test split accessed EXACTLY ONCE after training is complete
  - All early stopping uses VAL split exclusively

Run:
    PYTHONPATH=src python src/cascade2vec/phase08_10_sota_baselines/rpdnn.py
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from adapters.rpdnn_input import build_vocabulary, build_rpdnn_sequences

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
CKPT_FILE    = os.path.join(CKPT_DIR, "rpdnn_best.pt")
VOCAB_FILE   = os.path.join(CKPT_DIR, "rpdnn_vocab.pkl")

# ── Hyperparameters ──────────────────────────────────────────────────────────
HP = {
    "model": "RP-DNN",
    "seed": SEED,
    "vocab_size": 10000,
    "embed_dim": 64,
    "hidden_dim": 128,
    "struct_dim": 2,
    "struct_hidden": 64,
    "mlp_hidden": 128,
    "dropout": 0.5,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "batch_size": 128,
    "patience": 10,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss (class-weighted)",
    "max_len": 128,
    "max_depth": 30,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("rpdnn")


# ── Dataset ──────────────────────────────────────────────────────────────────

class RPDNNDataset(Dataset):
    def __init__(self, sequences: list[tuple]):
        self.data = sequences  # (text_tokens, struct_seq, label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_tokens, struct_seq, label = self.data[idx]
        return (
            torch.tensor(text_tokens, dtype=torch.long),
            torch.tensor(struct_seq, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ── Model ────────────────────────────────────────────────────────────────────

class RPDNN(nn.Module):
    """
    Two-branch model:
      1. BiGRU over text token embeddings (root tweet)
      2. GRU over structural depth-level feature sequence
    Concatenate -> MLP -> classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        struct_dim: int,
        struct_hidden: int,
        mlp_hidden: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=0)
        # Text branch: BiGRU
        self.text_gru = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        # Structural branch: GRU
        self.struct_gru = nn.GRU(
            struct_dim, struct_hidden, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        # MLP classifier
        in_dim = hidden_dim * 2 + struct_hidden
        self.fc1 = nn.Linear(in_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, num_classes)

    def forward(self, text_tokens, struct_seq):
        # Text branch
        emb = self.embedding(text_tokens)            # [B, L, embed_dim]
        _, h = self.text_gru(emb)                    # h: [2, B, hidden_dim] (bidirectional)
        h_text = torch.cat([h[0], h[1]], dim=-1)     # [B, 2*hidden_dim]

        # Structural branch
        _, h_struct = self.struct_gru(struct_seq)    # h_struct: [1, B, struct_hidden]
        h_struct = h_struct.squeeze(0)               # [B, struct_hidden]

        x = torch.cat([h_text, h_struct], dim=-1)   # [B, 2*hidden_dim + struct_hidden]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / (2.0 * counts + 1e-9)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for text_t, struct_t, labels in loader:
            text_t  = text_t.to(device)
            struct_t = struct_t.to(device)
            labels   = labels.to(device)
            out  = model(text_t, struct_t)
            loss = criterion(out, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
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


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for text_t, struct_t, labels in loader:
        text_t   = text_t.to(device)
        struct_t = struct_t.to(device)
        labels   = labels.to(device)
        optimizer.zero_grad()
        out  = model(text_t, struct_t)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / max(len(loader.dataset), 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log.info(f"[RP-DNN] Device: {DEVICE}")
    log.info(f"[RP-DNN] Hyperparameters: {HP}")

    df = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)
    cascade_split = dict(zip(split_df["cascade_id"], split_df["split"]))
    df["split"] = df["cascade_id"].map(cascade_split)

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    log.info("[RP-DNN] Building vocabulary on training text only...")
    word2idx = build_vocabulary(train_df, max_vocab=HP["vocab_size"])
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump(word2idx, f)

    log.info("[RP-DNN] Building sequences...")
    train_seqs = build_rpdnn_sequences(train_df, word2idx, HP["max_len"], HP["max_depth"])
    val_seqs   = build_rpdnn_sequences(val_df,   word2idx, HP["max_len"], HP["max_depth"])
    test_seqs  = build_rpdnn_sequences(test_df,  word2idx, HP["max_len"], HP["max_depth"])

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(RPDNNDataset(train_seqs), batch_size=HP["batch_size"],
                              shuffle=True, generator=g)
    val_loader   = DataLoader(RPDNNDataset(val_seqs),   batch_size=HP["batch_size"])
    test_loader  = DataLoader(RPDNNDataset(test_seqs),  batch_size=HP["batch_size"])

    train_labels = [s[2] for s in train_seqs]
    class_weights = compute_class_weights(train_labels).to(DEVICE)

    model = RPDNN(
        vocab_size=HP["vocab_size"], embed_dim=HP["embed_dim"],
        hidden_dim=HP["hidden_dim"], struct_dim=HP["struct_dim"],
        struct_hidden=HP["struct_hidden"], mlp_hidden=HP["mlp_hidden"],
        num_classes=2, dropout=HP["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=HP["lr"],
                                 weight_decay=HP["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_macro_f1 = 0.0
    patience_counter  = 0
    log_rows          = []
    t_start           = time.time()

    for epoch in range(1, HP["epochs"] + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        row = {"epoch": epoch, "train_loss": round(train_loss, 4),
               **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()}}
        log_rows.append(row)
        log.info(f"[RP-DNN] Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                 f"val_macro_f1={val_metrics['macro_f1']:.4f} | val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), CKPT_FILE)
            patience_counter = 0
            log.info(f"[RP-DNN] *** New best val Macro F1: {best_val_macro_f1:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= HP["patience"]:
                log.info(f"[RP-DNN] Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t_start
    pd.DataFrame(log_rows).to_csv(os.path.join(LOG_DIR, "rpdnn_training.csv"), index=False)

    # HARD RULE: test split accessed EXACTLY ONCE here
    model.load_state_dict(torch.load(CKPT_FILE, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    log.info("[RP-DNN] === FINAL TEST RESULTS ===")
    for k, v in test_metrics.items():
        log.info(f"  {k}: {v:.4f}")

    if test_metrics["macro_f1"] < 0.40:
        log.warning(f"[RP-DNN] WARNING: Test Macro F1 below 0.40 threshold.")

    results = {"model": "RP-DNN", "hyperparameters": HP,
               "best_val_macro_f1": best_val_macro_f1,
               "test_metrics": test_metrics, "runtime_minutes": round(elapsed / 60, 2), "seed": SEED}
    with open(os.path.join(LOG_DIR, "rpdnn_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"[RP-DNN] Done in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
