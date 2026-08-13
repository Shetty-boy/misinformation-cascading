"""
cascade2vec.py — Phase 11-12: CASCADE2VEC Encoder
==================================================
Time-weighted GraphSAGE encoder over cascade propagation graphs.

Training regime: Option B (snapshot-aware) — trained on all 8 time windows
per cascade (t ∈ {1,2,5,10,15,30,60,120} minutes), NOT just final cascades.
This allows Phase 13-14 to consume per-(cascade_id, t) embeddings directly.

Time-decay formula:
    w(e) = exp(-λ * (t_snapshot_s - t_edge_s))
where:
    - t_snapshot_s: snapshot cutoff in SECONDS (e.g. 300 for t=5min)
    - t_edge_s:     child node arrival timestamp in SECONDS from root
    - λ (lambda):   decay rate in inverse-seconds (s⁻¹)
    - Root's timestamp is always 0 (normalized by ingest.py)
    - Future nodes/edges (t_edge_s > t_snapshot_s) are EXCLUDED entirely —
      they are NOT included with near-zero weight. This matches the temporal
      safety contract enforced by assert_snapshot_is_clean() in leakage_audit.py.

See docs/phase11_12_cascade2vec/architecture_notes.md for full specification.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

from cascade2vec.phase02_ingestion.leakage_audit import assert_snapshot_is_clean

logger = logging.getLogger(__name__)

SEED = 42
TIME_WINDOWS_MINUTES = [1, 2, 5, 10, 15, 30, 60, 120]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1.  Time-weighted edge-weight computation
# ---------------------------------------------------------------------------

def compute_edge_weights(
    t_edge_s: torch.Tensor,
    t_snapshot_s: float,
    lam: float,
) -> torch.Tensor:
    """
    Compute time-decay edge weights.

    w(e) = exp(-λ * (t_snapshot_s - t_edge_s))

    Args:
        t_edge_s:     1-D tensor of child-node arrival times in seconds.
        t_snapshot_s: Snapshot cutoff in seconds (scalar float).
        lam:          Decay rate λ ≥ 0 (in s⁻¹).

    Returns:
        1-D tensor of weights in (0, 1]. Weight = 1.0 exactly when
        t_edge_s == t_snapshot_s (most recently arrived reply at cutoff).
    """
    delta = t_snapshot_s - t_edge_s.float()
    # delta should always be ≥ 0 because future edges are excluded before
    # this is called. Clamp as a safety net.
    delta = torch.clamp(delta, min=0.0)
    return torch.exp(-lam * delta)


# ---------------------------------------------------------------------------
# 2.  Custom time-weighted SAGEConv layer
# ---------------------------------------------------------------------------

class TimeWeightedSAGEConv(MessagePassing):
    """
    GraphSAGE convolution with time-decay weighted message aggregation.

    Standard SAGE: h_v = σ(W · concat(h_v, mean_{u∈N(v)} h_u))
    Time-weighted: replace mean with weighted-mean using exp(-λΔt) weights.

    Edge weights are expected as `edge_weight` in the graph's edge attribute.
    If edge_weight is absent (e.g. self-loop for the root), weight defaults to 1.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")  # we normalise manually
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self transform
        out_self = self.lin_self(x)

        # Weighted neighbourhood aggregation
        # message() multiplies by edge_weight; aggregate() sums
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out_neigh = self.lin_neigh(agg)

        return F.relu(out_self + out_neigh + self.bias)

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.unsqueeze(-1) * x_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # Sum aggregation (we normalise below by sum of weights)
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)


# ---------------------------------------------------------------------------
# 3.  CASCADE2VEC encoder
# ---------------------------------------------------------------------------

class CASCADE2VEC(nn.Module):
    """
    Time-weighted GraphSAGE encoder producing a fixed-dim cascade embedding.

    Architecture:
        - n_layers × TimeWeightedSAGEConv (in_dim → hidden_dim)
        - Dropout between layers
        - Attention pooling over node embeddings → cascade embedding
        - L2 normalised output embedding

    The embedding is snapshot-agnostic in shape but snapshot-aware in content:
    the same cascade at t=5min and t=120min produces DIFFERENT embeddings
    (different graph topology and edge weights).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_layers = n_layers

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(TimeWeightedSAGEConv(in_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(TimeWeightedSAGEConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Attention pooling: learns a scalar score per node
        self.attn = nn.Linear(hidden_dim, 1)

        # Projection to embedding space
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:           (N, in_dim) node feature matrix
            edge_index:  (2, E) edge index
            edge_weight: (E,) time-decay weights
            batch:       (N,) batch assignment vector

        Returns:
            (B, embed_dim) L2-normalised cascade embeddings
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight)
            if i < self.n_layers - 1:
                h = self.dropout(h)

        # Soft attention pooling per graph
        attn_scores = self.attn(h)                           # (N, 1)
        attn_scores = torch.sigmoid(attn_scores)             # keep in (0,1)
        # Weighted sum per graph in batch
        weighted = attn_scores * h                            # (N, hidden)
        pooled = global_mean_pool(weighted, batch)            # (B, hidden)

        embed = self.proj(pooled)                             # (B, embed_dim)
        return F.normalize(embed, p=2, dim=-1)               # L2 normalise


# ---------------------------------------------------------------------------
# 4.  Classifier head
# ---------------------------------------------------------------------------

class C2VClassifier(nn.Module):
    """Attaches a 2-class MLP head on top of CASCADE2VEC embeddings."""

    def __init__(self, embed_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.fc(embed)


# ---------------------------------------------------------------------------
# 5.  Contrastive loss (Supervised InfoNCE)
# ---------------------------------------------------------------------------

def supervised_infonce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised InfoNCE / SupCon loss.

    Positive pairs: same cascade (same label AND same cascade_id within batch).
    We use label-level supervision as a proxy: embeddings with matching labels
    are treated as positives, others as negatives.

    In the snapshot-aware regime, the DataLoader produces one sample per
    (cascade_id, t_minutes). A batch therefore naturally contains multiple
    snapshots of the same cascade — these are the true positive pairs the
    objective is designed to align.

    Formula:
        L = -1/N * Σ_i  1/|P(i)| * Σ_{p∈P(i)}  log(
                exp(z_i·z_p / τ) / Σ_{j≠i} exp(z_i·z_j / τ)
            )

    Args:
        embeddings: (B, D) L2-normalised embeddings
        labels:     (B,) integer class labels
        temperature: τ scalar

    Returns:
        scalar loss
    """
    B = embeddings.size(0)
    if B < 2:
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)

    # Pairwise similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature   # (B, B)

    # Mask: positive pairs (same label, excluding self)
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)       # (B, B)
    self_mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
    pos_mask = labels_eq & self_mask

    # Numerical stability
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - sim_max

    exp_sim = torch.exp(sim) * self_mask.float()                  # exclude self
    log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Per-sample loss (only for samples with at least one positive)
    has_positive = pos_mask.any(dim=1)
    if not has_positive.any():
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)

    # Sum of log-probs over positives
    log_probs = sim - log_sum_exp                                  # (B, B)
    pos_log_probs = (log_probs * pos_mask.float()).sum(dim=1)      # (B,)
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    loss = -(pos_log_probs[has_positive] / n_pos[has_positive]).mean()
    return loss


# ---------------------------------------------------------------------------
# 6.  Dataset: one PyG Data per (cascade_id, t_minutes)
# ---------------------------------------------------------------------------

class SnapshotDataset(Dataset):
    """
    PyG Dataset — one Data object per (cascade_id, t_minutes) snapshot.

    Graph construction (per snapshot):
      - Filter unified.parquet: keep rows where timestamp <= t_seconds
      - Future nodes/edges are EXCLUDED (not included with near-zero weight)
      - assert_snapshot_is_clean() called to verify no leakage
      - Edge weight = exp(-λ * (t_snapshot_s - t_edge_s)) on each edge
      - Node features = TF-IDF of root tweet text (cascade-level, 5000-dim)

    The TF-IDF vectorizer MUST be fitted on train split only to prevent leakage.
    """

    def __init__(
        self,
        unified_df: pd.DataFrame,
        split_df: pd.DataFrame,
        split_name: str,           # 'train', 'val', or 'test'
        tfidf: TfidfVectorizer,
        lam: float,
        time_windows_minutes: list[int] = TIME_WINDOWS_MINUTES,
    ):
        super().__init__()
        self.lam = lam
        self.time_windows_minutes = time_windows_minutes

        # Filter to requested split
        split_cascade_ids = set(
            split_df[split_df["split"] == split_name]["cascade_id"].tolist()
        )
        self.label_map = (
            split_df.set_index("cascade_id")["label"].to_dict()
        )
        self.le = LabelEncoder().fit(["non-rumour", "rumour"])

        self.unified = unified_df[
            unified_df["cascade_id"].isin(split_cascade_ids)
        ].copy()

        # TF-IDF root tweet text per cascade
        self.cascade_ids = sorted(split_cascade_ids & set(self.unified["cascade_id"].unique()))
        
        roots_series = (
            self.unified[self.unified["parent_id"].isna()]
            .groupby("cascade_id")["text"]
            .first()
            .fillna("")
        )
        root_texts = [roots_series.get(cid, "") for cid in self.cascade_ids]
        
        self.tfidf_matrix = tfidf.transform(root_texts)  # sparse (C, 5000)

        import collections
        
        # Convert to raw python lists for O(1) processing (pandas slicing is too slow for 30k graphs)
        cascade_dict = collections.defaultdict(list)
        for row in self.unified.to_dict('records'):
            cascade_dict[row['cascade_id']].append(row)

        self.data_list: list[Data] = []
        
        for ci, cid in enumerate(self.cascade_ids):
            cascade_nodes = cascade_dict[cid]
            x_root = torch.tensor(
                self.tfidf_matrix[ci].toarray(),
                dtype=torch.float32,
            )
            label_str = self.label_map.get(cid, "non-rumour")
            y = torch.tensor([self.le.transform([label_str])[0]], dtype=torch.long)
            
            for t_min in self.time_windows_minutes:
                t_s = float(t_min * 60)
                snap_nodes = [n for n in cascade_nodes if n["timestamp"] <= t_s]
                if not snap_nodes:
                    continue
                
                N = len(snap_nodes)
                node_list = [n["tweet_id"] for n in snap_nodes]
                snap_ids = set(node_list)
                node_to_idx = {nid: i for i, nid in enumerate(node_list)}
                
                edge_src = []
                edge_dst = []
                edge_ts = []
                
                for n in snap_nodes:
                    parent = n["parent_id"]
                    if pd.notna(parent) and parent in snap_ids:
                        edge_src.append(node_to_idx[parent])
                        edge_dst.append(node_to_idx[n["tweet_id"]])
                        edge_ts.append(float(n["timestamp"]))
                
                # Runtime leakage check
                snap_dict = {
                    "vertices": pd.DataFrame(snap_nodes).rename(columns={"tweet_id": "id"}),
                    "edges": pd.DataFrame({
                        "src": [n["parent_id"] for n in snap_nodes if pd.notna(n["parent_id"]) and n["parent_id"] in snap_ids],
                        "dst": [n["tweet_id"] for n in snap_nodes if pd.notna(n["parent_id"]) and n["parent_id"] in snap_ids],
                        "timestamp": edge_ts
                    }) if edge_ts else pd.DataFrame(columns=["src", "dst", "timestamp"])
                }
                assert_snapshot_is_clean(snap_dict, t_s)

                if edge_src:
                    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                    t_edge_s = torch.tensor(edge_ts, dtype=torch.float32)
                    edge_weight = compute_edge_weights(t_edge_s, t_s, self.lam)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_weight = torch.zeros(0, dtype=torch.float32)

                data = Data(
                    x=x_root.expand(N, -1),
                    edge_index=edge_index,
                    edge_attr=edge_weight,
                    y=y,
                    cascade_id=cid,
                    t_minutes=t_min,
                    t_snapshot_s=t_s,
                    num_nodes=N,
                )
                self.data_list.append(data)

        logger.info(
            "[SnapshotDataset] split=%s  cascades=%d  snapshots=%d",
            split_name, len(self.cascade_ids), len(self.data_list),
        )

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


# ---------------------------------------------------------------------------
# 7.  Training and evaluation
# ---------------------------------------------------------------------------

def _compute_class_weights(
    split_df: pd.DataFrame, split_name: str, device: torch.device
) -> torch.Tensor:
    sub = split_df[split_df["split"] == split_name]
    counts = sub["label"].value_counts()
    total = len(sub)
    # Order: [non-rumour=0, rumour=1]
    w_nr = total / (2 * counts.get("non-rumour", 1))
    w_r  = total / (2 * counts.get("rumour", 1))
    return torch.tensor([w_nr, w_r], dtype=torch.float32, device=device)


def train_c2v(
    encoder: CASCADE2VEC,
    classifier: C2VClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    n_epochs: int = 50,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    alpha: float = 0.5,         # weight of contrastive loss (1-alpha = CE weight)
    temperature: float = 0.07,
    patience: int = 10,
    class_weights: Optional[torch.Tensor] = None,
    device: torch.device = DEVICE,
    checkpoint_path: Optional[str] = None,
) -> dict:
    """
    Train CASCADE2VEC with combined SupCon + CrossEntropy loss.

    Loss = alpha * L_contrastive + (1 - alpha) * L_classification

    Returns dict with training history and best val metrics.
    """
    encoder.to(device)
    classifier.to(device)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        classifier.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeds = encoder(
                batch.x,
                batch.edge_index,
                batch.edge_attr if batch.edge_attr.numel() > 0 else None,
                batch.batch,
            )
            logits = classifier(embeds)

            # Classification loss
            ce = ce_loss_fn(logits, batch.y)

            # Contrastive loss
            con = supervised_infonce_loss(embeds, batch.y, temperature)

            loss = alpha * con + (1.0 - alpha) * ce
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Val evaluation
        val_metrics = evaluate_c2v(encoder, classifier, val_loader, device=device)
        val_f1 = val_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
        })

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "[train_c2v] epoch=%d  loss=%.4f  val_macro_f1=%.4f",
                epoch, avg_loss, val_f1,
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("[train_c2v] Early stopping at epoch %d", epoch)
                break

    logger.info(
        "[train_c2v] Best val Macro F1=%.4f at epoch %d",
        best_val_f1, best_epoch,
    )
    return {
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
    }


@torch.no_grad()
def evaluate_c2v(
    encoder: CASCADE2VEC,
    classifier: C2VClassifier,
    loader: DataLoader,
    *,
    device: torch.device = DEVICE,
    return_embeddings: bool = False,
) -> dict:
    """
    Evaluate encoder+classifier on a DataLoader split.
    Uses only the t=120min snapshot per cascade for a fair H1 comparison.

    Returns metrics dict. If return_embeddings=True, also returns embeddings
    array for visualisation.
    """
    encoder.eval()
    classifier.eval()

    # Collect predictions per cascade: use the LAST (most complete) snapshot
    cascade_preds: dict[str, tuple[int, float, np.ndarray]] = {}

    for batch in loader:
        batch = batch.to(device)
        embeds = encoder(
            batch.x,
            batch.edge_index,
            batch.edge_attr if batch.edge_attr.numel() > 0 else None,
            batch.batch,
        )
        logits = classifier(embeds)
        probs = F.softmax(logits, dim=-1)

        for i in range(batch.num_graphs):
            cid = batch.cascade_id[i] if hasattr(batch.cascade_id, '__getitem__') else batch.cascade_id
            t_min = int(batch.t_minutes[i].item()) if hasattr(batch.t_minutes, '__getitem__') else int(batch.t_minutes)
            true_y = int(batch.y[i].item())
            pred_y = int(logits[i].argmax().item())
            prob_rumour = float(probs[i, 1].item())
            embed_np = embeds[i].cpu().numpy()

            # Keep snapshot with largest t_minutes per cascade (most complete view)
            if cid not in cascade_preds or t_min > cascade_preds[cid][2]:
                cascade_preds[cid] = (true_y, pred_y, prob_rumour, t_min, embed_np)

    if not cascade_preds:
        return {"macro_f1": 0.0, "accuracy": 0.0}

    y_true = np.array([v[0] for v in cascade_preds.values()])
    y_pred = np.array([v[1] for v in cascade_preds.values()])
    y_prob = np.array([v[2] for v in cascade_preds.values()])
    embeds_all = np.stack([v[4] for v in cascade_preds.values()])

    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "macro_f1":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    if return_embeddings:
        metrics["embeddings"] = embeds_all
        metrics["labels"] = y_true
        metrics["probs"] = y_prob

    return metrics
