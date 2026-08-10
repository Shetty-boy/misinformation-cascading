"""
kpg_input.py — Adapter: raw unified.parquet -> PyG Data objects for KPG-simplified.

Contract: build_kpg_data(df, tfidf_vectorizer, k=20) -> List[Data]

KPG-simplified selects the top-K nodes by betweenness centrality score
(static selection — no RL-based training). This is an independent simplification;
see implementation_notes.md for the documented deviation from the original paper.

Each Data object:
  data.x          : [min(K,N), 5000] float32 TF-IDF features for selected nodes
  data.edge_index : [2, E']          long    edges among selected nodes
  data.y          : [1]              long    label
  data.cascade_id : str

FULL CASCADE ONLY: no temporal truncation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.sparse import issparse

LABEL_MAP = {"non-rumour": 0, "rumour": 1}


def _betweenness_centrality_approx(group: pd.DataFrame) -> dict:
    """
    Approximate betweenness centrality using BFS from each node.
    Returns dict: tweet_id -> centrality score.
    For trees, uses a simpler formula: subtree_size * (N - subtree_size).
    """
    from collections import defaultdict, deque

    parent_map = {}
    children_map: dict = defaultdict(list)
    root_id = None

    for _, row in group.iterrows():
        tid = row["tweet_id"]
        pid = row["parent_id"]
        parent_map[tid] = pid
        if pd.isna(pid):
            root_id = tid
        else:
            children_map[pid].append(tid)

    if root_id is None:
        # Disconnected — all equal
        return {row["tweet_id"]: 1.0 for _, row in group.iterrows()}

    N = len(group)
    subtree_sizes: dict[str, int] = {}

    # Compute subtree sizes via post-order DFS
    stack = [(root_id, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            size = 1 + sum(subtree_sizes.get(c, 1) for c in children_map.get(node, []))
            subtree_sizes[node] = size
        else:
            stack.append((node, True))
            for child in children_map.get(node, []):
                stack.append((child, False))

    # Betweenness approx for trees:
    # Each edge (u, parent(u)) has subtree_size(u) * (N - subtree_size(u)) paths passing through
    # Betweenness of node u ~ sum over children c of: subtree_size(c) * (N - subtree_size(c))
    centrality: dict[str, float] = {}
    for _, row in group.iterrows():
        tid = row["tweet_id"]
        score = sum(
            subtree_sizes.get(c, 1) * (N - subtree_sizes.get(c, 1))
            for c in children_map.get(tid, [])
        )
        centrality[tid] = float(score)

    # Root always has high centrality; normalize
    max_score = max(centrality.values()) if centrality else 1.0
    if max_score > 0:
        centrality = {k: v / max_score for k, v in centrality.items()}

    return centrality


def build_kpg_data(
    df: pd.DataFrame,
    tfidf_vectorizer,
    k: int = 20,
) -> list[Data]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Rows from unified.parquet for a single split.
    tfidf_vectorizer : fitted sklearn TfidfVectorizer
        Must already be fit on training data only.
    k : int
        Number of key nodes to select. If cascade has <= k nodes, all are kept.

    Returns
    -------
    List[Data]
    """
    data_list: list[Data] = []

    for cascade_id, group in df.groupby("cascade_id"):
        group = group.sort_values("tweet_id").reset_index(drop=True)
        N = len(group)

        # Select top-K nodes by betweenness centrality
        centrality = _betweenness_centrality_approx(group)
        sorted_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)
        selected = set(sorted_nodes[:k])

        # Filter group to selected nodes
        sel_group = group[group["tweet_id"].isin(selected)].reset_index(drop=True)
        node_id_to_idx = {tid: i for i, tid in enumerate(sel_group["tweet_id"])}

        # Node features (TF-IDF)
        texts = sel_group["text"].fillna("").tolist()
        feat_matrix = tfidf_vectorizer.transform(texts)
        if issparse(feat_matrix):
            feat_matrix = feat_matrix.toarray()
        x = torch.tensor(feat_matrix, dtype=torch.float32)

        # Edges among selected nodes
        srcs, dsts = [], []
        for _, row in sel_group.iterrows():
            pid = row["parent_id"]
            tid = row["tweet_id"]
            if pd.notna(pid) and pid in node_id_to_idx and tid in node_id_to_idx:
                srcs.append(node_id_to_idx[pid])
                dsts.append(node_id_to_idx[tid])

        if srcs:
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        label_str = group["label"].iloc[0]
        y = torch.tensor([LABEL_MAP[label_str]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.cascade_id = cascade_id
        data_list.append(data)

    return data_list
