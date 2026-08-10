"""
bigcn_input.py — Adapter: raw unified.parquet → PyG Data objects for Bi-GCN.

Contract: build_bigcn_data(df, tfidf_vectorizer) -> List[Data]
  - df: subset of unified.parquet for one split (train, val, or test)
  - tfidf_vectorizer: sklearn TfidfVectorizer already FIT on training data only

Each returned Data object represents ONE full cascade:
  data.x           : [N, 5000] float32 TF-IDF node features
  data.edge_index  : [2, E]   long  top-down edges (used as TD by BiGCN)
  data.edge_index_bu: [2, E]  long  bottom-up edges (reversed TD)
  data.y           : [1]      long  label (0=non-rumour, 1=rumour)
  data.cascade_id  : str      for bookkeeping

FULL CASCADE ONLY: no temporal truncation. All nodes/edges in the cascade
are included regardless of timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.sparse import issparse


LABEL_MAP = {"non-rumour": 0, "rumour": 1}


def build_bigcn_data(df: pd.DataFrame, tfidf_vectorizer) -> list[Data]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Rows from unified.parquet for a single split.
        Required columns: tweet_id, cascade_id, parent_id, text, label
    tfidf_vectorizer : fitted sklearn TfidfVectorizer
        Must already be fit on training data. This function calls .transform()
        only — never fit_transform() — to prevent leakage.

    Returns
    -------
    List[Data] — one PyG Data object per cascade.
    """
    data_list: list[Data] = []

    for cascade_id, group in df.groupby("cascade_id"):
        group = group.reset_index(drop=True)

        # Node ordering: deterministic sort by tweet_id
        group = group.sort_values("tweet_id").reset_index(drop=True)
        node_id_to_idx = {tid: i for i, tid in enumerate(group["tweet_id"])}

        # ── Node features (TF-IDF) ──────────────────────────────────────────
        texts = group["text"].fillna("").tolist()
        feat_matrix = tfidf_vectorizer.transform(texts)
        if issparse(feat_matrix):
            feat_matrix = feat_matrix.toarray()
        x = torch.tensor(feat_matrix, dtype=torch.float32)  # [N, vocab_size]

        # ── Edges ───────────────────────────────────────────────────────────
        src_td, dst_td = [], []
        for _, row in group.iterrows():
            pid = row["parent_id"]
            tid = row["tweet_id"]
            if pd.notna(pid) and pid in node_id_to_idx and tid in node_id_to_idx:
                src_td.append(node_id_to_idx[pid])
                dst_td.append(node_id_to_idx[tid])

        if src_td:
            edge_index_td = torch.tensor([src_td, dst_td], dtype=torch.long)
            edge_index_bu = torch.tensor([dst_td, src_td], dtype=torch.long)
        else:
            # Singleton or no edges
            edge_index_td = torch.zeros((2, 0), dtype=torch.long)
            edge_index_bu = torch.zeros((2, 0), dtype=torch.long)

        # ── Label ────────────────────────────────────────────────────────────
        label_str = group["label"].iloc[0]
        y = torch.tensor([LABEL_MAP[label_str]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index_td, y=y)
        data.edge_index_bu = edge_index_bu
        data.cascade_id = cascade_id
        data_list.append(data)

    return data_list


def fit_tfidf(train_df: pd.DataFrame, max_features: int = 5000):
    """
    Fit TF-IDF vectorizer on training text only.
    Call this ONCE on the training split, then pass the fitted vectorizer
    to build_bigcn_data() for all splits.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = train_df["text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    vectorizer.fit(texts)
    return vectorizer
