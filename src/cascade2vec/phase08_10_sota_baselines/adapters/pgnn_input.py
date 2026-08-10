"""
pgnn_input.py — Adapter: raw unified.parquet -> PyG Data objects for PGNN.

Contract: build_pgnn_data(df, tfidf_vectorizer) -> List[Data]

Each Data object represents ONE full cascade:
  data.x          : [N, 5000] float32 TF-IDF node features
  data.edge_index : [2, E]    long    top-down propagation edges (parent->child)
  data.y          : [1]       long    label
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


def build_pgnn_data(df: pd.DataFrame, tfidf_vectorizer) -> list[Data]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Rows from unified.parquet for a single split.
    tfidf_vectorizer : fitted sklearn TfidfVectorizer
        Must already be fit on training data only.

    Returns
    -------
    List[Data]
    """
    data_list: list[Data] = []

    for cascade_id, group in df.groupby("cascade_id"):
        group = group.sort_values("tweet_id").reset_index(drop=True)
        node_id_to_idx = {tid: i for i, tid in enumerate(group["tweet_id"])}

        # Node features
        texts = group["text"].fillna("").tolist()
        feat_matrix = tfidf_vectorizer.transform(texts)
        if issparse(feat_matrix):
            feat_matrix = feat_matrix.toarray()
        x = torch.tensor(feat_matrix, dtype=torch.float32)

        # Edges (top-down only for PGNN)
        srcs, dsts = [], []
        for _, row in group.iterrows():
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
