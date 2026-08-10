"""
rpdnn_input.py — Adapter: raw unified.parquet -> sequences for RP-DNN.

Contract: build_rpdnn_sequences(df, word2idx, max_len, max_depth) -> List[Tuple]

Each tuple: (text_tokens, struct_seq, label)
  text_tokens : [MAX_LEN]         int32 padded token indices for root tweet
  struct_seq  : [MAX_DEPTH, 2]    float32 structural features per BFS depth level
  label       : int               0=non-rumour, 1=rumour

Structural features per depth level:
  [0] fraction of nodes at this depth level (count / total nodes)
  [1] branching factor at this level (children / parents from prev level)

FULL CASCADE ONLY: no temporal truncation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict

LABEL_MAP = {"non-rumour": 0, "rumour": 1}


def build_vocabulary(train_df: pd.DataFrame, max_vocab: int = 10000) -> dict[str, int]:
    """Build word-to-index map from training text only."""
    from collections import Counter
    counter: Counter = Counter()
    for text in train_df["text"].fillna("").tolist():
        counter.update(text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def tokenize(text: str, word2idx: dict, max_len: int) -> list[int]:
    """Tokenize and pad/truncate a text string."""
    tokens = text.lower().split()[:max_len]
    ids = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    ids += [word2idx["<PAD>"]] * (max_len - len(ids))
    return ids


def _bfs_depth_features(group: pd.DataFrame, max_depth: int) -> np.ndarray:
    """
    Compute structural features per BFS depth level.
    Returns array of shape [max_depth, 2]:
      feat[d, 0] = fraction of nodes at depth d
      feat[d, 1] = branching factor at depth d (children / parents at depth d-1)
    """
    id_to_parent = {}
    root_id = None
    for _, row in group.iterrows():
        tid = row["tweet_id"]
        pid = row["parent_id"]
        id_to_parent[tid] = pid
        if pd.isna(pid):
            root_id = tid

    if root_id is None:
        # No root found (disconnected) — return zeros
        return np.zeros((max_depth, 2), dtype=np.float32)

    # BFS
    from collections import deque
    children: dict = defaultdict(list)
    for tid, pid in id_to_parent.items():
        if pd.notna(pid):
            children[pid].append(tid)

    depth_counts: dict[int, int] = defaultdict(int)
    queue = deque([(root_id, 0)])
    visited = {root_id}
    while queue:
        node, d = queue.popleft()
        depth_counts[d] += 1
        for child in children.get(node, []):
            if child not in visited:
                visited.add(child)
                queue.append((child, d + 1))

    total_nodes = len(group)
    feats = np.zeros((max_depth, 2), dtype=np.float32)
    for d in range(max_depth):
        count_at_d = depth_counts.get(d, 0)
        count_at_prev = depth_counts.get(d - 1, 1) if d > 0 else 1
        feats[d, 0] = count_at_d / max(total_nodes, 1)
        feats[d, 1] = count_at_d / max(count_at_prev, 1)
    return feats


def build_rpdnn_sequences(
    df: pd.DataFrame,
    word2idx: dict[str, int],
    max_len: int = 128,
    max_depth: int = 30,
) -> list[tuple]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Rows from unified.parquet for a single split.
    word2idx : dict
        Vocabulary built on training text only (see build_vocabulary()).
    max_len : int
        Max token length for text sequence.
    max_depth : int
        Max BFS depth levels for structural sequence.

    Returns
    -------
    List of (text_tokens, struct_seq, label) tuples.
    """
    results = []
    for cascade_id, group in df.groupby("cascade_id"):
        # Root tweet text
        root_rows = group[group["parent_id"].isna()]
        if len(root_rows) == 0:
            root_text = ""
        else:
            root_text = root_rows.iloc[0]["text"] if pd.notna(root_rows.iloc[0]["text"]) else ""

        text_tokens = np.array(tokenize(root_text, word2idx, max_len), dtype=np.int64)
        struct_seq  = _bfs_depth_features(group, max_depth)  # [max_depth, 2]
        label       = LABEL_MAP[group["label"].iloc[0]]

        results.append((text_tokens, struct_seq, label))

    return results
