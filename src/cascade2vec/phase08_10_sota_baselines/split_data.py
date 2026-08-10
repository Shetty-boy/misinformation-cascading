"""
split_data.py — Generate the fixed 70/15/15 train/val/test split for Phase 8-10.

Run once:
    python src/cascade2vec/phase08_10_sota_baselines/split_data.py

Saves to:
    data/processed/phase08_10_sota_baselines/train_val_test_split.parquet

Columns: cascade_id, split (train/val/test), label

Design decisions:
- Stratified by label so class ratio is preserved in each split
- Grouped at cascade level (no cascade_id appears in more than one split)
- Seed is fixed at 42 for full reproducibility
- FULL cascades only -- no temporal truncation (see data_interface_contract.md)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC = 0.15 (implicit remainder)

OUT_DIR  = "data/processed/phase08_10_sota_baselines"
OUT_FILE = os.path.join(OUT_DIR, "train_val_test_split.parquet")
SRC_FILE = "data/processed/phase02_ingestion/unified.parquet"


def build_split():
    print("[split_data] Loading unified dataset...")
    df = pd.read_parquet(SRC_FILE)

    # Build one row per cascade: cascade_id -> label
    cascade_labels = (
        df.groupby("cascade_id")["label"]
        .first()
        .reset_index()
    )
    cascade_labels.columns = ["cascade_id", "label"]

    n = len(cascade_labels)
    print(f"[split_data] Total cascades: {n}")
    print(f"[split_data] Label distribution:\n{cascade_labels['label'].value_counts()}")

    # Map labels to integers for stratification
    label_map = {"rumour": 1, "non-rumour": 0}
    cascade_labels["label_int"] = cascade_labels["label"].map(label_map)

    X = cascade_labels[["cascade_id"]].values
    y = cascade_labels["label_int"].values

    # Step 1: Split off test set (15% of total)
    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_FRAC + (1 - TRAIN_FRAC - VAL_FRAC),
        random_state=SEED
    )
    train_val_idx, test_idx = next(splitter1.split(X, y))

    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    X_test      = X[test_idx]

    # Step 2: Split train_val into train (70% of total) and val (15% of total)
    # val is 15/85 ~= 17.6% of the train+val pool
    val_frac_of_trainval = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_of_trainval, random_state=SEED
    )
    train_idx_local, val_idx_local = next(splitter2.split(X_train_val, y_train_val))

    train_cascade_ids = X_train_val[train_idx_local, 0]
    val_cascade_ids   = X_train_val[val_idx_local, 0]
    test_cascade_ids  = X_test[:, 0]

    # Verify no overlap
    assert len(set(train_cascade_ids) & set(val_cascade_ids)) == 0, "Train/Val overlap!"
    assert len(set(train_cascade_ids) & set(test_cascade_ids)) == 0, "Train/Test overlap!"
    assert len(set(val_cascade_ids)   & set(test_cascade_ids)) == 0, "Val/Test overlap!"
    assert len(train_cascade_ids) + len(val_cascade_ids) + len(test_cascade_ids) == n

    print(f"[split_data] Train: {len(train_cascade_ids)} | Val: {len(val_cascade_ids)} | Test: {len(test_cascade_ids)}")

    # Build output DataFrame
    rows = []
    for cid in train_cascade_ids:
        rows.append({"cascade_id": cid, "split": "train"})
    for cid in val_cascade_ids:
        rows.append({"cascade_id": cid, "split": "val"})
    for cid in test_cascade_ids:
        rows.append({"cascade_id": cid, "split": "test"})

    split_df = pd.DataFrame(rows)
    split_df = split_df.merge(cascade_labels[["cascade_id", "label"]], on="cascade_id")

    # Verify label distribution per split
    for split_name in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == split_name]
        rumour_frac = (sub["label"] == "rumour").mean()
        print(f"  [{split_name}] n={len(sub)}, rumour_frac={rumour_frac:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    split_df.to_parquet(OUT_FILE, index=False)
    print(f"[split_data] Saved split to {OUT_FILE}")
    return split_df


if __name__ == "__main__":
    np.random.seed(SEED)
    split_df = build_split()
    print("[split_data] Done.")
