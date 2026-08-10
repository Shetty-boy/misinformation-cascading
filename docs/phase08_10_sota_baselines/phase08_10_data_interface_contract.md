# Phase 8-10: Data Interface Contract

## Full-Cascade-Only Policy

**All 4 SOTA baselines in this phase are trained and evaluated on FULL final
cascades only.** This matches the original papers' standard benchmark setup
(Twitter15/16/PHEME).

### Limitation: No Snapshot/Early-Evaluation Support

These models do NOT currently support snapshot-truncated evaluation at
intermediate time cutoffs (t=1, 5, 10 ... 120 minutes). They receive the
complete propagation tree as input at inference time.

**Impact for Phase 13-14 (Adaptive Early-Stopping):** If Phase 13-14 needs
these SOTA baselines evaluated at early time cutoffs for a fair
early-detection comparison, this requires separate additional work:
  - Option A: Retrain each model on temporally truncated training data per window
  - Option B: Apply them at inference-time to truncated graphs — this will
    degrade performance since they weren't trained for that regime, and
    this degradation must be explicitly flagged in results

This is explicitly out of scope for Phase 8-10.

---

## What We Have

| Asset | Location | Contents |
|---|---|---|
| Raw text + structure | `data/processed/phase02_ingestion/unified.parquet` | tweet_id, cascade_id, parent_id, text, timestamp, label |
| Feature matrix | `data/processed/phase06_07_features/feature_matrix.parquet` | 19 structural/temporal features per (cascade_id, t) — NOT used by SOTA baselines |
| Fixed split | `data/processed/phase08_10_sota_baselines/train_val_test_split.parquet` | cascade_id → train/val/test, label |

---

## Per-Model Input Contract

### Bi-GCN

**Paper:** Bian et al., AAAI 2020, "Rumor Detection on Social Media with
Bi-Directional Graph Convolutional Networks"

**Input format:** PyG `Data` objects, one per cascade

| Field | Shape | Description |
|---|---|---|
| `x` | `[N, 5000]` | TF-IDF node feature vectors (vocab built on train split only) |
| `edge_index_td` | `[2, E]` | Top-down edges: parent→child |
| `edge_index_bu` | `[2, E]` | Bottom-up edges: child→parent (reversed graph) |
| `y` | `[1]` | Label (0=non-rumour, 1=rumour) |
| `num_nodes` | int | Number of nodes N |

**Adapter:** `adapters/bigcn_input.py::build_bigcn_data(df, tfidf_vectorizer)`

**TF-IDF:** max_features=5000, fit on training cascades only. Applied
identically to val/test (vocabulary frozen after training split fit).

---

### RP-DNN

**Paper:** Bian et al., 2020, "RP-DNN: A Tweet Level Propagation Context
Based Deep Neural Networks for Early Rumor Detection in Social Media"

**Input format:** Two sequences per cascade

| Field | Shape | Description |
|---|---|---|
| `text_seq` | `[MAX_LEN]` | Token indices for root tweet text (padded) |
| `struct_seq` | `[MAX_DEPTH, STRUCT_DIM]` | Structural feature sequence ordered by BFS depth level. Features: node_count_at_depth, branching_factor_at_depth |
| `y` | int | Label (0 or 1) |

MAX_LEN=128, MAX_DEPTH=30, STRUCT_DIM=2

**Adapter:** `adapters/rpdnn_input.py::build_rpdnn_sequences(df, tokenizer, max_len, max_depth)`

**Tokenizer:** Simple whitespace tokenizer, vocab built on training text only.

---

### PGNN

**Paper:** Wu et al., 2021, "Rumor Detection Based On Propagation Graph
Neural Network With Attention Mechanism"

**Input format:** PyG `Data` objects, one per cascade

| Field | Shape | Description |
|---|---|---|
| `x` | `[N, 5000]` | TF-IDF node feature vectors (same vocab as Bi-GCN) |
| `edge_index` | `[2, E]` | Propagation tree edges (top-down only) |
| `y` | `[1]` | Label |

**Adapter:** `adapters/pgnn_input.py::build_pgnn_data(df, tfidf_vectorizer)`

---

### KPG-simplified

**Note:** This is an independent simplification (not an attributed ablation
from the original paper). The original KPG uses reinforcement learning
(REINFORCE algorithm) to train a key-node selector. We implement a static
centrality-based variant due to the RL component's implementation complexity
and training instability. This deviation is explicitly documented in
`phase08_10_implementation_notes.md`.

**Input format:** PyG `Data` objects, one per cascade (pruned to key nodes)

| Field | Shape | Description |
|---|---|---|
| `x` | `[K, 5000]` | TF-IDF features for top-K key nodes (by betweenness centrality) |
| `edge_index` | `[2, E']` | Edges among the K selected nodes |
| `y` | `[1]` | Label |

K=20 (or full cascade if fewer than 20 nodes)

**Adapter:** `adapters/kpg_input.py::build_kpg_data(df, tfidf_vectorizer, k=20)`

---

## Leakage Prevention

1. TF-IDF vocabulary is fit on training cascades **only**
2. `transform()` (not `fit_transform()`) is used for val/test
3. Test split is accessed **once only**, for final metric reporting
4. All model selection and early stopping uses val split exclusively
