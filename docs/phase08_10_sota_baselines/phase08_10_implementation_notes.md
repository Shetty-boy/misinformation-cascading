# Phase 8-10: Implementation Notes

## Overview

Four SOTA baselines were implemented for the binary rumour detection task.
All trained on full final cascades only (see `phase08_10_data_interface_contract.md`).

---

## Model 1: Bi-GCN

### Source
**Adapted** from [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)
(MIT License, last updated December 2025).
Original paper: Bian et al., AAAI 2020, "Rumor Detection on Social Media with
Bi-Directional Graph Convolutional Networks."

### Architecture
- Two GCN branches (top-down + bottom-up propagation graph)
- Global mean pooling per branch
- Concatenated → Linear classifier (2-class)

### Deviations from Original Paper
| Deviation | Original | Ours | Reason |
|---|---|---|---|
| Node features | 5000-dim one-hot (spaCy) | TF-IDF (5000-dim, sklearn) | Avoids spaCy dependency |
| Datasets | Twitter15/16/PHEME | Our PHEME-derived dataset | Different data source |
| Batch size | 128 | 64 | Memory efficiency |

### Hyperparameters Used
| Param | Value |
|---|---|
| Seed | 42 |
| TF-IDF features | 5000 |
| Hidden dim | 128 |
| Dropout | 0.5 |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Max epochs | 50 |
| Batch size | 64 |
| Early stopping patience | 10 (on val Macro F1) |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Loss | CrossEntropyLoss (class-weighted) |
| GCN layers | 2 |
| Pooling | global_mean_pool |

### Results
- Best val Macro F1: 0.8573 (epoch 3)
- **Test Macro F1: 0.8237** ✅
- Test ROC-AUC: 0.9203
- Runtime: 0.27 min

---

## Model 2: RP-DNN

### Source
**Built from scratch** — no official public PyTorch implementation exists.
Original paper: Bian et al., 2020, "RP-DNN: A Tweet Level Propagation Context
Based Deep Neural Networks for Early Rumor Detection in Social Media."
Original code was in Theano and is not maintained.

### Architecture
- Branch 1: Bidirectional GRU over root tweet token sequence
- Branch 2: GRU over BFS-depth structural feature sequence
- Concatenated → 2-layer MLP → Binary classifier

### Deviations from Original Paper
| Deviation | Original | Ours | Reason |
|---|---|---|---|
| Text encoder | Theano LSTM | PyTorch BiGRU | Theano not maintained |
| Text features | Word2Vec embeddings | Simple token indices + nn.Embedding | Avoid external embedding dependency |
| Structural sequence | Proprietary features | [fraction_at_depth, branching_factor] | Closest reproducible equivalent |

### Hyperparameters Used
| Param | Value |
|---|---|
| Seed | 42 |
| Vocab size | 10000 |
| Embed dim | 64 |
| GRU hidden dim | 128 |
| Structural hidden | 64 |
| MLP hidden | 128 |
| Dropout | 0.5 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Max epochs | 50 |
| Batch size | 128 |
| Early stopping patience | 10 (on val Macro F1) |
| Optimizer | Adam |
| Loss | CrossEntropyLoss (class-weighted) |
| Max text length | 128 tokens |
| Max BFS depth | 30 levels |

### Results
- Best val Macro F1: 0.7943 (epoch 12)
- **Test Macro F1: 0.7709** ✅
- Test ROC-AUC: 0.8609
- Runtime: 0.11 min

---

## Model 3: PGNN

### Source
**Built from scratch** using PyTorch Geometric `MessagePassing` — no official
public implementation available.
Reference paper: Wu et al., 2021, "Rumor Detection Based On Propagation Graph
Neural Network With Attention Mechanism."

### Architecture
- 2-layer GCN with LeakyReLU activation
- Soft attention pooling (learned attention scores over node embeddings)
- 2-layer MLP classifier

### Deviations from Original Paper
| Deviation | Original | Ours | Reason |
|---|---|---|---|
| Base GNN | Gated Graph Neural Network (GGNN) | GCN (GCNConv) | GGNN has known gradient instability in PyG; GCN is equivalent in practice |
| Pooling | Not specified precisely | Soft attention pool | Consistent with paper's "attention mechanism" description |
| Node features | Paper uses content + structural | TF-IDF text features only | Simpler, still captures main signal |

### Hyperparameters Used
| Param | Value |
|---|---|
| Seed | 42 |
| TF-IDF features | 5000 |
| Hidden dim | 128 |
| Attention dim | 64 |
| MLP hidden | 128 |
| Dropout | 0.5 |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Max epochs | 50 |
| Batch size | 64 |
| Early stopping patience | 10 (on val Macro F1) |
| Optimizer | Adam |
| Loss | CrossEntropyLoss (class-weighted) |
| GCN layers | 2 |
| Pooling | Soft attention |

### Results
- Best val Macro F1: 0.8436 (epoch 8)
- **Test Macro F1: 0.8237** ✅
- Test ROC-AUC: 0.9232
- Runtime: 0.17 min

---

## Model 4: KPG-simplified

### Source
**Built from scratch** — independent simplification (NOT an attributed ablation
from the original paper). The repository [kkkkk001/KPG](https://github.com/kkkkk001/KPG)
exists (11 stars) but has no README, no requirements, and no documentation —
it is not usable as a reproducible baseline.

Reference paper: "Rumor Detection on Social Media with Reinforcement
Learning-based Key Propagation Graph Generator" (ACL Findings, 2023).

### Architecture
The original KPG trains a key-node selector using REINFORCE (policy gradient RL)
to identify informative propagation paths. We implement a simplified variant:

- **Key-node selection**: Top-K nodes by approximate betweenness centrality (static, no RL)
- K=20 (or full cascade if fewer nodes)
- 2-layer GCN on the pruned graph
- Global mean pooling → 2-layer MLP → Binary classifier

**This is an independent engineering simplification, not an attributed
ablation variant from the original paper. Results are lower-bound estimates
of what the full RL-trained KPG might achieve.**

### Deviations from Original Paper
| Deviation | Original | Ours | Reason |
|---|---|---|---|
| Key-node selector | RL-trained (REINFORCE) | Static betweenness centrality | RL training is unstable and time-consuming; no usable code available |
| Graph backbone | Custom propagation encoder | GCNConv | Simpler, matches other baseline architectures |

### Hyperparameters Used
| Param | Value |
|---|---|
| Seed | 42 |
| TF-IDF features | 5000 |
| Key nodes K | 20 |
| Hidden dim | 128 |
| MLP hidden | 128 |
| Dropout | 0.5 |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Max epochs | 50 |
| Batch size | 64 |
| Early stopping patience | 10 (on val Macro F1) |
| Optimizer | Adam |
| Loss | CrossEntropyLoss (class-weighted) |
| GCN layers | 2 |
| Pooling | global_mean_pool |
| Centrality | Approximate betweenness (tree formula) |

### Results
- Best val Macro F1: 0.8610 (epoch 5)
- **Test Macro F1: 0.8311** ✅
- Test ROC-AUC: 0.9187
- Runtime: 0.25 min

---

## Runtime Summary

| Model | Runtime | Flag |
|---|---|---|
| BiGCN | 0.27 min | ✅ Fast |
| RP-DNN | 0.11 min | ✅ Fast |
| PGNN | 0.17 min | ✅ Fast |
| KPG-simplified | 0.25 min | ✅ Fast |

All models are well within 30-minute retraining budget. No flags for Phase 18 ablation studies.

---

## Shared Protocol

All 4 models follow these shared rules:
- **SEED = 42** (weight init, dropout, DataLoader shuffling, all stochastic ops)
- **Test split accessed EXACTLY ONCE** per model, after all training/val-selection is done
- **Early stopping on val Macro F1** (patience=10), best checkpoint by val Macro F1
- **Class-weighted CrossEntropyLoss** to handle ~1.94:1 non-rumour:rumour imbalance
- **Gradient clipping** at max_norm=1.0 for training stability
- **TF-IDF vocabulary** built on training split only (no leakage to val/test)

### Problems Encountered & Resolutions
- **Missing or Unmaintained PyTorch SOTA Repos:** No official PyTorch code was available for RP-DNN, PGNN, or KPG (original KPG used unstable and undocumented RL).
  - *Fix:* Built RP-DNN and PGNN entirely from scratch based on paper descriptions. Implemented a simplified version of KPG using static betweenness centrality instead of RL to guarantee reproducibility within the time budget.
- **KeyError in Pandas Merge:** `compare_baselines.py` crashed during a DataFrame merge operation because both `feature_matrix.parquet` and the split mapping DataFrame contained a `label` column.
  - *Fix:* Corrected the merge logic to only pull `cascade_id` and `split` from the split DataFrame, relying on the `label` column natively present in the feature matrix.
- **Silent File Overwrites (Footgun):** The split generation script (`split_data.py`) wrote its output to a hardcoded path (`train_val_test_split.parquet`) without an explicit `--output` flag.
  - *Fix:* Documented the problem in a repo-wide audit to ensure future scripts implement explicit overwrite guards.
