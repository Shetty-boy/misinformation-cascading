# CASCADE2VEC: Adaptive Early Misinformation Detection via Time-Weighted Cascade Embeddings

## Technical Report

---

## 1. Introduction & Problem Statement

**Research Problem:** Early detection of misinformation and rumours from Twitter reply-cascades. The goal is to accurately classify a cascade (a source tweet and its reply tree) as a *rumour* or *non-rumour* based on its topological structure and textual content, ideally *before* it fully unfolds — minimizing the time required to flag harmful content.

**Core Hypotheses:**

| ID | Hypothesis | Status |
|---|---|---|
| **H1** | A time-weighted graph embedding (CASCADE2VEC), where edge influence decays based on temporal arrival, separates rumours from non-rumours more effectively than static SOTA graph models. | **NOT SUPPORTED** (statistically) |
| **H2** | A learned, per-cascade confidence threshold θ(t) can reliably flag rumours earlier than a fixed observation window, yielding equal accuracy with lower mean time-to-detection. | **NOT SUPPORTED** (MDT reduction < 10% threshold) |
| **H4** | The pipeline exhibits near-linear speedup under single-machine core scaling (local[1] → local[8]). | **Pending** (Phase 16-17 executing) |

---

## 2. Dataset

### 2.1 PHEME Dataset

The project uses the **PHEME dataset** (Zubiaga et al., 2016), a publicly available corpus of Twitter conversation threads annotated for rumour status.

| Property | Value |
|---|---|
| Total tweets (rows) | 102,440 |
| Unique cascades | 5,802 |
| Unique events | 5 |
| Rumour cascades | 1,972 (34.0%) |
| Non-rumour cascades | 3,830 (66.0%) |
| Class imbalance ratio | ~1.94:1 (non-rumour:rumour) |

**Events covered:**
- Charlie Hebdo shooting
- Sydney siege
- Ferguson unrest
- Ottawa shooting
- Germanwings crash

> **Why PHEME?** It is self-contained (text + structure released together), bypassing the need to re-hydrate deleted tweets from the Twitter API — a major issue with Twitter15/16 datasets. The "rumour vs. non-rumour" binary task was chosen over veracity classification because verifying truth often takes days, whereas identifying *that* a cascade is a rumour enables early intervention.

### 2.2 Unified Schema

All raw data is parsed into a single Parquet table (`unified.parquet`) with the following columns:

| Column | Type | Description |
|---|---|---|
| `tweet_id` | string | Unique tweet identifier |
| `user_id` | string | Author identifier |
| `timestamp` | long | Seconds since cascade root (normalized to 0 for root) |
| `text` | string | Tweet text content |
| `parent_id` | string (nullable) | ID of the tweet being replied to (NULL for root) |
| `cascade_id` | string | Source tweet ID (groups all replies into one cascade) |
| `event_id` | string | Real-world event category |
| `label` | string | `"rumour"` or `"non-rumour"` (cascade-level) |

### 2.3 Graph Statistics

| Metric | Value |
|---|---|
| Total cascades | 5,802 |
| Singleton cascades (0 edges) | ~358 |
| Disconnected cascades | ~606 |
| Connected cascades | ~4,838 |
| Mean node count per cascade | ~17.7 |
| Mean edge count per cascade | ~16.7 |
| Mean max depth | ~2.5 |

### 2.4 Train / Validation / Test Split

A fixed **70 / 15 / 15** cascade-level stratified split is used for all deep learning models:

| Split | Cascades | Rumour | Non-Rumour |
|---|---|---|---|
| Train | 3,344 | ~1,155 | ~2,189 |
| Val | 717 | ~247 | ~470 |
| Test | 1,741 | ~570 | ~1,171 |

> **Design rationale:** Switched from 5-fold CV (used for simple baselines) to a fixed split for deep learning due to computational cost. Cascade-level stratification ensures that all snapshots of the same cascade stay within the same split — preventing data leakage.

---

## 3. Technical Architecture

### 3.1 Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | ≥ 3.10 |
| Graph Processing | PySpark + GraphFrames | 3.5.1 / 0.9.0 |
| Deep Learning | PyTorch + PyTorch Geometric | 2.3.1 / 2.5.3 |
| Tabular ML | scikit-learn, XGBoost | 1.5.0 / 2.0.3 |
| Explainability | SHAP, GNNExplainer (PyG) | 0.45.1 |
| Visualization | matplotlib, seaborn | 3.9.0 / 0.13.2 |
| Data Format | Apache Parquet (via PyArrow) | 15.0.2 |
| Testing | pytest | 8.2.2 |
| JVM Requirement | Java 17 | (for Spark) |

### 3.2 Project Structure

The project is organized linearly by execution phase:

```
src/cascade2vec/
├── phase02_ingestion/         # Raw PHEME → unified Parquet
├── phase04_05_graph/          # PySpark graph construction + BFS depths
├── phase06_07_features/       # Feature engineering + simple ML baselines
├── phase08_10_sota_baselines/ # BiGCN, KPG, PGNN, RP-DNN implementations
├── phase11_12_cascade2vec/    # Custom time-weighted GraphSAGE model
├── phase13_14_adaptive_stopping/ # Early detection loop (H2)
├── phase15_xai/               # SHAP + GNNExplainer
├── phase16_17_scalability/    # Synthetic data generation + benchmarks
└── phase18_eval/              # Ablations + final evaluation
```

### 3.3 Leakage Prevention

A critical integrity measure is enforced throughout:

- **`assert_snapshot_is_clean()`** (in `leakage_audit.py`): Called on every temporal snapshot to verify that no node has a timestamp exceeding the snapshot cutoff. Future nodes/edges are **excluded entirely** — not included with near-zero weight.
- **`PIPELINE_VERSION` tracking**: After a contamination incident was discovered mid-sweep (stale checkpoint data), a strict version tag (`v2-dict-snapshot`) was introduced. All tainted data was hard-deleted, and the full 72-config sweep was re-run.
- **Cascade-level splits**: Snapshots of the same cascade never cross train/val/test boundaries.

---

## 4. Feature Engineering (Phase 6-7)

### 4.1 Temporal Snapshots

Each cascade is observed at **8 time windows**: t ∈ {1, 2, 5, 10, 15, 30, 60, 120} minutes. At each window, only tweets with `timestamp ≤ t_seconds` are included.

### 4.2 Structural Features (Phase 6A)

| Feature | Description |
|---|---|
| `node_count` | Number of tweets in the snapshot |
| `edge_count` | Number of reply edges |
| `max_depth` | Maximum BFS depth from root |
| `avg_depth` | Mean BFS depth of reachable nodes |
| `leaf_count` | Number of leaf nodes (no children) |
| `leaf_ratio` | `leaf_count / node_count` |
| `branching_factor` | `edge_count / internal_node_count` |
| `root_degree` | Number of direct replies to the root |
| `reachable_ratio` | Fraction of nodes reachable from root |
| `is_connected` | Boolean: is the cascade fully connected? |

### 4.3 Temporal Features (Phase 6A)

| Feature | Description |
|---|---|
| `tweets_per_minute` | Posting rate |
| `growth_velocity` | `node_count / cascade_age` |
| `mean_interarrival` | Mean time between consecutive tweets |
| `std_interarrival` | Standard deviation of inter-arrival times |
| `burstiness` | `(std - mean) / (std + mean)` of inter-arrival |
| `cascade_age` | Time of last tweet in snapshot (seconds) |

### 4.4 Velocity Features (Phase 6B)

| Feature | Description |
|---|---|
| `depth_velocity` | Rate of change of `max_depth` over Δt |
| `breadth_velocity` | Rate of change of `node_count` over Δt |
| `branching_velocity` | Rate of change of `branching_factor` over Δt |

---

## 5. Model Architectures

### 5.1 Simple Baselines (Phase 6-7)

Three classifiers trained on the tabular feature matrix using 5-fold Stratified Group K-Fold CV (grouped by `cascade_id`):

- **Logistic Regression** (class-weighted)
- **Random Forest** (class-weighted)
- **XGBoost** (class-weighted)

### 5.2 SOTA Deep Learning Baselines (Phase 8-10)

Four graph neural network architectures re-implemented from literature:

| Model | Architecture | Key Design |
|---|---|---|
| **KPG-simplified** | 2-layer GCN on key-node-pruned graph | Static betweenness centrality selection (K=20 nodes). Original RL-based selector was unstable; this is an independent simplification. |
| **BiGCN** | Bi-directional GCN | Processes both top-down and bottom-up propagation directions. Adapted from safe-graph/GNN-FakeNews (MIT license). |
| **PGNN** | Position-aware GNN with attention | 2-layer GCN + attention-based pooling. Built from scratch. |
| **RP-DNN** | Recurrent Propagation DNN | Text embedding (learned vocabulary) + structural features through LSTM. Built from scratch. |

**Common training protocol:**
- Optimizer: Adam (lr=5e-4, weight_decay=1e-4)
- Loss: CrossEntropyLoss with class weights
- Patience: 10 epochs
- Node features: TF-IDF (max_features=5000), fitted on train split only
- Seed: 42

### 5.3 CASCADE2VEC (Phase 11-12) — Proposed Model

**Architecture:**

```
Input: TF-IDF node features (5000-dim)
    ↓
n × TimeWeightedSAGEConv layers (in_dim → 128)
    ↓  (with Dropout between layers)
Attention Pooling (learned scalar score per node)
    ↓
Linear Projection → embed_dim
    ↓
L2 Normalization → cascade embedding
    ↓
C2VClassifier MLP → 2-class output
```

**Time-Weighted GraphSAGE Convolution:**

Standard SAGE: `h_v = σ(W · concat(h_v, mean_{u∈N(v)} h_u))`

Time-weighted modification: replace `mean` with **weighted aggregation** using exponential decay:

```
w(e) = exp(-λ × (t_snapshot - t_edge))
```

Where:
- `t_snapshot`: snapshot cutoff in seconds
- `t_edge`: child node arrival timestamp in seconds (relative to root = 0)
- `λ`: decay rate (inverse seconds)

**Training Objective:** Combined loss:
```
L = α × SupConLoss + (1 - α) × CrossEntropyLoss
```

Where `SupConLoss` is Supervised InfoNCE — positive pairs are same-label embeddings within a batch, and `α` balances the two objectives.

**Snapshot-Aware Training (Option B):** The model is trained on **all 8 time windows** per cascade simultaneously, not just the final cascade. This produces different embeddings for the same cascade at different times, enabling Phase 13-14's adaptive stopping to consume per-(cascade_id, t) embeddings directly.

### 5.4 Hyperparameter Sweep

A grid search over **72 configurations** was conducted:

| Parameter | Values Swept |
|---|---|
| `embed_dim` | 32, 64, 128 |
| `lam` (λ) | 0.0, 0.0001, 0.0005, 0.001 |
| `n_layers` | 1, 2 |
| `alpha` | 0.3, 0.5, 0.7 |

**Fixed hyperparameters:**

| Parameter | Value |
|---|---|
| `hidden_dim` | 128 |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| `n_epochs` | 30 |
| `patience` | 8 |
| `batch_size` | 64 |
| `temperature` (τ) | 0.07 |
| `dropout` | 0.5 |

**Best configuration:** `embed_dim=32, lam=0.0005, n_layers=2, alpha=0.5` (val Macro F1 = 0.8834)

### 5.5 Adaptive Early Stopping (Phase 13-14)

**Architecture:** XGBoost regression model predicting a per-cascade, per-time dynamic threshold θ(t).

**Features used:**
- `t_minutes`, `node_count`, `max_depth`, `growth_velocity`
- `burstiness`, `branching_factor`, `mean_interarrival`
- `confidence_{model_type}` (CASCADE2VEC or KPG softmax probability)

**XGBoost sweep:** Grid search over `max_depth ∈ {3,5,7}`, `n_estimators ∈ {100,200}`, `lr ∈ {0.05, 0.1}` (12 configs total).

**Detection loop:** Iterates chronologically through the 8 time windows per cascade. At each window, if the model's confidence exceeds the adaptive threshold, the cascade is flagged and stopped early.

---

## 6. Experimental Results

### 6.1 Master Comparison Table

| Model | Type | Macro F1 | Accuracy | Weighted F1 | ROC-AUC | Runtime |
|---|---|---|---|---|---|---|
| **CASCADE2VEC** | Proposed | **0.8433** | 0.8610 | 0.8602 | 0.8762 | 2.94 min |
| **KPG-simplified** | SOTA Baseline | 0.8304 | 0.8426 | 0.8450 | 0.9164 | 0.25 min |
| **BiGCN** | SOTA Baseline | 0.8237 | 0.8346 | 0.8377 | 0.9203 | 0.27 min |
| **PGNN** | SOTA Baseline | 0.8237 | 0.8340 | 0.8373 | 0.9232 | 0.17 min |
| **RP-DNN** | SOTA Baseline | 0.7709 | 0.7915 | 0.7929 | 0.8609 | 0.11 min |
| Logistic Regression | Simple Baseline | 0.5443 | 0.5629 | 0.5737 | 0.5683 | < 1 min |
| XGBoost | Simple Baseline | 0.5104 | 0.5709 | 0.5655 | 0.5148 | < 1 min |
| Random Forest | Simple Baseline | 0.4689 | 0.5985 | 0.5529 | 0.4963 | < 1 min |

### 6.2 H1 Verdict: NOT SUPPORTED

CASCADE2VEC achieved the highest nominal Macro F1 (+0.0129 over KPG-simplified), but rigorous statistical testing showed this is **not statistically significant**:

**Multi-Seed Variance Check (5 seeds):**

| Config | Seeds | Mean F1 | Std |
|---|---|---|---|
| Best overall (λ=0.0005) | 42-46 | 0.8705 | 0.0054 |
| Best zero-decay (λ=0.0) | 42-46 | 0.8696 | 0.0045 |
| Third overall (λ=0.001) | 42-46 | 0.8681 | 0.0025 |

The standard deviation across seeds (~0.005) **exceeds** the gap between time-decay and zero-decay configs (0.0009), proving that λ provides **no statistical benefit**.

**McNemar's Test:** p = 0.1928 ≥ 0.05 (no significant disagreement between CASCADE2VEC and KPG predictions).

**Bootstrap 95% CI (1000 resamples):** CI of performance gap = [-0.0096, 0.0244] — **crosses zero**.

> **Interpretation:** This is a methodologically bulletproof **null result**. The temporal dynamics (exponential time-decay) assumed necessary by the literature can be entirely matched by simpler, static topological convolutions. The pipeline contamination incident was caught and resolved *before* this conclusion was drawn.

### 6.3 H2 Verdict: NOT SUPPORTED

**CASCADE2VEC Adaptive Stopping Results:**

| Strategy | Macro F1 | MDT (mins) | Median MDT | Early Stop % |
|---|---|---|---|---|
| **Adaptive** | 0.8404 | 79.22 | 120.0 | 34.3% |
| Fixed threshold | 0.8363 | 82.79 | 120.0 | 31.3% |
| Fixed t=120 | 0.8350 | 120.0 | 120.0 | 0.0% |
| Fixed t=30 | 0.8346 | 30.0 | 30.0 | 100.0% |

**MDT by class (Adaptive):**
- Rumour cascades: 25.67 minutes (detected early)
- Non-rumour cascades: 106.81 minutes (most wait until 120 min)

**H2 Criteria:** MDT reduction ≥ 10% vs. best fixed threshold. Actual reduction: ~4.3% (79.22 vs 82.79 min) — **below the 10% threshold**.

**Bootstrap 95% CI on MDT Gap:** Mean gap = 3.59 mins, CI = [2.62, 4.52]. The reduction is real but modest.

**KPG Adaptive Stopping (secondary check):**

| Strategy | Macro F1 | MDT (mins) | Early Stop % |
|---|---|---|---|
| Adaptive | 0.8288 | 72.44 | 40.0% |
| Fixed threshold | 0.8385 | 78.88 | 34.8% |

### 6.4 Phase 15: Explainability (XAI)

**SHAP Analysis (Tabular — XGBoost Adaptive Threshold):**
SHAP values were computed for the adaptive threshold model, identifying which structural/temporal features most influence the dynamic stopping threshold. Key outputs include global feature importance bar plots, beeswarm distributions, and dependence plots for top features.

**GNNExplainer (Graph — CASCADE2VEC):**
PyG's GNNExplainer was applied to a sample of 20 connected, non-singleton test cascades (10 rumour, 10 non-rumour).

> **Limitation:** ~358 singleton cascades and ~606 disconnected cascades were explicitly excluded from the GNNExplainer sample, as edge-attribution methods cannot produce meaningful results without connected edges.

**Aggregate GNNExplainer Statistics (20 cascades):**

| Metric | Mean | Range |
|---|---|---|
| Mean edge importance | 0.20 | 0.09 – 0.79 |
| Mean node importance | 0.33 | 0.14 – 0.71 |
| Max edge importance | 0.50 | 0.09 – 0.94 |

---

## 7. Pipeline Integrity & Reproducibility

### 7.1 Contamination Incident & Resolution

During the Phase 11-12 hyperparameter sweep, a **pipeline contamination incident** was discovered: the dataset pipeline had been updated, but the sweep checkpoint system silently resumed from old, stale results. Resolution:

1. Implemented strict `PIPELINE_VERSION` tracking tag (`v2-dict-snapshot`)
2. Hard-deleted all tainted sweep data
3. Re-ran the full 72-config sweep from scratch
4. All conclusions drawn *after* this resolution

### 7.2 Reproducibility Measures

- **Fixed seed:** 42 for all models (torch, numpy, random, CUDA)
- **Deterministic backends:** `torch.backends.cudnn.deterministic = True`
- **Single test-split access:** Test set evaluated exactly once per model after training
- **Version-tracked checkpoints:** All model checkpoints include pipeline version metadata
- **`--force` flag pattern:** All scripts require explicit `--force` to overwrite existing results

---

## 8. Software Engineering Details

### 8.1 Testing

```bash
pytest tests/ -v --tb=short
```

Test suites cover:
- PySpark BFS depth vs. Pandas reference implementation (exact match)
- Snapshot leakage assertions
- Feature engineering regression tests
- Cascade filter validity (singletons, disconnected)
- Model checkpoint serialization

### 8.2 Key Dependencies & Versions

```
pyspark==3.5.1          # Distributed graph construction
graphframes-py==0.9.0   # Graph algorithms (BFS, connected components)
torch==2.3.1            # Deep learning
torch-geometric==2.5.3  # GNN layers (SAGEConv, GCNConv, DataLoader)
scikit-learn==1.5.0     # Feature processing, metrics, splits
xgboost==2.0.3          # Adaptive threshold model
shap==0.45.1            # Tabular explainability
networkx==3.3           # Graph visualization (XAI plots)
ndlib==5.1.1            # SIR epidemic simulation (synthetic data)
```

### 8.3 Compute Environment

- **GPU:** CUDA-enabled (model auto-detects `cuda` vs `cpu`)
- **JVM:** Java 17 required for PySpark/GraphFrames
- **Package installation:** `pip install -e .` (editable install via hatchling)

---

## 9. Phase 16-17: Scalability (In Progress)

### 9.1 Data Volume Scaling

Synthetic cascades generated using PySpark with proper tree structures (random attachment model). Benchmarked at scales: 1K, 5K, 10K, 50K, 100K cascades.

Three pipeline stages timed:
1. Graph Construction (PySpark `to_vertices` + `to_edges`)
2. Feature Engineering (Pandas multiprocessing)
3. Model Inference (CASCADE2VEC forward pass)

### 9.2 Core Scaling (H4 Test)

Fixed 50K cascades, varying `local[1]`, `local[2]`, `local[4]`, `local[8]` parallelism. Measures speedup factor T₁/Tₚ against ideal linear speedup.

---

## 10. Phase 18: Ablation Study (Planned)

### 10.1 CASCADE2VEC Architecture Ablations

| ID | Description | Change |
|---|---|---|
| A1 | Remove time decay | λ = 0 |
| A2 | Replace attention pooling | Mean pooling instead |
| A3 | Remove contrastive loss | α = 0 (pure CE) |
| A4 | Remove classification loss | α = 1 (pure contrastive) |
| A5 | Single GNN layer | n_layers = 1 |
| A6 | Halved embedding dim | embed_dim = 16 |

### 10.2 Feature Subset Ablations

- Structural features only
- Temporal features only
- Top-5 SHAP features only
- All features (control)

---

## 11. Limitations

1. **Single dataset:** All results are on PHEME only. Generalization to Twitter15/16 or cross-platform data is untested.
2. **Simplified SOTA:** KPG uses static betweenness centrality instead of the original RL-based key-node selector. This may understate KPG's true performance.
3. **GNNExplainer coverage:** ~17% of cascades (singletons + disconnected) are excluded from graph-level explanations.
4. **Class imbalance:** Despite class weighting, the ~2:1 non-rumour:rumour ratio may bias predictions toward the majority class.
5. **Temporal scope:** PHEME events are from 2014-2015; linguistic patterns of misinformation may have evolved.

---

## 12. Key Takeaways

1. **Time-decay does not help (H1):** The temporal weighting mechanism — the core novelty of CASCADE2VEC — provides no statistically significant improvement over static graph convolutions. Seed variance dominates the signal.

2. **Adaptive stopping shows promise but is insufficient (H2):** The learned threshold reduces MDT by ~3.6 minutes (4.3%), with rumours detected in ~26 minutes on average. However, this falls short of the 10% MDT reduction criterion.

3. **Graph structure is powerful:** All GNN-based models (CASCADE2VEC, KPG, BiGCN, PGNN) vastly outperform tabular baselines (0.82-0.84 vs 0.47-0.54 Macro F1), confirming that propagation tree topology is the dominant signal for rumour detection.

4. **Scientific integrity matters:** The pipeline contamination incident and its resolution demonstrate the importance of version tracking, checkpoint hygiene, and multi-seed variance analysis before drawing conclusions.
