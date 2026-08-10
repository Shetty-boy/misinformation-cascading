# CASCADE2VEC — Implementation Notes (All Phases)

This file consolidates implementation decisions, key facts, and status for
every completed phase. It is the single reference document for the methods
section of any paper or handoff.

**Project:** Misinformation Cascade Detection (CASCADE2VEC)  
**Dataset:** PHEME (5,802 cascades, 102,440 tweets, 9 events)  
**Primary Task:** Rumour Detection — binary (rumour vs non-rumour)  
**Seed:** 42 (all phases)

---

## Phase 1: Data Acquisition

**Status:** ✅ Complete  
**Source doc:** [`docs/phase01_data_acquisition/design_doc.md`](docs/phase01_data_acquisition/design_doc.md)

### What Was Done
- Acquired PHEME dataset from Zubiaga et al. (2016)
- Events: charliehebdo, ebola, ferguson, germanwings-crash, gurlitt, ottawashooting, prince-toronto, putinmissing, sydneysiege
- Organised raw JSON into project directory structure

### Key Decisions
- Raw data kept intact under `data/raw/` — never modified in place
- All transformations go through Phase 2 ingestion pipeline

---

## Phase 2: Data Ingestion

**Status:** ✅ Complete  
**Log:** [`logs/phase02_ingestion/data_audit.md`](logs/phase02_ingestion/data_audit.md)

### What Was Done
- Parsed all PHEME JSON files into a unified flat parquet
- Output: `data/processed/phase02_ingestion/unified.parquet`

### Schema
| Column | Type | Description |
|---|---|---|
| `tweet_id` | str | Unique tweet identifier |
| `cascade_id` | str | Root tweet ID (cascade group key) |
| `parent_id` | str/NaN | Parent tweet ID (NaN for root) |
| `text` | str | Tweet text |
| `timestamp` | datetime | UTC timestamp |
| `label` | str | `rumour` / `non-rumour` (thread-level) |
| `event` | str | PHEME event name |

### Dataset Statistics
| Metric | Value |
|---|---|
| Total cascades | 5,802 |
| Total tweets | 102,440 |
| Rumour cascades | 1,972 (34.0%) |
| Non-rumour cascades | 3,830 (66.0%) |
| Class imbalance | 1.94:1 (non-rumour:rumour) |

### Key Decisions
- Labels come from PHEME folder structure (`rumours/` vs `non-rumours/`), not tweet content
- Cascade-level label only — every tweet in a cascade shares the same label
- Task B (veracity classification: true/false/unverified) deferred indefinitely — not the target task

---

## Phase 3: EDA & Leakage Audit

**Status:** ✅ Complete  
**Doc:** [`docs/phase03_eda_leakage/validated_features.md`](docs/phase03_eda_leakage/validated_features.md)

### What Was Done
- Exploratory analysis of cascade structure, text, and temporal patterns
- Leakage audit: identified which features are safe to compute without lookahead

### Key Decisions
- All temporal features must be computed within a time window `t` — no using future timestamps
- Text features from replies cannot use reply-level labels (would be leakage)
- Reachability (`reachable_ratio`) and connectivity (`is_connected`) are safe per-snapshot metrics

---

## Phase 4-5: Graph Construction & Depth Computation

**Status:** ✅ Complete  
**Log:** [`logs/phase04_05_graph/graph_stats.md`](logs/phase04_05_graph/graph_stats.md)

### What Was Done
- Built propagation tree graph from `parent_id` structure
- Implemented BFS depth computation in PySpark (`compute_depths`) and Pandas (`compute_depths_pandas`)
- Verified both implementations produce **exactly identical results** on 5 regression cascades

### Implementation
- `src/cascade2vec/phase04_05_graph/build_graph.py` — graph construction
- `src/cascade2vec/phase04_05_graph/depth.py` — PySpark BFS
- `src/cascade2vec/phase04_05_graph/depth_pandas.py` — Pandas BFS (regression reference)

### Dataset Graph Statistics
| Metric | Value |
|---|---|
| Total cascades | 5,802 |
| Singleton cascades | 358 |
| Fully connected cascades | 5,193 |
| Mean node count | 17.7 |
| Mean edge count | 16.4 |
| Mean max depth | 3.4 |

### Key Decisions
- **DAG guarantee:** Temporal ordering (reply always chronologically after parent) + in-degree ≤ 1 → forest structure, no cycles possible. No cycle-detection needed in BFS.
- 1,460 orphaned edges dropped (parent_id not found in same cascade's vertex set)
- `persist()` used in BFS loop instead of `localCheckpoint()` to avoid Py4J re-entry in test environments

### Bugs Fixed
- BFS was originally returning flat depth=0.0 for all nodes — fixed by correcting frontier propagation logic
- Comment in `depth.py` now correctly states the DAG guarantee as "temporal ordering + in-degree ≤ 1 → forest"

---

## Phase 6-7: Feature Engineering & Baseline Classifiers

**Status:** ✅ Complete  
**Docs:**  
- [`docs/phase06_07_features/classification_protocol.md`](docs/phase06_07_features/classification_protocol.md)  
- [`logs/phase06_07_features/feature_dictionary.md`](logs/phase06_07_features/feature_dictionary.md)  
- [`logs/phase06_07_features/baseline_results.md`](logs/phase06_07_features/baseline_results.md)  
- [`logs/phase06_07_features/feature_correlation.md`](logs/phase06_07_features/feature_correlation.md)  
- [`logs/phase06_07_features/leakage_report.md`](logs/phase06_07_features/leakage_report.md)

### Feature Set (19 features)
Computed per `(cascade_id, t)` snapshot — no future lookahead.

| Feature | Description |
|---|---|
| `node_count` | Nodes seen up to time t |
| `edge_count` | Edges in snapshot |
| `max_depth` | Maximum BFS depth |
| `avg_depth` | Mean BFS depth |
| `leaf_count` | Nodes with no children |
| `leaf_ratio` | leaf_count / node_count |
| `branching_factor` | Avg children per non-leaf node |
| `root_degree` | Out-degree of root node |
| `reachable_ratio` | BFS-reachable nodes / total nodes |
| `is_connected` | 1 if reachable_ratio == 1.0 |
| `tweets_per_minute` | node_count / cascade_age |
| `growth_velocity` | Δ(node_count) / Δt |
| `mean_interarrival` | Mean gap between consecutive tweets |
| `std_interarrival` | Std of interarrival gaps |
| `burstiness` | (std - mean) / (std + mean) |
| `cascade_age` | Elapsed time since root tweet |
| `depth_velocity` | Δ(max_depth) / Δt |
| `breadth_velocity` | Δ(leaf_count) / Δt |
| `branching_velocity` | Δ(branching_factor) / Δt |

### Leakage Rules
- `is_connected` computed per `(cascade_id, t)` snapshot — varies dynamically across time ✅
- No future timestamps used in any feature ✅
- TF-IDF (Phase 8-10) vocabulary built on training split only ✅
- Feature pruning deferred to Phase 18 ablations — no features silently removed here

### Simple Baseline Results (StratifiedGroupKFold, 5-fold)

**Full Dataset:**
| Model | Macro F1 | Weighted F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression (no weights) | 0.4037 | 0.5285 | 0.5696 |
| Random Forest (no weights) | 0.4871 | 0.5700 | 0.5238 |

**Disconnected Cascades Excluded:**
| Model | Macro F1 | Weighted F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression (no weights) | 0.4004 | 0.5257 | 0.5646 |
| Random Forest (no weights) | 0.4867 | 0.5685 | 0.5205 |

*Note: Simple baselines use per-snapshot structural features. The ~0.40 Macro F1 is the floor that Phase 8-10 SOTA models must beat.*

### Key Decisions
- Evaluation: `StratifiedGroupKFold(n_splits=5, groups=cascade_id)` — no cascade appears in both train and test
- Class weights tested but did not consistently improve performance
- Two separate reported runs (full dataset vs. disconnected excluded) — both in `baseline_results.md`

---

## Phase 8-10: SOTA Baseline Reimplementation

**Status:** ✅ Complete  
**Docs:**  
- [`docs/phase08_10_sota_baselines/data_interface_contract.md`](docs/phase08_10_sota_baselines/data_interface_contract.md)  
- [`docs/phase08_10_sota_baselines/implementation_notes.md`](docs/phase08_10_sota_baselines/implementation_notes.md)  
- [`logs/phase08_10_sota_baselines/sota_comparison.md`](logs/phase08_10_sota_baselines/sota_comparison.md)

### Models Implemented

| Model | Source | Approach |
|---|---|---|
| Bi-GCN | Adapted from [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews) (MIT, Dec 2025) | Two-branch GCN (TD + BU) |
| RP-DNN | Built from scratch (no public PyTorch repo) | BiGRU (text) + GRU (structural sequence) |
| PGNN | Built from scratch (no public PyTorch repo) | GCN + soft attention pooling |
| KPG-simplified | Built from scratch (original RL repo unusable) | Static centrality key-node selection + GCN |

### Final Results (Test Set — fixed 70/15/15 split, seed=42)

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC | Runtime |
|---|---|---|---|---|---|
| KPG-simplified | 0.846 | **0.831** | 0.847 | 0.919 | 0.25 min |
| Bi-GCN | 0.835 | **0.824** | 0.838 | 0.920 | 0.27 min |
| PGNN | 0.834 | **0.824** | 0.837 | 0.923 | 0.17 min |
| RP-DNN | 0.792 | **0.771** | 0.793 | 0.861 | 0.11 min |

All models exceed the 0.40 Macro F1 floor. All checkpoints saved to `data/processed/phase08_10_sota_baselines/checkpoints/`.

### Key Decisions
- **Full cascades only** — SOTA baselines trained on complete final propagation trees, matching the original papers' benchmark setup
- **No snapshot-truncated evaluation** — if Phase 13-14 needs SOTA baselines at early time cutoffs, separate work is required
- **70/15/15 hold-out split** (vs. 5-fold CV for simple baselines) — justified by deep model training cost
- **Test split accessed exactly once** per model — hard rule enforced in every training script
- **Class-weighted CrossEntropyLoss** for all models
- **KPG-simplified** uses static betweenness centrality (not RL) — documented as independent simplification, NOT an attributed paper ablation
- Gradient clipping at max_norm=1.0 for training stability

### Known Gap
The simple baseline rows in `sota_comparison.md` use 4 basic structural features (derived from `unified.parquet`) because `feature_matrix.parquet` is partial (only 79 rows). The full 19-feature matrix needs to be regenerated on the full dataset for a fully faithful simple-vs-SOTA comparison. This does not affect the SOTA model numbers.

---

## Phases 11-18: Not Yet Started

| Phase | Description | Status |
|---|---|---|
| 11-12 | CASCADE2VEC embedding (main model) | 🔲 Not started |
| 13-14 | Adaptive early-stopping | 🔲 Not started |
| 15 | XAI / explainability | 🔲 Not started |
| 16-17 | Scalability | 🔲 Not started |
| 18 | Evaluation & ablations | 🔲 Not started |

---

*Last updated: Phase 8-10 complete (2026-08-10)*
