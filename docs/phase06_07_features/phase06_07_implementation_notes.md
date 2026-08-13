# Phase 6-7: Feature Engineering & Baseline Classifiers

**Status:** ✅ Complete
**Docs:**
- [`phase06_07_classification_protocol.md`](phase06_07_classification_protocol.md)
- [`../../logs/phase06_07_features/phase06_07_feature_dictionary.md`](../../logs/phase06_07_features/phase06_07_feature_dictionary.md)
- [`../../logs/phase06_07_features/phase06_07_baseline_results.md`](../../logs/phase06_07_features/phase06_07_baseline_results.md)
- [`../../logs/phase06_07_features/phase06_07_feature_correlation.md`](../../logs/phase06_07_features/phase06_07_feature_correlation.md)
- [`../../logs/phase06_07_features/phase06_07_leakage_report.md`](../../logs/phase06_07_features/phase06_07_leakage_report.md)

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

### Problems Encountered & Resolutions
- **Silent Dataset Overwrite (Critical):** `build_feature_matrix.py` silently overwrote the full 5,802-cascade production `feature_matrix.parquet` with a tiny 10-cascade sample when executed with `--limit 10` during testing.
  - *Fix:* Re-generated the full dataset (46,416 rows). Updated `build_feature_matrix.py` to enforce an `--output` path when `--limit` is passed, and added a `RuntimeError` guard preventing smaller matrices from overwriting larger ones without a `--force` flag.
- **Markdown Rendering Crash:** `baselines_simple.py` crashed loudly with an `ImportError` when attempting to render results into markdown tables via pandas.
  - *Fix:* Installed the missing `tabulate` dependency in the virtual environment.

---

### Change Log: Feature Selection Pipeline (Added Aug 2026)

**Problem:** The original 19-feature set contained significant redundancy. The existing
[feature correlation report](../../logs/phase06_07_features/phase06_07_feature_correlation.md)
flagged 10 highly correlated pairs (|r| > 0.95), including `tweets_per_minute` ↔ `growth_velocity`
(r = 1.0, literally identical up to ÷60) and `node_count` ↔ `edge_count` (r = 0.998).
Multicollinearity harms LR convergence, inflates tree-model feature importances, and makes
SHAP/LIME explanations ambiguous.

**What was done:**
A 3-stage feature selection pipeline was added in
[`feature_selection.py`](../../src/cascade2vec/phase06_07_features/feature_selection.py):

| Stage | Method | Purpose |
|-------|--------|---------|
| **Stage 1** | Correlation pruning (|r| > 0.95) | Drop one of each correlated pair, keeping the feature with higher univariate F-statistic (ANOVA) |
| **Stage 2** | RandomForest importance ranking | Rank surviving features by `feature_importances_` on train set only |
| **Stage 3** | Ablation comparison | Compare Full 19 → Pruned → Top-10 → Top-8 feature sets via StratifiedGroupKFold baselines |

**Files changed:**
- **[NEW]** `src/cascade2vec/phase06_07_features/feature_selection.py` — full pipeline
- **[NEW]** `tests/phase06_07_features/test_feature_selection.py` — unit tests for correlation pruning
- **[MODIFIED]** `src/cascade2vec/phase06_07_features/baselines_simple.py` — `run_baselines()` now accepts optional `selected_features` parameter (backward compatible)
- **[MODIFIED]** `src/cascade2vec/phase08_10_sota_baselines/compare_baselines.py` — auto-loads `selected_features.json` if present, else uses original 19

**What it improved:**
- Removed ~6-7 mathematically redundant features (e.g. `growth_velocity`, `edge_count`)
- Preserved interpretability for Phase 15 XAI (no opaque PCA components)
- Results comparison saved to `logs/phase06_07_features/feature_selection_report.md`
- Final selected feature set saved to `data/processed/phase06_07_features/selected_features.json`

**Design decision — PCA was NOT used as the primary method** because:
1. PCA creates opaque linear combinations that make SHAP/LIME explanations unintelligible
2. With only 19 features, correlation pruning alone removes the redundancy
3. PCA is included as a comparison point in the Stage 3 ablation only

**Important Architectural Note: Scope of Feature Selection**
> [!IMPORTANT]
> **The feature selection pipeline ONLY applies to the Simple Baselines (Logistic Regression, Random Forest, XGBoost) and Phase 15 XAI.**
> It does **NOT** apply to the Phase 8-10 SOTA baselines (BiGCN, PGNN, etc.) or Phase 11-12 CASCADE2VEC. 
> 
> *Why?* The SOTA baselines and CASCADE2VEC are Graph Neural Networks (GNNs). They bypass these 19 manually engineered features entirely, instead operating directly on the raw propagation tree edges and raw TF-IDF text embeddings to learn their own internal feature representations during training.

---

### Change Log: Cascade Visualization Tooling (Added Aug 2026)

**Problem:** There was no way to visually inspect the tweet propagation trees or see how
models separate rumour vs non-rumour in embedding space.

**What was done:**
Added [`visualize_cascades.py`](../../src/cascade2vec/phase06_07_features/visualize_cascades.py)
which generates:

1. **Cascade propagation tree grid** — 12 sample cascades from the test/val set (6 rumour,
   6 non-rumour) rendered as NetworkX directed graphs. Root tweets highlighted in gold,
   rumour trees in coral, non-rumour trees in teal.
2. **t-SNE feature embedding plot** — all test-set cascades at t=120min projected to 2D via
   PCA(10) → t-SNE(2), color-coded by label. Uses premium dark theme.

**Files changed:**
- **[NEW]** `src/cascade2vec/phase06_07_features/visualize_cascades.py`

**Output:**
- `logs/visualizations/cascade_trees_test.png`
- `logs/visualizations/tsne_features_test.png`
- `logs/visualizations/cascade_trees_val.png` (if val split exists)
- `logs/visualizations/tsne_features_val.png` (if val split exists)

