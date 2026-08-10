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
