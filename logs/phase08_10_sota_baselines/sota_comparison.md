# Phase 8-10: SOTA vs. Simple Baseline Comparison

Task: **Rumour Detection** (rumour vs non-rumour)
Evaluation: Fixed 70/15/15 hold-out split (cascade-level stratified)
Test split accessed EXACTLY ONCE per model after training/selection on val.
All models trained on FULL final cascades (no temporal truncation).
Seed: 42 for all models.

## Results

| Model | Type | Accuracy | Macro F1 | Weighted F1 | ROC-AUC | Runtime (min) |
|---|---|---|---|---|---|---|
| KPG-simplified | SOTA Baseline | 0.8461 | **0.8311** | 0.8472 | 0.9187 | 0.25 |
| BiGCN | SOTA Baseline | 0.8346 | **0.8237** | 0.8377 | 0.9203 | 0.27 |
| PGNN | SOTA Baseline | 0.834 | **0.8237** | 0.8373 | 0.9232 | 0.17 |
| RP-DNN | SOTA Baseline | 0.7915 | **0.7709** | 0.7929 | 0.8609 | 0.11 |
| Logistic Regression | Simple Baseline | 0.5629 | **0.5443** | 0.5737 | 0.5683 | < 1 |
| XGBoost | Simple Baseline | 0.5709 | **0.5104** | 0.5655 | 0.5148 | < 1 |
| Random Forest | Simple Baseline | 0.5985 | **0.4689** | 0.5529 | 0.4963 | < 1 |

## Notes

- **Macro F1 floor:** All SOTA models must exceed ~0.40 (Phase 6-7 simple baseline Macro F1). All models above satisfied this criterion.
- **KPG-simplified** uses static betweenness centrality key-node selection (K=20), not the RL-based training of the original KPG paper. This is an independent simplification; see `implementation_notes.md`.
- **Bi-GCN** adapted from safe-graph/GNN-FakeNews (MIT license, updated Dec 2025).
- **RP-DNN** and **PGNN** built from scratch (no official public PyTorch repos available).
- Simple baselines use per-cascade last-snapshot features from the Phase 6-7 feature matrix. SOTA baselines use raw propagation tree structure + TF-IDF text.
- See `data_interface_contract.md` for full-cascade vs. snapshot policy.
