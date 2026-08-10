# Phase 7: Baseline Results

Task: **Rumour Detection** (rumour vs non-rumour)
Cross-validation: StratifiedGroupKFold(n_splits=5), groups=cascade_id
Primary metric: Macro F1

> Class imbalance: ~1.94:1 non-rumour:rumour. Results reported BOTH with
> and without class weights so the effect of imbalance handling is visible.

---

### Full Dataset

| Model               | Class Weights   |   Accuracy |   Macro F1 |   Weighted F1 |   ROC-AUC |
|:--------------------|:----------------|-----------:|-----------:|--------------:|----------:|
| Logistic Regression | No              |     0.6584 |     0.4037 |        0.5285 |    0.5696 |
| Random Forest       | No              |     0.6178 |     0.4871 |        0.57   |    0.5238 |
| XGBoost             | No              |     0.6481 |     0.4519 |        0.5569 |    0.5541 |
| Logistic Regression | Yes             |     0.5597 |     0.5448 |        0.5711 |    0.5697 |
| Random Forest       | Yes             |     0.6144 |     0.4937 |        0.5728 |    0.5007 |
| XGBoost             | Yes             |     0.5718 |     0.5375 |        0.5778 |    0.5475 |


### Disconnected Cascades Excluded

| Model               | Class Weights   |   Accuracy |   Macro F1 |   Weighted F1 |   ROC-AUC |
|:--------------------|:----------------|-----------:|-----------:|--------------:|----------:|
| Logistic Regression | No              |     0.6578 |     0.4004 |        0.5257 |    0.5646 |
| Random Forest       | No              |     0.6147 |     0.4867 |        0.5685 |    0.5205 |
| XGBoost             | No              |     0.6461 |     0.4548 |        0.5578 |    0.5504 |
| Logistic Regression | Yes             |     0.5528 |     0.538  |        0.5643 |    0.5645 |
| Random Forest       | Yes             |     0.6125 |     0.492  |        0.5709 |    0.4993 |
| XGBoost             | Yes             |     0.5732 |     0.5352 |        0.5776 |    0.5453 |


---

## Notes
- All snapshots of a cascade always go to the same fold (group=cascade_id)
  to prevent cascade-level leakage.
- No F1 threshold is enforced — values are reported exactly as produced.
- XGBoost scale_pos_weight=1.94 used for the weighted run (non-rumour:rumour ratio).
- Pruning of features is deferred to Phase 18 ablations.
