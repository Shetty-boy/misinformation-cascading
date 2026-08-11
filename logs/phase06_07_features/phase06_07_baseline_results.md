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
| Random Forest       | No              |     0.6214 |     0.4915 |        0.5738 |    0.5251 |
| XGBoost             | No              |     0.6486 |     0.4525 |        0.5574 |    0.5515 |
| Logistic Regression | Yes             |     0.5597 |     0.5448 |        0.5711 |    0.5697 |
| Random Forest       | Yes             |     0.6133 |     0.4929 |        0.572  |    0.4994 |
| XGBoost             | Yes             |     0.5779 |     0.5434 |        0.5835 |    0.5513 |


### Disconnected Cascades Excluded

| Model               | Class Weights   |   Accuracy |   Macro F1 |   Weighted F1 |   ROC-AUC |
|:--------------------|:----------------|-----------:|-----------:|--------------:|----------:|
| Logistic Regression | No              |     0.6577 |     0.4006 |        0.5259 |    0.5661 |
| Random Forest       | No              |     0.6189 |     0.4924 |        0.5732 |    0.5284 |
| XGBoost             | No              |     0.647  |     0.4545 |        0.5579 |    0.5539 |
| Logistic Regression | Yes             |     0.5553 |     0.5409 |        0.5666 |    0.5662 |
| Random Forest       | Yes             |     0.6164 |     0.4979 |        0.5757 |    0.5056 |
| XGBoost             | Yes             |     0.5773 |     0.5413 |        0.5823 |    0.5516 |


---

## Notes
- All snapshots of a cascade always go to the same fold (group=cascade_id)
  to prevent cascade-level leakage.
- No F1 threshold is enforced — values are reported exactly as produced.
- XGBoost scale_pos_weight=1.94 used for the weighted run (non-rumour:rumour ratio).
- Pruning of features is deferred to Phase 18 ablations.
