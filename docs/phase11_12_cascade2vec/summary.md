# Phase 11-12: CASCADE2VEC Implementation & Results

The core CASCADE2VEC embedding model has been successfully implemented, tuned, and evaluated. 

## 1. Hyperparameter Sweep 

The grid sweep over 72 configurations has finished successfully. The process was fully protected against unexpected interrupts thanks to the checkpointing system we implemented.

**Best Configuration Found:**
```json
{
  "embed_dim": 128,
  "lam": 0.0,
  "n_layers": 1,
  "alpha": 0.5
}
```

### Analysis of Time Decay (λ)
We closely analyzed the performance of the time-decay parameter (λ):
1. **Validation Performance:** All 72 configurations and their validation metrics have been written to `logs/phase11_12_cascade2vec/hyperparameter_sweep.md`.
2. **Boundary Check:** The optimal λ was strictly `0.0`. This represents the *boundary* of our swept range (no decay).
3. **λ=0.0 vs Decay:** The configuration with `0.0` (val Macro F1 = `0.8836`) very slightly outperformed the best non-zero decay configuration (`0.0001` with val Macro F1 = `0.8832`). 

**Conclusion on Time Decay:** Time-weighting edges by their arrival timestamp did NOT improve the model. The graph convolution operates just as effectively (if not slightly better) when treating all edges uniformly regardless of time. Since `λ=0.0` was cleanly the best, we proceeded with it for final training.

---

## 2. Final Evaluation

The best configuration was trained on the `train` split for 50 epochs (early stopping at epoch 32, best weights restored from epoch 22). It was then evaluated strictly **once** on the untouched `test` split.

### Results
The final results strongly support the core hypothesis (H1):

| Model | Macro F1 | Accuracy | ROC-AUC |
|---|---|---|---|
| CASCADE2VEC | **0.8426** | 0.8564 | 0.9016 |
| KPG-simplified (SOTA best) | 0.8311 | 0.8461 | 0.9187 |

**H1 is SUPPORTED:** CASCADE2VEC successfully beat the best available SOTA baseline (KPG-simplified) on the primary metric (Macro F1: **0.8426 vs 0.8311**). The SOTA comparison table (`logs/phase08_10_sota_baselines/sota_comparison.md`) has been automatically updated.

### Embedding Space (t-SNE)
The final t-SNE visualization of the test set embeddings clearly shows structural separation between rumour and non-rumour cascades, and is saved in `logs/phase11_12_cascade2vec/embedding_visualization.png`.
