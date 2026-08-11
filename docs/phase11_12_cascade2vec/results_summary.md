# Phase 11-12: Cascade2Vec Results Summary

## 1. Hyperparameter Sweep (Clean Run - v2-dict-snapshot)
* **Best Config (with decay):** $\lambda = 0.0005$, `embed_dim=32`, `n_layers=2`, `alpha=0.5` $\to$ **Val Macro F1 = 0.8834**
* **Best Config (no decay):** $\lambda = 0.0$, `embed_dim=128`, `n_layers=2`, `alpha=0.3` $\to$ **Val Macro F1 = 0.8808**

The winning $\lambda$ value ($0.0005$) sits securely within our tested bounds `[0.0, 0.0001, 0.0005, 0.001]`, confirming the optimal time-decay factor was captured.

*(Full sweep table is permanently saved at `logs/phase11_12_cascade2vec/hyperparameter_sweep.md`)*

## 2. Variance Check
The top 3 configs were retrained across 5 random seeds (42-46) to test if the performance gaps found in the sweep were robust against random initialization noise.

* **best_overall** ($\lambda = 0.0005$): Mean F1 = 0.8697, Std = **0.0049**
* **best_zero_decay** ($\lambda = 0.0$): Mean F1 = 0.8696, Std = **0.0045**
* **third_overall** ($\lambda = 0.001$): Mean F1 = 0.8681, Std = **0.0025**

**Finding:** The standard deviation across seeds ($\approx 0.0045 - 0.0049$) is larger than the gap that separated these configurations in the sweep ($0.0026$). Based on the agreed statistical decision rule, **no significant difference is detected between these configs**. Time decay ($\lambda \neq 0$) does not yield a statistically distinct advantage over the zero-decay variant in this dataset.

## 3. Statistical Significance vs Baselines
The final CASCADE2VEC model was trained on the `best_overall` configuration (albeit with time-decay proven not to be a statistically significant factor). It was evaluated exactly once on the test split and compared against the best baseline (KPG-simplified).

* **CASCADE2VEC Test Macro F1:** 0.8388
* **KPG-simplified Test Macro F1:** 0.8311
* **Gap:** +0.0078

We performed two rigorous tests to determine if this gap represents a genuine improvement:

1. **McNemar's Test:**
   * **Statistic:** 1.6963
   * **p-value:** 0.19277 ($p \ge 0.05$)
   * **Conclusion:** NO significant disagreement between the models' predictions.

2. **Bootstrap 95% CI (1000 resamples):**
   * **Mean Bootstrapped Gap:** 0.0078
   * **95% Confidence Interval:** `[-0.0096, 0.0244]`
   * **Conclusion:** NO significant improvement (the CI comfortably crosses zero).

**Final Verdict on H1:** Although CASCADE2VEC achieves a marginally higher point estimate for Macro F1, the rigorous statistical tests confirm that it does **not** significantly outperform the KPG-simplified baseline. The initial hypothesis (H1) is **NOT SUPPORTED**.
