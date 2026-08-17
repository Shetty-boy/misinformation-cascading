# Phase 13-14: Adaptive Early Stopping (H2)

## H2 Verdict: NOT SUPPORTED

**Criteria for Support:** 
- Mean Detection Time (MDT) reduced by ≥10% compared to best fixed threshold.
- Macro F1 at detection time must match or exceed fixed threshold (within -0.01 margin).

### Key Results (CASCADE2VEC - Headline)
- **Adaptive MDT:** 79.22 mins (Median: 120.00 mins)
- **Fixed Best MDT:** 82.79 mins (Median: 120.00 mins)
- **Fixed t=120 MDT:** 120.00 mins

- **Adaptive Macro F1:** 0.8404
- **Fixed Best Macro F1:** 0.8363
- **Fixed t=120 Macro F1:** 0.8350

- **% Cascades Stopped Early (Adaptive):** 34.3%

### Statistical Testing (Bootstrap 95% CI on MDT Gap)
- **Mean Gap (Fixed - Adaptive):** 3.59 mins
- **95% CI:** [2.62, 4.52]
- Significant time reduction observed (CI > 0).

### Secondary Check (KPG-Simplified)
- **Adaptive MDT:** 72.44 mins
- **Fixed Best MDT:** 78.88 mins
- **Adaptive Macro F1:** 0.8288
