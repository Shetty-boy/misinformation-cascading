# Phase 15: Explainability (XAI) Design Notes

## Methodology

We employ two complementary techniques for explainability because our pipeline relies on a combination of a Graph Neural Network (CASCADE2VEC) and tabular classifiers (XGBoost for Adaptive Thresholding).

### 1. Tabular XAI via SHAP
For the adaptive thresholding model (H2), we use SHAP (SHapley Additive exPlanations). Since it's a tree-based XGBoost model, `TreeExplainer` provides exact SHAP values efficiently. 

**Why SHAP?**
It gives global feature importance and granular local explanations (e.g., this cascade was stopped early because `burstiness` was exceptionally high).

### 2. Graph XAI via GNNExplainer
For the underlying CASCADE2VEC embeddings, tabular XAI is insufficient because the input is a raw graph structure, not tabular features. We use PyTorch Geometric's `GNNExplainer` to identify the most critical nodes and edges that led to a specific rumour/non-rumour prediction.

**Limitation: Singletons & Disconnected Cascades**
`GNNExplainer` operates by learning a soft mask over edges to maximize mutual information with the prediction.
Phase 4-5 EDA revealed:
- ~358 singleton cascades (0 edges)
- ~606 disconnected cascades (multiple connected components)

For singletons, edge masking is undefined. For severely disconnected cascades, the explanation is often degenerate. Therefore, we explicitly filter the evaluation set for GNNExplainer to only sample from connected, non-singleton cascades. This is a known limitation of edge-attribution techniques on real-world noisy graph data.
