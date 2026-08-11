# Complete Project Results Index

This document serves as the central hub for all readable, human-facing result summaries generated across every phase of the project.

## Phase 11-12: CASCADE2VEC Implementation
- **High-Level Summary:** [summary.md](file:///home/omen/projects/misinformation-cascading/docs/phase11_12_cascade2vec/summary.md) (Contains the narrative walkthrough, H1 claim status, and t-SNE plot)
- **Detailed Hyperparameter Sweep Log:** [hyperparameter_sweep.md](file:///home/omen/projects/misinformation-cascading/logs/phase11_12_cascade2vec/hyperparameter_sweep.md) (Contains all 72 configurations and their validation metrics)
- **Raw Metrics JSON:** [c2v_results.json](file:///home/omen/projects/misinformation-cascading/logs/phase11_12_cascade2vec/c2v_results.json)

## Phase 08-10: SOTA Baselines (KPG, BiGCN, PGNN, RP-DNN)
- **Master SOTA Comparison Table:** [sota_comparison.md](file:///home/omen/projects/misinformation-cascading/logs/phase08_10_sota_baselines/sota_comparison.md) (This is the primary readable markdown table containing Accuracy, Macro F1, Weighted F1, and ROC-AUC for all SOTA models, as well as the final CASCADE2VEC row).
- **Implementation Notes:** [implementation_notes.md](file:///home/omen/projects/misinformation-cascading/docs/phase08_10_sota_baselines/implementation_notes.md)
- **Raw Metrics:** Found in `logs/phase08_10_sota_baselines/` as individual JSON files for each model.

## Phase 06-07: Feature Matrix & Simple Baselines (Logistic Regression, XGBoost, RF)
- **Simple Baseline Results:** These are also tracked on the master SOTA table above ([sota_comparison.md](file:///home/omen/projects/misinformation-cascading/logs/phase08_10_sota_baselines/sota_comparison.md)) for easy comparison against the SOTA models.
- **Feature Matrix Notes:** [feature_matrix_notes.md](file:///home/omen/projects/misinformation-cascading/docs/phase06_07_features/feature_matrix_notes.md)

## Phase 04-05: Graph Construction
- **Graph Policy & Interface:** [data_interface_contract.md](file:///home/omen/projects/misinformation-cascading/docs/phase04_05_graph/data_interface_contract.md) (Explains how full-cascade vs snapshot truncation was enforced).

## Phase 03: EDA & Leakage Prevention
- **Exploratory Data Analysis Notes:** [eda_notes.md](file:///home/omen/projects/misinformation-cascading/docs/phase03_eda_leakage/eda_notes.md)

## Phase 02: Ingestion & Preprocessing
- **Ingestion Log:** `logs/phase02_ingestion/ingestion_report.txt` (Contains cascade filtering statistics and dataset sizes).
