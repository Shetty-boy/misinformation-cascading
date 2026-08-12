# Misinformation Cascading: Project Overview

## 1. Project Summary

**The Research Problem:** Early misinformation and rumour detection from Twitter reply-cascades. The goal is to accurately classify a cascade as a rumour or non-rumour based on its topological structure and text content before it fully unfolds, minimizing the time it takes to flag harmful content. We use the PHEME dataset for primary development.

**Core Hypotheses:**
- **H1 (Representation - *Tested*):** A time-weighted graph embedding (CASCADE2VEC), where edge influence decays based on temporal arrival, separates rumours from non-rumours more effectively than static state-of-the-art (SOTA) graph models (RP-DNN, PGNN, BiGCN, KPG).
- **H2 (Adaptive Early-Stopping - *Pending*):** A learned, per-cascade confidence threshold $\theta(t)$ can reliably flag rumours earlier than a fixed observation window, yielding equal accuracy with lower mean time-to-detection.

**Current Status:** The project has completed through Phase 12. We have successfully implemented a fully distributed data ingestion and graph-building pipeline, benchmarked both simple ML models and advanced PyTorch SOTA baselines, and implemented the custom CASCADE2VEC architecture. Rigorous statistical tests have concluded that H1 is **not supported** by the current evidence (CASCADE2VEC matched, but did not significantly beat, the SOTA baseline). The project is now ready to begin testing H2 (Adaptive Early-Stopping).

---

## 2. Repository Structure

```text
misinformation-cascading/
├── README.md                  # Quickstart guide and high-level findings
├── pyproject.toml             # Package configuration
├── requirements.txt           # Pinned dependencies
├── src/cascade2vec/           # Core library code
│   ├── phase02_ingestion/     # Parsing raw Twitter data to Parquet
│   ├── phase04_05_graph/      # PySpark graph construction and BFS
│   ├── phase06_07_features/   # Feature engineering & baseline evaluation
│   ├── phase08_10_sota_baselines/ # Implementation of BiGCN, KPG, PGNN, RP-DNN
│   ├── phase11_12_cascade2vec/    # Custom time-weighted GraphSAGE model
│   ├── phase13_14_adaptive_stopping/ # (Pending) Early detection loop
│   ├── phase15_xai/           # (Pending) Explainability
│   ├── phase16_17_scalability/ # (Pending) Synthetic graphs
│   └── phase18_eval/          # (Pending) Final ablations
├── data/                      
│   ├── raw/pheme/             # The source PHEME dataset
│   └── processed/             # Parquet files (graphs, features)
├── tests/                     # Unit and regression tests
├── docs/                      # Extensive design and summary documentation
├── logs/                      # Artifacts, test outputs, and JSON results
├── experiments/               # Training scripts and yaml configs
└── notebooks/                 # Jupyter notebooks for EDA and validation
```

### Directory Explanations
- **`src/cascade2vec/phaseNN_*`**: The project is structured linearly by execution phases. Code is isolated by phase to guarantee reproducible steps matching the research roadmap.
- **`data/` & `logs/`**: Keeps heavy artifacts (parquets, tensors, run logs) isolated from code.
- **`docs/` & `notebooks/`**: Contains phase-specific implementation notes, EDA, and architectural design documents.
- **`tests/`**: Contains `pytest` suites ensuring structural integrity across phases (e.g., verifying PySpark graphs match Pandas, confirming snapshot loader regression safety).
- **`pyproject.toml` / `pip install -e .`**: This setup allows all code to cleanly import from `cascade2vec.*` globally, resolving complex Python path issues across notebooks, tests, and deep nested scripts.

---

## 3. Phase-by-Phase Walkthrough

### Phase 1-3: Data Acquisition, EDA & Leakage Audit
- **What it does:** Ingests the raw PHEME dataset, parses JSON into flattened Parquet tables, and audits the data for structural/temporal leakage.
- **Key Files:** `src/cascade2vec/phase02_ingestion/schema.py`, `src/cascade2vec/phase02_ingestion/ingest.py`.
- **Decisions:** We focused exclusively on PHEME as the primary development dataset because it is self-contained (text + structure are released), bypassing the need to re-hydrate deleted tweets from the Twitter API (a major issue with Twitter15/16). "Rumour vs. Non-rumour" was chosen over veracity classification because verifying truth often takes days, whereas identifying *that* a cascade is a rumour can trigger early interventions.
- **Bugs Caught:** We immediately discovered severe risks of temporal and label leakage during EDA. To fix this, we implemented strict dynamic time windows (`t_minutes`) to prevent future lookahead.

### Phase 4-5: Graph Construction & Snapshots
- **What it does:** Converts the flattened tables into graph edge-lists and computes Breadth-First-Search (BFS) depths using PySpark.
- **Key Files:** `src/cascade2vec/phase04_05_graph/build_graph.py`, `src/cascade2vec/phase04_05_graph/depth.py`.
- **Decisions:** PySpark was chosen for BFS to ensure the pipeline scales to massive synthetic graphs later in the project. 
- **Bugs Caught:** The PySpark BFS initially returned a flat depth of `0.0` for all nodes due to a frontier propagation error. We fixed this and wrote a Pandas-based reference BFS (`depth_pandas.py`), adding a strict unit test verifying both implementations produce identical results. We also experienced JVM crashes due to `.localCheckpoint()`, which we resolved by migrating to `.persist()`.

### Phase 6-7: Feature Engineering & Simple Baselines
- **What it does:** Extracts structural (e.g., branching factor) and content (TF-IDF) features to train simple ML baselines (Logistic Regression, Random Forest, XGBoost).
- **Key Files:** `src/cascade2vec/phase06_07_features/engineering.py`, `src/cascade2vec/phase06_07_features/baselines_simple.py`.
- **Decisions:** Used a 5-fold Stratified Group K-Fold cross-validation strategy, grouping by `cascade_id` so snapshots from the same cascade never cross train/val boundaries. Class weights were introduced to combat the ~1.94:1 class imbalance.
- **Results:** Logistic Regression achieved the highest baseline Macro F1 (0.5448).

### Phase 8-10: SOTA Baselines
- **What it does:** Re-implements four state-of-the-art deep learning models from the literature (BiGCN, PGNN, RP-DNN, KPG). 
- **Key Files:** `src/cascade2vec/phase08_10_sota_baselines/` (one file per model architecture) and `split_data.py`.
- **Decisions:** Switched from 5-fold CV to a fixed 70/15/15 train/val/test split, standard practice for deep neural networks due to the high computational cost of 5-fold training. We implemented a *simplified* KPG (using static betweenness centrality instead of unstable Reinforcement Learning) because the original repo was unusable. 
- **Bugs Caught:** We caught a silent file-overwrite bug where `split_data.py` overwrote splits without warning. We fixed it by requiring `--force` flags.

### Phase 11-12: CASCADE2VEC
- **What it does:** Implements the custom time-weighted GraphSAGE model, optimizes hyperparameter sweeping, and evaluates final performance.
- **Key Files:** `src/cascade2vec/phase11_12_cascade2vec/cascade2vec.py`, `sweep.py`, `significance_test.py`.
- **Decisions:** Used an extensive grid sweep (72 configs) to find the optimal exponential time decay factor ($\lambda$). 
- **Bugs Caught:** We discovered a massive **pipeline contamination incident** mid-sweep. The dataset pipeline had been updated, but the `sweep.py` checkpoint system silently resumed from old, stale results. We implemented a strict `PIPELINE_VERSION` tracking tag, hard-deleted the tainted data, and re-ran the entire 72-config sweep to guarantee scientific integrity.

---

## 4. Key Results Table

This consolidated table shows the performance of all 8 evaluated models on the PHEME dataset (using the fixed test split where applicable).

| Model | Type | Macro F1 | Accuracy | Weighted F1 | ROC-AUC |
|---|---|---|---|---|---|
| **CASCADE2VEC** | Proposed Model | **0.8388** | 0.8570 | 0.8561 | 0.8900 |
| **KPG-simplified** | SOTA Baseline | 0.8311 | 0.8461 | 0.8472 | 0.9187 |
| **PGNN** | SOTA Baseline | 0.8237 | 0.8340 | 0.8373 | 0.9232 |
| **BiGCN** | SOTA Baseline | 0.8237 | 0.8346 | 0.8377 | 0.9203 |
| **RP-DNN** | SOTA Baseline | 0.7709 | 0.7915 | 0.7929 | 0.8609 |
| **Logistic Regression** | Simple Baseline | 0.5443 | 0.5629 | 0.5737 | 0.5683 |
| **XGBoost** | Simple Baseline | 0.5104 | 0.5709 | 0.5655 | 0.5148 |
| **Random Forest** | Simple Baseline | 0.4689 | 0.5985 | 0.5529 | 0.4963 |

---

## 5. The H1 Finding: Explained Honestly

**Hypothesis H1 is NOT SUPPORTED.** 

While CASCADE2VEC achieved the highest nominal point-estimate on the test set (+0.0078 Macro F1 gap over KPG-simplified), we applied strict statistical tests to verify if this improvement was genuine or simply the result of random test-set noise.

**Rigorous Testing Methodology:**
1. **Multi-Seed Variance Check:** We tested the top configurations across 5 random initialization seeds. We found that the standard deviation across seeds ($\sim 0.0049$) vastly exceeded the gap between the time-decay config and the zero-decay config. This proved that time-weighting edges ($\lambda$) provided **no statistical benefit** over treating edges uniformly.
2. **McNemar's Test:** Comparing predictions against the SOTA baseline yielded $p = 0.1928 \ge 0.05$ (No significant disagreement).
3. **Bootstrap 95% Confidence Interval (1000 resamples):** The CI of the performance gap was `[-0.0096, 0.0244]`, confidently crossing zero.

Because the CI crosses zero, we must conclude that CASCADE2VEC does not significantly outperform the SOTA baseline. This is a highly valuable, methodologically bulletproof **null result**. It demonstrates that the complex temporal dynamics (exponential time-decay) assumed necessary by much of the literature can be entirely matched by simpler, static topological convolutions. The fact that we caught and resolved the pipeline contamination bug *before* drawing this conclusion reinforces the absolute integrity of these findings.

---

## 6. What's Next

**Immediate Next Step (Phase 13-14): Adaptive Early Stopping (H2)**
The project now shifts to Hypothesis 2: testing whether we can learn a dynamic confidence threshold $\theta(t)$ to flag rumours *before* a cascade fully unfolds. This phase is entirely independent of H1. We will use the existing trained baseline models to see if early-stopping logic can beat fixed-observation windows.

**Future Considerations:**
- **Revisiting H1:** We may iterate on the CASCADE2VEC architecture (e.g., injecting raw text embeddings directly into the graph convolution, or applying attention mechanisms) to see if we can genuinely break past the SOTA ceiling.
- **Phases 15-19:** The remaining roadmap includes building Explainability pipelines (SHAP/LIME), generating massive synthetic datasets for Scalability testing, and concluding with final ablations.
