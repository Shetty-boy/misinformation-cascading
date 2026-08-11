# Misinformation Cascading

## Project Structure
```text
misinformation-cascading/
├── src/cascade2vec/           
│   ├── phase02_ingestion/     # Schema, ingestion pipeline, dataset adapters
│   ├── phase04_05_graph/      # Spark/GraphFrames graph construction & snapshots
│   ├── phase06_07_features/   # Feature engineering & baseline ML models
│   ├── phase08_10_sota_baselines/ # BiGCN, KPG, PGNN, RP-DNN reimplementations
│   ├── phase11_12_cascade2vec/    # Time-weighted GraphSAGE model & sweep scripts
│   │   ├── cascade2vec.py         # Core model and fast dataset loader
│   │   ├── sweep.py               # 72-config hyperparameter sweep script
│   │   ├── run_variance_check.py  # Multi-seed evaluation for stability
│   │   ├── run_c2v.py             # Final model retraining & evaluation
│   │   └── significance_test.py   # McNemar's & Bootstrap tests vs SOTA
│   └── phase13_18/            # Future/inactive phases (Adaptive stopping, XAI, etc)
├── tests/
│   ├── phase04_05_graph/          # Graph construction unit tests
│   ├── phase06_07_features/       # Leakage and feature integrity tests
│   └── phase11_12_cascade2vec/    # Sweep integrity and regression tests
├── data/
│   ├── raw/pheme/                 # PHEME rumour-detection dataset (primary)
│   └── processed/                 # Generated parquets & trained model checkpoints
├── docs/
│   ├── all_phases_results_index.md    # Master index of all phase results
│   ├── phase01_03/                # Design docs, EDA, leakage audits
│   └── phase11_12_cascade2vec/    # Final modeling results & summaries
├── logs/
│   ├── phase08_10_sota_baselines/ # SOTA comparison metrics
│   └── phase11_12_cascade2vec/    # Sweep logs, variance stats, t-SNE embeddings
├── pyproject.toml             # Package config — makes cascade2vec importable
├── requirements.txt           # Pinned Python dependencies
└── README.md
```

## Setup & Installation

**Prerequisite:** Java 17 must be installed on your system (required for Spark and GraphFrames to run).

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the `cascade2vec` package in editable mode:
   ```bash
   pip install -e .
   ```
4. Verify your installation by running the smoke test:
   ```bash
   python smoke_test.py
   ```
   Confirm you see "SMOKE TEST PASSED" before proceeding.

## Project Status & Key Findings

This repository implements a complete misinformation cascade analysis and modeling pipeline, culminating in the **CASCADE2VEC** architecture (Phases 11-12). Our primary research goal was to determine if time-decay message passing (weighting information propagation by temporal arrival) significantly outperforms state-of-the-art (SOTA) graph baseline models.

### Final Results

After extensive hyperparameter sweeping, pipeline integrity auditing, and rigorous statistical testing, we compared the optimized CASCADE2VEC against the best baseline (KPG-simplified) on the PHEME dataset:

* **CASCADE2VEC Test Macro F1:** 0.8388
* **KPG-simplified Test Macro F1:** 0.8311

While CASCADE2VEC marginally beat the SOTA baseline by a point estimate of `+0.0078`, we applied strict statistical tests to validate the significance of this improvement:

1. **McNemar's Test:** $p = 0.1928 \ge 0.05$ (No significant disagreement)
2. **Bootstrap 95% Confidence Interval (1000 resamples):** `[-0.0096, 0.0244]` (Crosses zero)

### Research Conclusion
**Hypothesis H1 is NOT SUPPORTED.** The statistical tests rigorously confirm that the custom time-decay architecture (CASCADE2VEC) does **not** significantly outperform the SOTA baseline on this dataset. Furthermore, a multi-seed variance check demonstrated that the time-decay parameter ($\lambda$) itself provided no statistically distinct advantage over uniform edge weighting.

This represents a highly valuable scientific "null result"—demonstrating that complex temporal decay dynamics often assumed to be necessary in cascade modeling can sometimes be completely matched by simpler, uniform topological graph convolutions. 

For full details on the phase-by-phase execution and test results, please see the [All Phases Results Index](docs/all_phases_results_index.md).
