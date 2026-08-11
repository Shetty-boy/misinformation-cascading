# Misinformation Cascading

## Project Structure
```text
misinformation-cascading/
├── src/cascade2vec/           # Installable Python package (pip install -e .)
│   ├── phase02_ingestion/     # Schema, ingestion pipeline, dataset adapters
│   ├── phase04_05_graph/      # Spark/GraphFrames graph construction & snapshots
│   ├── phase06_07_features/   # Feature engineering & simple baselines
│   ├── phase08_10_sota_baselines/ # BiGCN, KPG, PGNN, RP-DNN reimplementations
│   ├── phase11_12_cascade2vec/    # Time-weighted GraphSAGE embedding model
│   ├── phase13_14_adaptive_stopping/ # Adaptive θ(t) early-stopping mechanism
│   ├── phase15_xai/           # SHAP / LIME explainability
│   ├── phase16_17_scalability/ # Synthetic data generation & scalability benchmarks
│   └── phase18_eval/          # Ablations, baseline comparison, statistical tests
├── tests/
│   └── phase04_05_graph/      # Unit tests for graph construction & snapshots
├── data/
│   ├── raw/
│   │   ├── pheme/             # PHEME rumour-detection dataset (primary)
│   │   └── unused_datasets/   # Twitter15, Twitter16, FakeNewsNet (inactive)
│   ├── processed/
│   │   ├── phase02_ingestion/ # unified.parquet (102,440 rows)
│   │   ├── phase04_05_graph/  # vertices, edges, singletons, graph_stats Parquets
│   │   └── phase16_17_scalability/ # synthetic_cascades.parquet
│   └── external/
├── notebooks/
│   ├── phase03_eda_leakage/   # EDA & feature leakage audit (executed)
│   ├── phase04_05_graph/      # Snapshot validation notebook & script
│   └── phase15_xai/           # XAI notebooks (future)
├── docs/
│   ├── phase01_data_acquisition/ # design_doc.md (problem statement, hypotheses)
│   └── phase03_eda_leakage/   # validated_features.md (leakage table)
├── experiments/
│   ├── phase11_12_cascade2vec/ # train_embedding.py, embedding_sweep.yaml
│   └── phase18_eval/outputs/  # Evaluation output artefacts
├── logs/
│   ├── phase02_ingestion/     # data_audit.md
│   └── phase04_05_graph/      # graph_stats.md, Spark checkpoints
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
