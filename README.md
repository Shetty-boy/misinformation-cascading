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
