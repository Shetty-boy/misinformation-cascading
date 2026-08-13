"""
baselines_simple.py — Phase 7 Baseline Classifiers
====================================================
Trains and evaluates three baseline classifiers for the Rumour Detection task:
  - Logistic Regression
  - Random Forest
  - XGBoost

Task:
    Binary classification — rumour (1) vs non-rumour (0)
    Labels from 'label_binary' column in feature_matrix.parquet.
    See docs/phase06_07_features/phase06_07_classification_protocol.md.

Cross-validation:
    StratifiedGroupKFold(n_splits=5), groups=cascade_id
    # Prevents cascade-level leakage across train/test folds:
    # All snapshots from the same cascade always stay in the same fold,
    # so the model can never see future snapshots of a cascade it was tested on.

Class imbalance:
    Both weighted and unweighted runs are reported for each model.
    ~1.94:1 non-rumour:rumour imbalance warrants explicit logging of both.
    No mandatory F1 threshold — report exactly what the data produces.

Metrics:
    Primary:   Macro F1
    Secondary: Weighted F1, Accuracy, Precision, Recall, ROC-AUC
"""

import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import (
    make_scorer, f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Paths
FEATURE_MATRIX_PATH = "data/processed/phase06_07_features/feature_matrix.parquet"
RESULTS_PATH = "logs/phase06_07_features/phase06_07_baseline_results.md"
OUT_DIR = "logs/phase06_07_features"

# Feature columns to use (exclude non-feature columns)
EXCLUDE_COLS = {"cascade_id", "t_minutes", "label", "label_binary"}

# Cross-validation
N_SPLITS = 5

# Scoring
SCORERS = {
    "accuracy": make_scorer(accuracy_score),
    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    "weighted_f1": make_scorer(f1_score, average="weighted", zero_division=0),
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def _run_model(
    model,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    use_class_weights: bool,
) -> dict:
    """
    Run cross-validated evaluation for a single model.

    Parameters
    ----------
    use_class_weights : bool
        If True, enable class_weight='balanced' (or sample_weight equivalent).
        Both runs are logged — see module docstring.
    """
    cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    # Suppress convergence warnings for LR with small max_iter
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        cv_results = cross_validate(
            model,
            X, y,
            groups=groups,
            cv=cv,
            scoring=SCORERS,
            return_train_score=False,
            n_jobs=-1,
        )

    result = {
        "model": model_name,
        "weighted": use_class_weights,
        "accuracy": float(np.mean(cv_results["test_accuracy"])),
        "macro_f1": float(np.mean(cv_results["test_macro_f1"])),
        "weighted_f1": float(np.mean(cv_results["test_weighted_f1"])),
        "precision_macro": float(np.mean(cv_results["test_precision_macro"])),
        "recall_macro": float(np.mean(cv_results["test_recall_macro"])),
        "roc_auc": float(np.mean(cv_results["test_roc_auc"])),
    }
    return result


def _build_models(use_class_weights: bool) -> list[tuple]:
    """Build (name, pipeline) pairs for each baseline model."""
    cw = "balanced" if use_class_weights else None
    models = [
        (
            "Logistic Regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight=cw,
                    max_iter=1000,
                    solver="lbfgs",
                    multi_class="auto",
                    random_state=42,
                )),
            ]),
        ),
        (
            "Random Forest",
            Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=200,
                    class_weight=cw,
                    n_jobs=-1,
                    random_state=42,
                )),
            ]),
        ),
        (
            "XGBoost",
            Pipeline([
                ("clf", _make_xgb(use_class_weights)),
            ]),
        ),
    ]
    return models


def _make_xgb(use_class_weights: bool):
    """Build XGBoost classifier, handling optional xgboost import gracefully."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=1.94 if use_class_weights else 1.0,  # non-rumour:rumour ratio
            n_jobs=-1,
            random_state=42,
        )
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBoost baseline")
        return None


def run_baselines(
    feature_matrix_path: str = FEATURE_MATRIX_PATH,
    filter_disconnected: bool = False,
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run all baseline classifiers and return results DataFrame.

    Parameters
    ----------
    feature_matrix_path : str
        Path to the feature_matrix.parquet output of build_feature_matrix.py.
    filter_disconnected : bool
        If True, exclude snapshots from known-disconnected cascades before
        evaluation. This produces the "disconnected-excluded" metrics for
        comparison with the "full-dataset" metrics.

    Returns
    -------
    pd.DataFrame
        One row per (model, weighted/unweighted) pair with metric columns.
    """
    logger.info("[baselines] Loading feature matrix from %s", feature_matrix_path)
    df = pd.read_parquet(feature_matrix_path)
    logger.info("[baselines] Loaded %d rows", len(df))

    if filter_disconnected:
        stats_path = "data/processed/phase04_05_graph/graph_stats.parquet"
        if os.path.exists(stats_path):
            stats = pd.read_parquet(stats_path)
            disc_ids = set(stats[~stats["is_connected"]]["cascade_id"].tolist())
            before = len(df)
            df = df[~df["cascade_id"].isin(disc_ids)]
            logger.info(
                "[baselines] Excluded %d rows from %d disconnected cascades (from %d total)",
                before - len(df), len(disc_ids), before
            )

    if selected_features is not None:
        feature_cols = selected_features
    else:
        feature_cols = _get_feature_cols(df)
    logger.info("[baselines] Using %d features: %s", len(feature_cols), feature_cols)

    # Drop rows with any NaN in feature columns
    before = len(df)
    df = df.dropna(subset=feature_cols)
    if len(df) < before:
        logger.warning("[baselines] Dropped %d rows with NaN features", before - len(df))

    X = df[feature_cols].values.astype(float)
    y = df["label_binary"].values.astype(int)
    groups = df["cascade_id"].values

    logger.info(
        "[baselines] Label distribution: rumour=%d, non-rumour=%d",
        (y == 1).sum(), (y == 0).sum()
    )

    all_results = []
    for use_weights in [False, True]:
        weight_label = "weighted" if use_weights else "unweighted"
        logger.info("[baselines] Running %s runs...", weight_label)
        for name, pipeline in _build_models(use_weights):
            if pipeline is None:
                continue
            # Check if XGBoost's underlying estimator is None (import failed)
            from sklearn.pipeline import Pipeline as SKPipeline
            estimators = pipeline.steps if isinstance(pipeline, SKPipeline) else []
            if any(est is None for _, est in estimators):
                continue

            logger.info("  [baselines] %s (%s)...", name, weight_label)
            t0 = time.time()
            result = _run_model(pipeline, name, X, y, groups, use_weights)
            t1 = time.time()
            result["runtime_s"] = round(t1 - t0, 1)
            result["filter_disconnected"] = filter_disconnected
            all_results.append(result)
            logger.info("    Macro F1=%.4f | ROC-AUC=%.4f | Runtime=%.1fs",
                        result["macro_f1"], result["roc_auc"], result["runtime_s"])

    return pd.DataFrame(all_results)


def _write_results_md(full_results: pd.DataFrame, excl_results: pd.DataFrame) -> None:
    """Write the baseline results markdown report."""
    os.makedirs(OUT_DIR, exist_ok=True)

    def fmt(df: pd.DataFrame, title: str) -> str:
        lines = [f"### {title}\n"]
        table_cols = ["model", "weighted", "accuracy", "macro_f1", "weighted_f1", "roc_auc"]
        df_fmt = df[table_cols].copy()
        df_fmt.columns = ["Model", "Class Weights", "Accuracy", "Macro F1", "Weighted F1", "ROC-AUC"]
        df_fmt["Class Weights"] = df_fmt["Class Weights"].map({True: "Yes", False: "No"})
        for col in ["Accuracy", "Macro F1", "Weighted F1", "ROC-AUC"]:
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.4f}")
        lines.append(df_fmt.to_markdown(index=False))
        lines.append("")
        return "\n".join(lines)

    report = f"""# Phase 7: Baseline Results

Task: **Rumour Detection** (rumour vs non-rumour)
Cross-validation: StratifiedGroupKFold(n_splits=5), groups=cascade_id
Primary metric: Macro F1

> Class imbalance: ~1.94:1 non-rumour:rumour. Results reported BOTH with
> and without class weights so the effect of imbalance handling is visible.

---

{fmt(full_results, "Full Dataset")}

{fmt(excl_results, "Disconnected Cascades Excluded")}

---

## Notes
- All snapshots of a cascade always go to the same fold (group=cascade_id)
  to prevent cascade-level leakage.
- No F1 threshold is enforced — values are reported exactly as produced.
- XGBoost scale_pos_weight=1.94 used for the weighted run (non-rumour:rumour ratio).
- Pruning of features is deferred to Phase 18 ablations.
"""
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    logger.info("[baselines] Results written to %s", RESULTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    t0 = time.time()

    logger.info("=== PHASE 7 BASELINES ===")
    logger.info("Run 1: Full dataset")
    full_results = run_baselines(filter_disconnected=False)

    logger.info("Run 2: Disconnected cascades excluded")
    excl_results = run_baselines(filter_disconnected=True)

    _write_results_md(full_results, excl_results)

    t1 = time.time()
    print(f"\n=== BASELINE RUNS COMPLETE === ({t1-t0:.1f}s total)")
    print("\n--- Full Dataset ---")
    print(full_results[["model","weighted","macro_f1","roc_auc"]].to_string(index=False))
    print("\n--- Disconnected Excluded ---")
    print(excl_results[["model","weighted","macro_f1","roc_auc"]].to_string(index=False))
    print(f"\nResults: {RESULTS_PATH}")
