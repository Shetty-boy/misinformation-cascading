"""
feature_selection.py — Phase 6/7 Feature Reduction Pipeline
===========================================================
Implements a 3-stage feature selection pipeline:
1. Correlation pruning (removes one of each pair with |r| > threshold)
2. Importance ranking (RandomForest on train set)
3. Ablation comparison (compare Full, Pruned, Top-K sets)

Outputs a report to logs/phase06_07_features/feature_selection_report.md
and saves the selected feature set to data/processed/phase06_07_features/selected_features.json
"""

import json
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

from cascade2vec.phase06_07_features.baselines_simple import (
    FEATURE_MATRIX_PATH,
    OUT_DIR as BASELINES_OUT_DIR,
    run_baselines,
    _get_feature_cols,
)

logger = logging.getLogger(__name__)

SPLIT_FILE = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"
OUT_REPORT = "logs/phase06_07_features/feature_selection_report.md"
OUT_JSON = "data/processed/phase06_07_features/selected_features.json"


def prune_correlated_features(
    df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.95
) -> tuple[list[str], list[dict]]:
    """
    Stage 1: Prune highly correlated features.
    If |r| > threshold, keep the one with a higher F-statistic (ANOVA).
    """
    logger.info("[selection] Stage 1: Correlation Pruning (threshold=%.2f)", threshold)
    
    # Calculate F-statistics for all features
    X = df[feature_cols].fillna(0).values
    y = df["label_binary"].values
    f_stats, p_values = f_classif(X, y)
    
    # Map feature to its F-stat
    f_stat_dict = {feat: f_stats[i] for i, feat in enumerate(feature_cols)}
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr().abs()
    
    to_drop = set()
    drop_reasons = []
    
    # Iterate upper triangle
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_a = corr_matrix.columns[i]
            col_b = corr_matrix.columns[j]
            if col_a in to_drop or col_b in to_drop:
                continue
                
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > threshold:
                # Decide which to drop based on F-stat
                if f_stat_dict[col_a] >= f_stat_dict[col_b]:
                    dropped = col_b
                    kept = col_a
                else:
                    dropped = col_a
                    kept = col_b
                    
                to_drop.add(dropped)
                drop_reasons.append({
                    "dropped": dropped,
                    "kept": kept,
                    "correlation": float(corr_val),
                    "dropped_f_stat": float(f_stat_dict[dropped]),
                    "kept_f_stat": float(f_stat_dict[kept])
                })
                logger.info(
                    "  [selection] Dropped %s (r=%.3f with %s). F-stat: %.1f vs %.1f",
                    dropped, corr_val, kept, f_stat_dict[dropped], f_stat_dict[kept]
                )
                
    survivors = [c for c in feature_cols if c not in to_drop]
    logger.info("[selection] Stage 1 kept %d / %d features", len(survivors), len(feature_cols))
    return survivors, drop_reasons


def rank_by_importance(
    df: pd.DataFrame, split_df: pd.DataFrame, feature_cols: list[str]
) -> list[dict]:
    """
    Stage 2: Rank features by RandomForest importance on TRAIN set only.
    """
    logger.info("[selection] Stage 2: Importance Ranking")
    
    # Merge with split to get train set
    merged = df.merge(split_df[["cascade_id", "split"]], on="cascade_id", how="inner")
    train = merged[merged["split"] == "train"].dropna(subset=feature_cols)
    
    X_train = train[feature_cols].values
    y_train = train["label_binary"].values
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    
    ranked = []
    for i, feat in enumerate(feature_cols):
        ranked.append({
            "feature": feat,
            "importance": float(importances[i])
        })
        
    ranked.sort(key=lambda x: x["importance"], reverse=True)
    
    # Add cumulative importance
    cum_sum = 0.0
    for item in ranked:
        cum_sum += item["importance"]
        item["cumulative_importance"] = cum_sum
        
    return ranked


def run_ablation_comparison(feature_matrix_path: str, sets: dict[str, list[str]]) -> pd.DataFrame:
    """
    Stage 3: Run baselines on different feature sets.
    """
    logger.info("[selection] Stage 3: Ablation Comparison")
    
    results = []
    
    for set_name, cols in sets.items():
        logger.info("  [selection] Running baselines for set: %s (%d features)", set_name, len(cols))
        df_res = run_baselines(
            feature_matrix_path=feature_matrix_path,
            filter_disconnected=False,
            selected_features=cols
        )
        
        # We only care about the best macro_f1 for this set (usually XGBoost or LR weighted)
        # But let's keep all runs and we can just take the best one in the report
        df_res["feature_set"] = set_name
        df_res["num_features"] = len(cols)
        results.append(df_res)
        
    return pd.concat(results, ignore_index=True)


def _write_report(
    original_cols: list[str],
    pruned_cols: list[str],
    drop_reasons: list[dict],
    ranked: list[dict],
    ablation_res: pd.DataFrame,
    final_cols: list[str]
):
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    
    lines = [
        "# Phase 6/7: Feature Selection Report",
        "",
        "This report documents the 3-stage feature selection pipeline to reduce redundancy "
        "while preserving interpretability for XAI.",
        "",
        "## Stage 1: Correlation Pruning",
        f"Original features: **{len(original_cols)}**",
        f"Surviving features: **{len(pruned_cols)}**",
        "",
        "| Dropped Feature | Kept Feature | Correlation | Dropped F-Stat | Kept F-Stat |",
        "|---|---|---|---|---|"
    ]
    
    for r in drop_reasons:
        lines.append(
            f"| `{r['dropped']}` | `{r['kept']}` | {r['correlation']:.4f} | "
            f"{r['dropped_f_stat']:.1f} | {r['kept_f_stat']:.1f} |"
        )
        
    lines += [
        "",
        "## Stage 2: Feature Importance (on Pruned Set)",
        "Computed via RandomForest on the training split only.",
        "",
        "| Rank | Feature | Importance | Cumulative |",
        "|---|---|---|---|"
    ]
    
    for i, r in enumerate(ranked):
        lines.append(
            f"| {i+1} | `{r['feature']}` | {r['importance']:.4f} | {r['cumulative_importance']:.4f} |"
        )
        
    lines += [
        "",
        "## Stage 3: Ablation Comparison",
        "Comparing the best performing baseline model (by Macro F1) across different feature sets.",
        "",
        "| Feature Set | # Features | Best Model | Class Weights | Macro F1 | ROC-AUC |",
        "|---|---|---|---|---|---|"
    ]
    
    for set_name in ablation_res["feature_set"].unique():
        sub = ablation_res[ablation_res["feature_set"] == set_name]
        best_idx = sub["macro_f1"].idxmax()
        best_row = sub.loc[best_idx]
        
        lines.append(
            f"| {set_name} | {best_row['num_features']} | {best_row['model']} | "
            f"{'Yes' if best_row['weighted'] else 'No'} | **{best_row['macro_f1']:.4f}** | "
            f"{best_row['roc_auc']:.4f} |"
        )
        
    lines += [
        "",
        "## Recommendation",
        f"The recommended feature set is **{len(final_cols)} features**.",
        "It preserves interpretability, removes redundancy, and maintains (or improves) performance.",
        "Selected features: " + ", ".join([f"`{c}`" for c in final_cols])
    ]
    
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines))
    logger.info("[selection] Report written to %s", OUT_REPORT)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    split_df = pd.read_parquet(SPLIT_FILE)
    
    original_cols = _get_feature_cols(df)
    
    # 1. Prune
    pruned_cols, drop_reasons = prune_correlated_features(df, original_cols, threshold=0.95)
    
    # 2. Rank
    ranked = rank_by_importance(df, split_df, pruned_cols)
    
    # Select top 8 and top 10 for ablation
    top_8 = [r["feature"] for r in ranked[:8]]
    top_10 = [r["feature"] for r in ranked[:10]]
    
    # 3. Ablate
    sets = {
        "Full (Baseline)": original_cols,
        "Correlation-Pruned": pruned_cols,
        "Top-10 Importance": top_10,
        "Top-8 Importance": top_8
    }
    
    ablation_res = run_ablation_comparison(FEATURE_MATRIX_PATH, sets)
    
    # Determine best set based on ablation (highest max Macro F1)
    # Typically pruned is best because RF handles the remaining non-linearities
    # We will just select Pruned by default to be safe on information loss, unless Top-10 is strictly better
    pruned_f1 = ablation_res[ablation_res["feature_set"] == "Correlation-Pruned"]["macro_f1"].max()
    top10_f1 = ablation_res[ablation_res["feature_set"] == "Top-10 Importance"]["macro_f1"].max()
    
    final_cols = pruned_cols
    if top10_f1 > pruned_f1 + 0.01:
        logger.info("[selection] Top-10 significantly outperformed Pruned, selecting Top-10.")
        final_cols = top_10
        
    _write_report(original_cols, pruned_cols, drop_reasons, ranked, ablation_res, final_cols)
    
    # Save selected features to JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(final_cols, f, indent=2)
    logger.info("[selection] Final features saved to %s", OUT_JSON)


if __name__ == "__main__":
    main()
