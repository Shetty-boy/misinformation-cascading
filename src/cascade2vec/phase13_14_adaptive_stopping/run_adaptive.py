import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cascade2vec.phase13_14_adaptive_stopping.adaptive_threshold import fit_adaptive_threshold
from cascade2vec.phase13_14_adaptive_stopping.detection_loop import run_detection_loop, compute_detection_metrics

logger = logging.getLogger(__name__)

CONFIDENCE_FILE = "data/processed/phase13_14_adaptive_stopping/confidence_features.parquet"
OUT_DIR = "logs/phase13_14_adaptive_stopping"
RESULTS_JSON = os.path.join(OUT_DIR, "detection_results.json")
SUMMARY_MD = os.path.join(OUT_DIR, "h2_summary.md")
PLOT_FILE = os.path.join(OUT_DIR, "detection_delay_curve.png")
INDEX_FILE = "docs/all_phases_results_index.md"


def bootstrap_mdt_gap(results_df: pd.DataFrame, n_resamples=1000):
    """
    Bootstrap 95% CI on the MDT gap between adaptive and fixed_best.
    """
    gaps = []
    n = len(results_df)
    for _ in range(n_resamples):
        indices = np.random.randint(0, n, n)
        sample = results_df.iloc[indices]
        mdt_ada = sample["adaptive_time"].mean()
        mdt_fix = sample["fixed_thresh_time"].mean()
        gaps.append(mdt_fix - mdt_ada) # Positive means adaptive is faster
        
    gaps = np.array(gaps)
    ci_lower = np.percentile(gaps, 2.5)
    ci_upper = np.percentile(gaps, 97.5)
    mean_gap = np.mean(gaps)
    return mean_gap, ci_lower, ci_upper


def plot_delay_curve(results_c2v, results_kpg):
    # This is a placeholder for a more complex plot.
    # It shows the cumulative % of cascades stopped over time.
    plt.figure(figsize=(10, 6))
    
    t_windows = [1, 2, 5, 10, 15, 30, 60, 120]
    
    for df, name, color in [(results_c2v, "CASCADE2VEC", "blue"), (results_kpg, "KPG-Simplified", "green")]:
        y_ada = []
        y_fix = []
        for t in t_windows:
            y_ada.append(np.mean(df["adaptive_time"] <= t) * 100)
            y_fix.append(np.mean(df["fixed_thresh_time"] <= t) * 100)
            
        plt.plot(t_windows, y_ada, marker='o', color=color, linestyle='-', label=f"{name} (Adaptive)")
        plt.plot(t_windows, y_fix, marker='s', color=color, linestyle='--', label=f"{name} (Fixed Thresh)")

    plt.xlabel("Time Window (minutes)")
    plt.ylabel("% of Cascades Detected")
    plt.title("Detection Delay Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_FILE)
    plt.close()


def update_results_index():
    with open(INDEX_FILE, "r") as f:
        content = f.read()
        
    if "Phase 13-14: Adaptive Early Stopping (H2)" not in content:
        section = """
## Phase 13-14: Adaptive Early Stopping (H2)
- **Summary:** [h2_summary.md](file:///home/dr_shetty/misinformation-cascading/logs/phase13_14_adaptive_stopping/h2_summary.md)
- **Detection Delay Curve:** [detection_delay_curve.png](file:///home/dr_shetty/misinformation-cascading/logs/phase13_14_adaptive_stopping/detection_delay_curve.png)
- **Raw Results:** [detection_results.json](file:///home/dr_shetty/misinformation-cascading/logs/phase13_14_adaptive_stopping/detection_results.json)
"""
        with open(INDEX_FILE, "a") as f:
            f.write(section)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    parser.add_argument("--limit", type=int, default=None, help="Limit cascades for quick testing")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    
    if os.path.exists(RESULTS_JSON) and not args.force:
        logger.error(f"{RESULTS_JSON} exists. Use --force to overwrite.")
        sys.exit(1)
        
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(CONFIDENCE_FILE):
        logger.error(f"{CONFIDENCE_FILE} not found. Run confidence_extractor.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(CONFIDENCE_FILE)
    if args.limit:
        cids = df["cascade_id"].unique()[:args.limit]
        df = df[df["cascade_id"].isin(cids)]
        
    split_df = df[["cascade_id", "split"]].drop_duplicates()
    
    results = {}
    test_results_df = {}
    
    for model_type in ["c2v", "kpg"]:
        logger.info(f"Processing {model_type}...")
        
        # 1. Fit adaptive threshold
        thresh_res = fit_adaptive_threshold(df, split_df, model_type=model_type)
        model = thresh_res["model"]
        best_fixed = thresh_res["best_fixed_threshold"]
        features_cols = thresh_res["features"]
        
        # 2. Run detection loop on test set
        res_df = run_detection_loop(df, "test", model, features_cols, best_fixed, model_type=model_type)
        test_results_df[model_type] = res_df
        
        # 3. Compute metrics
        metrics = compute_detection_metrics(res_df)
        results[model_type] = metrics
        
    # Write JSON results
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Plot
    plot_delay_curve(test_results_df["c2v"], test_results_df["kpg"])
    
    # Analyze C2V for H2 Verdict
    c2v_metrics = results["c2v"]
    ada_mdt = c2v_metrics["adaptive"]["mdt"]
    fix_mdt = c2v_metrics["fixed_thresh"]["mdt"]
    
    ada_f1 = c2v_metrics["adaptive"]["macro_f1"]
    fix_f1 = c2v_metrics["fixed_thresh"]["macro_f1"]
    
    mean_gap, ci_lower, ci_upper = bootstrap_mdt_gap(test_results_df["c2v"])
    
    h2_supported = (ada_mdt < fix_mdt * 0.90) and (ada_f1 >= fix_f1 - 0.01)
    
    summary = f"""# Phase 13-14: Adaptive Early Stopping (H2)

## H2 Verdict: {'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}

**Criteria for Support:** 
- Mean Detection Time (MDT) reduced by ≥10% compared to best fixed threshold.
- Macro F1 at detection time must match or exceed fixed threshold (within -0.01 margin).

### Key Results (CASCADE2VEC - Headline)
- **Adaptive MDT:** {ada_mdt:.2f} mins (Median: {c2v_metrics['adaptive']['median_mdt']:.2f} mins)
- **Fixed Best MDT:** {fix_mdt:.2f} mins (Median: {c2v_metrics['fixed_thresh']['median_mdt']:.2f} mins)
- **Fixed t=120 MDT:** 120.00 mins

- **Adaptive Macro F1:** {ada_f1:.4f}
- **Fixed Best Macro F1:** {fix_f1:.4f}
- **Fixed t=120 Macro F1:** {c2v_metrics['fixed_120']['macro_f1']:.4f}

- **% Cascades Stopped Early (Adaptive):** {c2v_metrics['adaptive']['early_stop_pct']:.1f}%

### Statistical Testing (Bootstrap 95% CI on MDT Gap)
- **Mean Gap (Fixed - Adaptive):** {mean_gap:.2f} mins
- **95% CI:** [{ci_lower:.2f}, {ci_upper:.2f}]
- {'Significant time reduction observed (CI > 0).' if ci_lower > 0 else 'No significant time reduction (CI crosses zero).'}

### Secondary Check (KPG-Simplified)
- **Adaptive MDT:** {results['kpg']['adaptive']['mdt']:.2f} mins
- **Fixed Best MDT:** {results['kpg']['fixed_thresh']['mdt']:.2f} mins
- **Adaptive Macro F1:** {results['kpg']['adaptive']['macro_f1']:.4f}
"""

    with open(SUMMARY_MD, "w") as f:
        f.write(summary)
        
    update_results_index()
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
