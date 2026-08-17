import argparse
import logging
import os
import sys
from cascade2vec.phase18_eval.ablations import run_all_ablations
from cascade2vec.phase18_eval.compare_baselines import build_master_comparison
from cascade2vec.phase18_eval.stats_tests import run_stats_tests

logger = logging.getLogger(__name__)
OUT_DIR = "logs/phase18_eval"

def generate_final_report():
    report_path = "docs/phase18_eval/final_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Phase 18: Final Evaluation Report\n\n")
        f.write("This report consolidates the findings of the entire misinformation cascading project.\n\n")
        
        # 1. Master Comparison
        f.write("## 1. Master Comparison\n")
        try:
            with open(os.path.join(OUT_DIR, "master_comparison.md"), "r") as cm:
                f.write(cm.read())
        except FileNotFoundError:
            f.write("*Master comparison not generated.*\n")
            
        f.write("\n\n## 2. Ablations\n")
        f.write("The following architectural ablations were performed on CASCADE2VEC to isolate the impact of its components:\n")
        try:
            with open(os.path.join(OUT_DIR, "ablation_results.json"), "r") as ar:
                ab_res = json.load(ar)
                f.write("```json\n")
                f.write(json.dumps(ab_res, indent=4))
                f.write("\n```\n")
        except (FileNotFoundError, Exception):
            f.write("*Ablations not run or failed to load.*\n")
            
        f.write("\n\n## 3. Statistical Testing\n")
        try:
            with open(os.path.join(OUT_DIR, "stats_report.md"), "r") as sr:
                f.write(sr.read())
        except FileNotFoundError:
            f.write("*Stats tests not run.*\n")
            
        f.write("\n\n## 4. Final Verdicts\n")
        f.write("- **H1 (Time-Decay > SOTA):** NOT SUPPORTED (See Phase 11-12)\n")
        f.write("- **H2 (Adaptive Early Stopping):** Evaluated in Phase 13-14.\n")
        f.write("- **H4 (Single-Machine Scalability):** Evaluated in Phase 16-17.\n")
        
    logger.info(f"Final report generated at {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    parser.add_argument("--skip-ablations", action="store_true", help="Skip running the expensive ablations")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    
    if os.path.exists(os.path.join(OUT_DIR, "master_comparison.md")) and not args.force:
        logger.error("Results already exist. Use --force to overwrite.")
        sys.exit(1)
        
    if not args.skip_ablations:
        run_all_ablations()
    
    build_master_comparison()
    run_stats_tests()
    
    import json
    generate_final_report()
    
    idx_file = "docs/all_phases_results_index.md"
    if os.path.exists(idx_file):
        with open(idx_file, "a") as f:
            f.write("\n## Phase 18: Final Evaluation\n- **Summary:** [final_report.md](file:///home/dr_shetty/misinformation-cascading/docs/phase18_eval/final_report.md)\n")
    
    logger.info("Phase 18 Pipeline Complete.")

if __name__ == "__main__":
    main()
