import argparse
import logging
import sys
import os

from cascade2vec.phase15_xai.xai import run_all, OUT_DIR

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Phase 15: Run Explainability (XAI)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing explanations")
    parser.add_argument("--n-cascades", type=int, default=20, help="Number of cascades to explain with GNNExplainer")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    
    summary_path = os.path.join(OUT_DIR, "xai_summary.md")
    if os.path.exists(summary_path) and not args.force:
        logger.error(f"{summary_path} exists. Use --force to overwrite.")
        sys.exit(1)
        
    logger.info("Starting Phase 15 XAI Pipeline...")
    run_all(n_cascades=args.n_cascades)
    logger.info("Phase 15 XAI Pipeline Complete.")

if __name__ == "__main__":
    main()
