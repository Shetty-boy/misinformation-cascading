import argparse
import logging
import os
from cascade2vec.phase16_17_scalability.scalability import run_volume_sweep, run_core_sweep, generate_summary, OUT_DIR

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    
    if os.path.exists(os.path.join(OUT_DIR, "scalability_summary.md")) and not args.force:
        logger.error("Results already exist. Use --force to overwrite.")
        return
        
    os.makedirs(OUT_DIR, exist_ok=True)
    
    run_volume_sweep()
    run_core_sweep()
    generate_summary()
    
    logger.info("Scalability benchmarks complete.")

if __name__ == "__main__":
    main()
