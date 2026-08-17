"""
run_c2v.py — Phase 11-12: CASCADE2VEC Final Training & Evaluation
=================================================================
Loads best config from sweep, trains final model on train split,
evaluates on test split EXACTLY ONCE.

Outputs:
  - data/processed/phase11_12_cascade2vec/checkpoints/final_model.pt
  - logs/phase11_12_cascade2vec/c2v_results.json
  - logs/phase11_12_cascade2vec/embedding_visualization.png (t-SNE)
  - logs/phase08_10_sota_baselines/sota_comparison.md  (updated with C2V row)

Overwrite safety:
  - --force required to overwrite existing final checkpoint
  - Sweep log must already exist (run sweep.py first)

Usage:
    PYTHONPATH=src python src/cascade2vec/phase11_12_cascade2vec/run_c2v.py
    PYTHONPATH=src python src/cascade2vec/phase11_12_cascade2vec/run_c2v.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.loader import DataLoader

from cascade2vec.phase11_12_cascade2vec.cascade2vec import (
    CASCADE2VEC, C2VClassifier, SnapshotDataset,
    DEVICE, SEED, train_c2v, evaluate_c2v,
)
from cascade2vec.phase11_12_cascade2vec.sweep import (
    _build_tfidf, SWEEP_FIXED, SWEEP_LOG,
)

logger = logging.getLogger(__name__)

UNIFIED_FILE   = "data/processed/phase02_ingestion/unified.parquet"
SPLIT_FILE     = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"
CKPT_DIR       = "data/processed/phase11_12_cascade2vec/checkpoints"
FINAL_CKPT     = os.path.join(CKPT_DIR, "final_model.pt")
LOG_DIR        = "logs/phase11_12_cascade2vec"
RESULTS_JSON   = os.path.join(LOG_DIR, "c2v_results.json")
VIZ_PATH       = os.path.join(LOG_DIR, "embedding_visualization.png")
SOTA_TABLE     = "logs/phase08_10_sota_baselines/sota_comparison.md"

# Full training epochs (sweep used 30 for speed)
FINAL_N_EPOCHS = 50
FINAL_PATIENCE = 10


def _load_best_config() -> dict:
    """Parse best config from sweep log markdown."""
    if not os.path.exists(SWEEP_LOG):
        raise FileNotFoundError(
            f"Sweep log not found: {SWEEP_LOG}\n"
            "Run sweep.py first: PYTHONPATH=src python "
            "src/cascade2vec/phase11_12_cascade2vec/sweep.py"
        )
    with open(SWEEP_LOG) as f:
        content = f.read()
    # Extract JSON block between first ```json and ``` markers
    start = content.index("```json\n") + len("```json\n")
    end   = content.index("\n```", start)
    cfg = json.loads(content[start:end])
    logger.info("[run_c2v] Loaded best config: %s", cfg)
    return cfg


def generate_tsne(embeddings: np.ndarray, labels: np.ndarray, path: str):
    """Generate t-SNE visualization of cascade embeddings."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        logger.info("[run_c2v] Running t-SNE on %d embeddings...", len(embeddings))
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(embeddings) // 3))
        proj = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = {0: "#4C72B0", 1: "#DD8452"}
        label_names = {0: "non-rumour", 1: "rumour"}

        for label_val in [0, 1]:
            mask = labels == label_val
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=colors[label_val],
                label=label_names[label_val],
                alpha=0.6, s=15, linewidths=0,
            )

        ax.set_title("CASCADE2VEC Embedding Space (t-SNE)\ncolored by rumour/non-rumour label",
                     fontsize=13)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("[run_c2v] t-SNE saved to %s", path)
    except Exception as e:
        logger.warning("[run_c2v] t-SNE failed: %s", e)


def update_sota_table(test_metrics: dict, runtime_min: float):
    """Append CASCADE2VEC row to the existing SOTA comparison table."""
    if not os.path.exists(SOTA_TABLE):
        logger.warning("[run_c2v] sota_comparison.md not found, skipping update.")
        return

    with open(SOTA_TABLE) as f:
        content = f.read()

    # Check if C2V row already present
    if "CASCADE2VEC" in content:
        logger.info("[run_c2v] CASCADE2VEC row already in sota_comparison.md — skipping.")
        return

    new_row = (
        f"| CASCADE2VEC (Phase 11-12) | CASCADE2VEC | "
        f"{round(test_metrics['accuracy'], 4)} | "
        f"**{round(test_metrics['macro_f1'], 4)}** | "
        f"{round(test_metrics['weighted_f1'], 4)} | "
        f"{round(test_metrics['roc_auc'], 4)} | "
        f"{round(runtime_min, 2)} |"
    )

    # Insert CASCADE2VEC row right after the header separator line
    # Find the last row of the table and insert before Notes section
    note_idx = content.find("\n## Notes")
    if note_idx == -1:
        content = content + "\n" + new_row + "\n"
    else:
        content = content[:note_idx] + "\n" + new_row + content[note_idx:]

    # Update the H1 claim in Notes
    h1_note = (
        "\n- **H1 (CASCADE2VEC vs. SOTA):** See CASCADE2VEC row above. "
        "Training regime: Option B (all 8 snapshot windows). "
        "H1 comparison uses t=120min evaluation, matching full-cascade SOTA setup."
    )
    content = content + h1_note if "H1" not in content else content

    with open(SOTA_TABLE, "w") as f:
        f.write(content)
    logger.info("[run_c2v] Updated sota_comparison.md with CASCADE2VEC row.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing final checkpoint and results")
    args = parser.parse_args()

    if os.path.exists(FINAL_CKPT) and not args.force:
        raise RuntimeError(
            f"Final checkpoint {FINAL_CKPT} already exists. "
            "Use --force to retrain."
        )

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1. Load data
    logger.info("[run_c2v] Loading data...")
    unified  = pd.read_parquet(UNIFIED_FILE)
    split_df = pd.read_parquet(SPLIT_FILE)

    # 2. Load best config from sweep
    best_cfg = _load_best_config()
    logger.info("[run_c2v] Best config: %s", best_cfg)

    # 3. Build TF-IDF (train only)
    logger.info("[run_c2v] Fitting TF-IDF on train split...")
    tfidf = _build_tfidf(unified, split_df)

    # 4. Build datasets & loaders
    batch_size = SWEEP_FIXED["batch_size"]
    train_ds = SnapshotDataset(unified, split_df, "train", tfidf, best_cfg["lam"])
    val_ds   = SnapshotDataset(unified, split_df, "val",   tfidf, best_cfg["lam"])
    test_ds  = SnapshotDataset(unified, split_df, "test",  tfidf, best_cfg["lam"])

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  generator=g, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. Build model
    in_dim = 5000
    encoder = CASCADE2VEC(
        in_dim=in_dim,
        hidden_dim=SWEEP_FIXED["hidden_dim"],
        embed_dim=best_cfg["embed_dim"],
        n_layers=best_cfg["n_layers"],
        dropout=SWEEP_FIXED["dropout"],
    )
    classifier = C2VClassifier(embed_dim=best_cfg["embed_dim"], num_classes=2)

    # Class weights
    train_sub = split_df[split_df["split"] == "train"]
    counts = train_sub["label"].value_counts()
    total = len(train_sub)
    cw = torch.tensor(
        [total / (2 * counts.get("non-rumour", 1)),
         total / (2 * counts.get("rumour", 1))],
        dtype=torch.float32, device=DEVICE,
    )

    # 6. Train
    t_train_start = time.time()
    logger.info("[run_c2v] Training final model (50 epochs, patience=10)...")
    train_result = train_c2v(
        encoder, classifier, train_loader, val_loader,
        n_epochs=FINAL_N_EPOCHS,
        lr=SWEEP_FIXED["lr"],
        weight_decay=SWEEP_FIXED["weight_decay"],
        alpha=best_cfg["alpha"],
        temperature=SWEEP_FIXED["temperature"],
        patience=FINAL_PATIENCE,
        class_weights=cw,
        device=DEVICE,
        checkpoint_path=FINAL_CKPT,
    )
    runtime_min = (time.time() - t_train_start) / 60.0

    # 7. Load best checkpoint
    ckpt = torch.load(FINAL_CKPT, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    classifier.load_state_dict(ckpt["classifier"])
    logger.info("[run_c2v] Loaded best checkpoint from epoch %d", ckpt["epoch"])

    # 8. Evaluate on TEST SPLIT (exactly once)
    logger.info("[run_c2v] *** Evaluating on TEST split (single use) ***")
    test_metrics = evaluate_c2v(
        encoder, classifier, test_loader,
        device=DEVICE, return_embeddings=True,
    )
    embeddings = test_metrics.pop("embeddings", None)
    labels     = test_metrics.pop("labels", None)
    probs      = test_metrics.pop("probs", None)

    logger.info("[run_c2v] TEST RESULTS:")
    for k, v in test_metrics.items():
        logger.info("  %s = %.4f", k, v)

    # 9. Save results JSON
    results = {
        "best_config": best_cfg,
        "best_val_epoch": train_result["best_epoch"],
        "best_val_macro_f1": train_result["best_val_macro_f1"],
        "test_metrics": test_metrics,
        "runtime_minutes": round(runtime_min, 2),
        "training_history": train_result["history"],
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("[run_c2v] Results saved to %s", RESULTS_JSON)

    # 10. t-SNE visualization
    generate_tsne(embeddings, labels, VIZ_PATH)

    # 11. Update SOTA comparison table
    update_sota_table(test_metrics, runtime_min)

    # 12. H1 verdict
    sota_best = 0.8311  # KPG-simplified from sota_comparison.md
    macro_f1  = test_metrics["macro_f1"]
    h1_supported = macro_f1 > sota_best
    verdict = "SUPPORTED" if h1_supported else "NOT SUPPORTED"
    logger.info(
        "[run_c2v] H1 (CASCADE2VEC > SOTA best KPG-simplified): %s "
        "(CASCADE2VEC=%.4f vs KPG=%.4f)",
        verdict, macro_f1, sota_best,
    )
    print(f"\n{'='*60}")
    print(f"CASCADE2VEC Test Macro F1: {macro_f1:.4f}")
    print(f"SOTA Best (KPG-simplified): {sota_best:.4f}")
    print(f"H1 Hypothesis: {verdict}")
    print(f"Runtime: {runtime_min:.2f} min")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
