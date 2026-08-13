"""
visualize_cascades.py — Cascade Visualization Dashboard
========================================================
Generates two types of visualizations from the test/validation set:

1. Cascade propagation tree graphs — shows how tweets spread as reply trees
2. t-SNE embedding plots — shows model separation of rumour vs non-rumour

Outputs to: logs/visualizations/

Usage:
    PYTHONPATH=src python src/cascade2vec/phase06_07_features/visualize_cascades.py
"""

import json
import logging
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

UNIFIED_FILE = "data/processed/phase02_ingestion/unified.parquet"
FEATURE_MATRIX = "data/processed/phase06_07_features/feature_matrix.parquet"
SPLIT_FILE = "data/processed/phase08_10_sota_baselines/train_val_test_split.parquet"
OUT_DIR = "logs/visualizations"
SEED = 42

# --- Colour palette (premium dark theme) ---
COLORS = {
    "rumour":     "#FF6B6B",    # coral red
    "non-rumour": "#4ECDC4",    # teal green
    "bg":         "#1A1A2E",    # deep navy
    "grid":       "#2A2A4A",    # subtle grid
    "text":       "#E8E8F0",    # light text
    "accent":     "#FFD93D",    # gold accent
    "edge":       "#555577",    # muted edges
}


def _setup_dark_style():
    """Apply a premium dark style to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "font.family": "sans-serif",
        "font.size": 11,
    })


# ===========================================================================
# 1.  Cascade Propagation Tree Visualization
# ===========================================================================

def _build_cascade_tree(unified: pd.DataFrame, cascade_id: str) -> nx.DiGraph:
    """Build a NetworkX directed graph for a single cascade."""
    cascade_df = unified[unified["cascade_id"] == cascade_id].copy()
    G = nx.DiGraph()

    for _, row in cascade_df.iterrows():
        tid = row["tweet_id"]
        ts = row["timestamp"]
        is_root = pd.isna(row["parent_id"])
        G.add_node(tid, timestamp=ts, is_root=is_root)

        if not is_root and row["parent_id"] in G.nodes:
            G.add_edge(row["parent_id"], tid)

    return G


def _plot_single_cascade(G: nx.DiGraph, cascade_id: str, label: str, ax: plt.Axes):
    """Plot a single cascade tree on the given axes."""
    if len(G) == 0:
        ax.text(0.5, 0.5, "Empty cascade", ha="center", va="center",
                fontsize=10, color=COLORS["text"], transform=ax.transAxes)
        ax.set_title(f"{cascade_id[:20]}...\n({label})", fontsize=9, color=COLORS["text"])
        return

    # Use hierarchical layout
    try:
        roots = [n for n, d in G.nodes(data=True) if d.get("is_root", False)]
        root = roots[0] if roots else list(G.nodes)[0]

        # BFS layers for hierarchy
        pos = nx.bfs_layout(G, root, align="horizontal")
    except Exception:
        pos = nx.spring_layout(G, seed=SEED, k=2.0 / max(1, len(G) ** 0.5))

    color = COLORS["rumour"] if label == "rumour" else COLORS["non-rumour"]

    # Node sizes based on whether root or not
    node_sizes = []
    node_colors = []
    for n in G.nodes:
        if G.nodes[n].get("is_root", False):
            node_sizes.append(120)
            node_colors.append(COLORS["accent"])
        else:
            node_sizes.append(30)
            node_colors.append(color)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=COLORS["edge"],
                           arrows=True, arrowsize=6, width=0.8, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, linewidths=0)

    short_id = cascade_id[:15] + "..." if len(cascade_id) > 15 else cascade_id
    ax.set_title(f"{short_id}\n{label} | {len(G)} nodes",
                 fontsize=8, color=color, fontweight="bold")
    ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.1)
    ax.set_ylim(ax.get_ylim()[0] - 0.1, ax.get_ylim()[1] + 0.1)
    ax.axis("off")


def visualize_cascade_trees(
    unified: pd.DataFrame,
    split_df: pd.DataFrame,
    split_name: str = "test",
    n_samples: int = 12,
    out_path: str | None = None,
):
    """
    Visualize sample cascade propagation trees from the given split.
    Shows a grid of n_samples trees (half rumour, half non-rumour).
    """
    _setup_dark_style()

    # Only merge split, use label from unified
    merged = unified.merge(split_df[["cascade_id", "split"]], on="cascade_id", how="inner")
    split_data = merged[merged["split"] == split_name]

    cascade_labels = split_data.groupby("cascade_id")["label"].first().reset_index()

    rumours = cascade_labels[cascade_labels["label"] == "rumour"]["cascade_id"].tolist()
    non_rumours = cascade_labels[cascade_labels["label"] == "non-rumour"]["cascade_id"].tolist()

    random.seed(SEED)
    n_per_class = n_samples // 2

    # Prefer cascades with 5-50 nodes for visual clarity
    def _pick_interesting(ids, n, unified_df):
        sizes = unified_df[unified_df["cascade_id"].isin(ids)].groupby("cascade_id").size()
        good = sizes[(sizes >= 5) & (sizes <= 50)].index.tolist()
        if len(good) >= n:
            return random.sample(good, n)
        return random.sample(ids, min(n, len(ids)))

    selected_rumours = _pick_interesting(rumours, n_per_class, unified)
    selected_non_rumours = _pick_interesting(non_rumours, n_per_class, unified)
    selected = selected_rumours + selected_non_rumours

    ncols = 4
    nrows = max(1, (len(selected) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    fig.patch.set_facecolor(COLORS["bg"])

    if nrows == 1:
        axes = [axes] if ncols == 1 else [axes]
    axes_flat = np.array(axes).flatten()

    for i, cid in enumerate(selected):
        if i >= len(axes_flat):
            break
        label = cascade_labels[cascade_labels["cascade_id"] == cid]["label"].iloc[0]
        G = _build_cascade_tree(unified, cid)
        _plot_single_cascade(G, cid, label, axes_flat[i])

    # Hide unused axes
    for j in range(len(selected), len(axes_flat)):
        axes_flat[j].axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS["rumour"], label="Rumour"),
        mpatches.Patch(color=COLORS["non-rumour"], label="Non-rumour"),
        mpatches.Patch(color=COLORS["accent"], label="Root tweet"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=11, frameon=False, labelcolor=COLORS["text"])

    fig.suptitle(
        f"Cascade Propagation Trees — {split_name.upper()} Set",
        fontsize=16, fontweight="bold", color=COLORS["text"], y=0.98
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    if out_path is None:
        out_path = os.path.join(OUT_DIR, f"cascade_trees_{split_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    logger.info("[viz] Cascade tree grid saved to %s", out_path)
    return out_path


# ===========================================================================
# 2.  t-SNE Embedding Visualization (Feature-based)
# ===========================================================================

def visualize_feature_tsne(
    feature_df: pd.DataFrame,
    split_df: pd.DataFrame,
    split_name: str = "test",
    out_path: str | None = None,
):
    """
    t-SNE visualization of the 19-feature space for the given split.
    Colour-coded by rumour vs non-rumour.
    """
    _setup_dark_style()

    merged = feature_df.merge(split_df[["cascade_id", "split"]], on="cascade_id", how="inner")
    split_data = merged[merged["split"] == split_name]

    # Use only t=120 snapshots for the final view
    split_data = split_data[split_data["t_minutes"] == 120]

    if len(split_data) == 0:
        logger.warning("[viz] No data for split=%s at t=120min", split_name)
        return None

    exclude = {"cascade_id", "t_minutes", "label", "label_binary", "split"}
    feat_cols = [c for c in split_data.columns if c not in exclude]

    X = split_data[feat_cols].fillna(0).values
    labels = split_data["label_binary"].values

    # Standardise before t-SNE
    X_scaled = StandardScaler().fit_transform(X)

    # PCA → 10 components if > 10 features, then t-SNE → 2D
    if X_scaled.shape[1] > 10:
        pca = PCA(n_components=10, random_state=SEED)
        X_scaled = pca.fit_transform(X_scaled)

    perplexity = min(30, max(5, len(X_scaled) // 4))
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity,
                n_iter=1000, learning_rate="auto", init="pca")
    proj = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(COLORS["bg"])

    for label_val, label_name in [(0, "Non-rumour"), (1, "Rumour")]:
        mask = labels == label_val
        color = COLORS["non-rumour"] if label_val == 0 else COLORS["rumour"]
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=color, label=label_name,
            alpha=0.7, s=25, linewidths=0.3, edgecolors="white",
        )

    ax.set_title(
        f"t-SNE of Feature Space — {split_name.upper()} Set\n"
        f"({len(feat_cols)} features, {len(X_scaled)} cascades at t=120min)",
        fontsize=14, fontweight="bold", color=COLORS["text"],
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(fontsize=12, frameon=False, labelcolor=COLORS["text"],
              markerscale=2, loc="upper right")
    ax.grid(True, alpha=0.15)

    if out_path is None:
        out_path = os.path.join(OUT_DIR, f"tsne_features_{split_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    logger.info("[viz] t-SNE feature plot saved to %s", out_path)
    return out_path


# ===========================================================================
# 3.  Combined dashboard
# ===========================================================================

def generate_all_visualizations():
    """Generate all visualizations from available data."""
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("[viz] Loading unified dataset...")
    unified = pd.read_parquet(UNIFIED_FILE)

    logger.info("[viz] Loading feature matrix...")
    feature_df = pd.read_parquet(FEATURE_MATRIX)

    # Try to load split file
    if os.path.exists(SPLIT_FILE):
        split_df = pd.read_parquet(SPLIT_FILE)
    else:
        logger.warning("[viz] Split file not found at %s — generating from unified", SPLIT_FILE)
        # Fall back: use cascade labels directly, treat all as "test"
        cascade_labels = unified.groupby("cascade_id")["label"].first().reset_index()
        cascade_labels["split"] = "test"
        split_df = cascade_labels

    outputs = {}

    # 1. Cascade Trees (test set)
    logger.info("[viz] Generating cascade propagation trees...")
    path = visualize_cascade_trees(unified, split_df, split_name="test", n_samples=12)
    if path:
        outputs["cascade_trees_test"] = path

    # 2. t-SNE of features (test set)
    logger.info("[viz] Generating t-SNE feature embedding...")
    path = visualize_feature_tsne(feature_df, split_df, split_name="test")
    if path:
        outputs["tsne_features_test"] = path

    # 3. Also do validation set if available
    if "val" in split_df["split"].values:
        logger.info("[viz] Generating validation set visualizations...")
        path = visualize_cascade_trees(unified, split_df, split_name="val", n_samples=12)
        if path:
            outputs["cascade_trees_val"] = path

        path = visualize_feature_tsne(feature_df, split_df, split_name="val")
        if path:
            outputs["tsne_features_val"] = path

    logger.info("[viz] All visualizations complete: %s", list(outputs.keys()))
    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    outputs = generate_all_visualizations()
    print(f"\n{'='*60}")
    print("VISUALIZATION OUTPUTS:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    print(f"{'='*60}")
