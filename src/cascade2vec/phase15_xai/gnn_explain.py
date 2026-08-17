import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def filter_explainable_cascades(data_list: List[Data]) -> List[Data]:
    """
    Filter out singletons and disconnected cascades.
    GNNExplainer needs edges to attribute importance over.
    """
    valid_data = []
    for data in data_list:
        # Check if it has edges
        if data.edge_index.numel() == 0:
            continue
            
        # Optional: More advanced check for connectivity could be added here
        # For now, having at least one edge is the minimum requirement for GNNExplainer
        valid_data.append(data)
        
    return valid_data

def build_explainer(model: torch.nn.Module, epochs: int = 200) -> Explainer:
    """
    Initialize a PyG Explainer with GNNExplainer algorithm.
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='probs',
        ),
    )
    return explainer

def explain_cascade(explainer: Explainer, data: Data, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run GNNExplainer on a single cascade data object.
    Returns (node_mask, edge_mask).
    """
    explanation = explainer(
        x=data.x, 
        edge_index=data.edge_index, 
        edge_weight=data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr.numel() > 0 else None,
        batch=data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long)
    )
    return explanation.node_mask, explanation.edge_mask

def plot_cascade_explanation(data: Data, edge_mask: torch.Tensor, node_mask: torch.Tensor, save_path: str):
    """
    Visualize the explanation masks over the graph topology using NetworkX.
    """
    try:
        # Move to CPU for plotting
        edge_index = data.edge_index.cpu().numpy()
        
        # Flatten the edge mask properly
        if len(edge_mask.shape) > 1:
            edge_mask_np = edge_mask.squeeze().cpu().numpy()
        else:
            edge_mask_np = edge_mask.cpu().numpy()
            
        if len(node_mask.shape) > 1:
            node_mask_np = node_mask.squeeze().cpu().numpy()
        else:
            node_mask_np = node_mask.cpu().numpy()

        # Handle case where node_mask_np is a 2D array of node features
        if node_mask_np.ndim > 1:
            node_mask_np = node_mask_np.sum(axis=1) # Sum importance across features
            
        G = nx.DiGraph()
        
        num_nodes = data.x.size(0)
        for i in range(num_nodes):
            G.add_node(i, weight=float(node_mask_np[i]) if i < len(node_mask_np) else 0.0)
            
        if edge_index.shape[0] == 2:
            num_edges = edge_index.shape[1]
            src = edge_index[0]
            dst = edge_index[1]
        else:
            num_edges = edge_index.shape[0]
            src = edge_index[:, 0]
            dst = edge_index[:, 1]
            
        for i in range(num_edges):
            u, v = int(src[i]), int(dst[i])
            w = float(edge_mask_np[i]) if i < len(edge_mask_np) else 0.0
            G.add_edge(u, v, weight=w)
            
        plt.figure(figsize=(10, 8))
        
        node_colors = [G.nodes[i].get('weight', 0) for i in G.nodes()]
        edge_colors = [G[u][v].get('weight', 0) for u, v in G.edges()]
        edge_widths = [1 + 3 * w for w in edge_colors]
        
        try:
            pos = nx.spring_layout(G, seed=42)
        except Exception:
            pos = nx.random_layout(G)
            
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Blues, node_size=200)
        
        if len(G.edges()) > 0:
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Reds, 
                                 width=edge_widths, arrows=True, arrowsize=15)
                                 
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f"Cascade {getattr(data, 'cascade_id', 'unknown')} GNNExplainer Edge & Node Importance")
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), label="Edge Importance")
        plt.axis('off')
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting cascade {getattr(data, 'cascade_id', 'unknown')}: {e}")
        plt.close()

def aggregate_explanations(results: List[dict]) -> pd.DataFrame:
    """
    Aggregate explanation results to compute mean edge importance by depth level.
    """
    df_rows = []
    for res in results:
        df_rows.append({
            'cascade_id': res['cascade_id'],
            'mean_edge_importance': float(res['edge_mask'].mean()) if res['edge_mask'].numel() > 0 else 0.0,
            'mean_node_importance': float(res['node_mask'].mean()) if res['node_mask'].numel() > 0 else 0.0,
            'max_edge_importance': float(res['edge_mask'].max()) if res['edge_mask'].numel() > 0 else 0.0,
        })
    return pd.DataFrame(df_rows)
