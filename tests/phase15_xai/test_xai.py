import pytest
import os
import torch
from torch_geometric.data import Data
from cascade2vec.phase15_xai.gnn_explain import filter_explainable_cascades

def test_filter_explainable_cascades():
    # 1. Singleton cascade (no edges)
    d1 = Data(
        x=torch.randn(1, 10),
        edge_index=torch.zeros((2, 0), dtype=torch.long)
    )
    
    # 2. Connected cascade (has edges)
    d2 = Data(
        x=torch.randn(3, 10),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    )
    
    data_list = [d1, d2]
    filtered = filter_explainable_cascades(data_list)
    
    assert len(filtered) == 1
    assert filtered[0].x.size(0) == 3
