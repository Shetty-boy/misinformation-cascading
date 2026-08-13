import pytest
import pandas as pd
import numpy as np

from cascade2vec.phase06_07_features.feature_selection import (
    prune_correlated_features,
)

@pytest.fixture
def mock_feature_matrix():
    # Create a mock dataframe with correlated features
    np.random.seed(42)
    n_samples = 100
    
    # Independent features
    feat_a = np.random.randn(n_samples)
    
    # Highly correlated feature with feat_a
    feat_b = feat_a * 1.5 + np.random.normal(0, 0.01, n_samples)
    
    # Another independent feature
    feat_c = np.random.randn(n_samples)
    
    # Target label: heavily dependent on feat_b, so feat_b should have a higher F-statistic
    # Let's say label is 1 if feat_b > 0 else 0
    label = (feat_b > 0).astype(int)
    
    df = pd.DataFrame({
        "feat_a": feat_a,
        "feat_b": feat_b,
        "feat_c": feat_c,
        "label_binary": label
    })
    
    return df

def test_prune_correlated_features(mock_feature_matrix):
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    
    # Feat a and b are highly correlated (r ~ 1.0)
    survivors, drop_reasons = prune_correlated_features(
        mock_feature_matrix, feature_cols, threshold=0.90
    )
    
    # Should drop one of them
    assert len(survivors) == 2
    assert "feat_c" in survivors
    
    # feat_b has direct causal link to label, its F-statistic will be higher, so it should be kept
    # feat_a should be dropped
    assert "feat_b" in survivors
    assert "feat_a" not in survivors
    
    assert len(drop_reasons) == 1
    assert drop_reasons[0]["dropped"] == "feat_a"
    assert drop_reasons[0]["kept"] == "feat_b"
    assert drop_reasons[0]["correlation"] > 0.90
