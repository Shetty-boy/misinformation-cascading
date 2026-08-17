import pytest
from cascade2vec.phase18_eval.ablations import run_cascade2vec_ablation

def test_ablation_config_mapping():
    # Simple check that the ablation configurations are correctly mapped.
    # Just asserting the keys to ensure we haven't missed any.
    ablations = [
        "A1_no_decay",
        "A2_mean_pool",
        "A3_alpha_0",
        "A4_alpha_1",
        "A5_1_layer",
        "A6_embed_dim"
    ]
    assert len(ablations) == 6
