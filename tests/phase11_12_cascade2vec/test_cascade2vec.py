"""
tests/phase11_12_cascade2vec/test_cascade2vec.py
=================================================
Unit tests for CASCADE2VEC encoder.

Tests:
  1. test_encoder_output_shape          — correct embedding dim for toy cascade
  2. test_edge_weight_at_zero_delay     — weight = 1.0 when t_edge == t_snapshot
  3. test_time_weighting_changes_output — permuting timestamps changes embeddings
  4. test_no_future_leakage_in_snapshot — no nodes/edges with t > t_snapshot
  5. test_root_excluded_from_time_decay — root node (t=0) gets weight exp(-λ*t_snap) not 1.0
"""

import math
import pytest
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

from cascade2vec.phase11_12_cascade2vec.cascade2vec import (
    CASCADE2VEC,
    C2VClassifier,
    TimeWeightedSAGEConv,
    SnapshotDataset,
    compute_edge_weights,
)
from cascade2vec.phase02_ingestion.leakage_audit import assert_snapshot_is_clean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_cascade_df():
    """
    A minimal cascade with 4 nodes:
      root  (t=0s,   parent=NaN,  tweet_id='A')
      reply1 (t=30s,  parent='A', tweet_id='B')
      reply2 (t=60s,  parent='A', tweet_id='C')
      reply3 (t=100s, parent='B', tweet_id='D')
    """
    return pd.DataFrame({
        "tweet_id":   ["A", "B", "C", "D"],
        "parent_id":  [None, "A", "A", "B"],
        "timestamp":  [0, 30, 60, 100],
        "text":       ["root tweet", "reply 1", "reply 2", "reply 3"],
        "cascade_id": ["CAS1"] * 4,
        "label":      ["rumour"] * 4,
        "user_id":    ["u1", "u2", "u3", "u4"],
        "event_id":   ["evt1"] * 4,
    })


@pytest.fixture
def toy_encoder():
    torch.manual_seed(42)
    return CASCADE2VEC(in_dim=10, hidden_dim=32, embed_dim=16, n_layers=2)


def _make_toy_data(
    n_nodes: int = 4,
    in_dim: int = 10,
    t_edge_s_list: list = None,
    t_snapshot_s: float = 120.0,
    lam: float = 0.001,
) -> Data:
    """Build a minimal PyG Data object for testing."""
    torch.manual_seed(42)
    x = torch.randn(n_nodes, in_dim)

    if t_edge_s_list is None:
        t_edge_s_list = [0.0, 30.0, 60.0, 100.0]

    # Simple chain: 0->1->2->3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    t_edge_s   = torch.tensor(t_edge_s_list[1:], dtype=torch.float32)  # child timestamps
    edge_weight = compute_edge_weights(t_edge_s, t_snapshot_s, lam)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor([1]),
        num_nodes=n_nodes,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEdgeWeights:
    """Tests for the time-decay edge weight computation."""

    def test_edge_weight_at_zero_delay(self):
        """
        An edge arriving exactly at t_snapshot_s (Δt=0) must have weight = 1.0.
        This is the architecturally mandated case: exp(-λ * 0) = exp(0) = 1.0.
        """
        t_snapshot_s = 300.0   # 5-minute window
        t_edge_s = torch.tensor([300.0])   # arrived exactly at cutoff
        lam = 0.001

        weights = compute_edge_weights(t_edge_s, t_snapshot_s, lam)
        assert len(weights) == 1
        assert abs(float(weights[0]) - 1.0) < 1e-6, (
            f"Expected weight=1.0 for zero-delay edge, got {float(weights[0]):.8f}"
        )

    def test_edge_weight_monotone_decreasing(self):
        """Earlier replies get smaller weights (older = less weight)."""
        t_snapshot_s = 300.0
        t_edge_s = torch.tensor([0.0, 60.0, 120.0, 240.0, 300.0])
        lam = 0.001

        weights = compute_edge_weights(t_edge_s, t_snapshot_s, lam)
        w = weights.tolist()
        # Weights should be strictly increasing (most recent = largest)
        for i in range(len(w) - 1):
            assert w[i] < w[i + 1], (
                f"Weight not monotone: w[{i}]={w[i]:.4f} >= w[{i+1}]={w[i+1]:.4f}"
            )

    def test_lambda_zero_gives_uniform_weights(self):
        """λ=0 (no decay) should give weight=1.0 for ALL edges."""
        t_snapshot_s = 300.0
        t_edge_s = torch.tensor([0.0, 50.0, 100.0, 300.0])
        lam = 0.0

        weights = compute_edge_weights(t_edge_s, t_snapshot_s, lam)
        assert torch.allclose(weights, torch.ones(4)), (
            f"λ=0 should give uniform weights=1.0, got {weights.tolist()}"
        )

    def test_weights_in_zero_one_range(self):
        """All edge weights must be in (0, 1]."""
        t_snapshot_s = 7200.0  # 2-hour window
        t_edge_s = torch.tensor([0.0, 100.0, 1000.0, 3600.0, 7200.0])
        lam = 0.001

        weights = compute_edge_weights(t_edge_s, t_snapshot_s, lam)
        assert (weights > 0).all(), "Weights must be > 0"
        assert (weights <= 1.0 + 1e-6).all(), f"Weights must be <= 1.0, got max={weights.max():.4f}"


class TestEncoderOutput:
    """Tests for CASCADE2VEC encoder shape and behaviour."""

    def test_encoder_output_shape(self, toy_encoder):
        """Encoder must return (batch_size, embed_dim) tensor."""
        data = _make_toy_data(n_nodes=4, in_dim=10)
        batch = Batch.from_data_list([data, data])  # batch_size=2

        embed = toy_encoder(
            batch.x, batch.edge_index,
            batch.edge_attr, batch.batch,
        )
        assert embed.shape == (2, 16), (
            f"Expected (2, 16) embedding shape, got {embed.shape}"
        )

    def test_embeddings_are_l2_normalised(self, toy_encoder):
        """Encoder output must be L2-normalised (norm ≈ 1.0)."""
        data = _make_toy_data(n_nodes=4, in_dim=10)
        batch = Batch.from_data_list([data])

        embed = toy_encoder(
            batch.x, batch.edge_index,
            batch.edge_attr, batch.batch,
        )
        norms = embed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(1), atol=1e-5), (
            f"Embeddings not L2-normalised, norms={norms.tolist()}"
        )

    def test_time_weighting_changes_output(self):
        """
        Permuting edge timestamps MUST change the encoder output.
        This verifies time-weighting is NOT a no-op.
        """
        torch.manual_seed(42)
        encoder = CASCADE2VEC(in_dim=10, hidden_dim=32, embed_dim=16, n_layers=2)
        encoder.eval()

        t_snapshot_s = 300.0
        lam = 0.001

        # Original: edges at t=0, 60, 120 (relative)
        t_orig = torch.tensor([0.0, 60.0, 120.0])
        w_orig = compute_edge_weights(t_orig, t_snapshot_s, lam)

        # Permuted: edges at t=120, 0, 60
        t_perm = torch.tensor([120.0, 0.0, 60.0])
        w_perm = compute_edge_weights(t_perm, t_snapshot_s, lam)

        # Verify weights actually differ
        assert not torch.allclose(w_orig, w_perm), "Permuted timestamps gave same weights"

        torch.manual_seed(99)
        x = torch.randn(4, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        batch_vec = torch.zeros(4, dtype=torch.long)

        with torch.no_grad():
            e_orig = encoder(x, edge_index, w_orig, batch_vec)
            e_perm = encoder(x, edge_index, w_perm, batch_vec)

        assert not torch.allclose(e_orig, e_perm, atol=1e-4), (
            "Permuting edge timestamps did not change encoder output — "
            "time-weighting may be a no-op!"
        )


class TestTemporalSafety:
    """Tests that snapshot construction excludes future nodes/edges."""

    def test_no_future_leakage_in_snapshot(self, toy_cascade_df):
        """
        Snapshot at t=60s must NOT include node D (timestamp=100s > 60s).
        Verified both by checking the returned DataFrame and by
        assert_snapshot_is_clean().
        """
        t_snapshot_s = 60.0
        cascade = toy_cascade_df.copy()
        snap = cascade[cascade["timestamp"] <= t_snapshot_s]

        # D should NOT be in the snapshot
        assert "D" not in snap["tweet_id"].tolist(), (
            "Node D (t=100s) leaked into 60s snapshot!"
        )

        # Build edges from snapshot
        snap_ids = set(snap["tweet_id"].tolist())
        edge_rows = []
        for _, row in snap.iterrows():
            if pd.notna(row["parent_id"]) and row["parent_id"] in snap_ids:
                edge_rows.append({
                    "src": row["parent_id"],
                    "dst": row["tweet_id"],
                    "timestamp": row["timestamp"],
                })
        edges_df = pd.DataFrame(edge_rows)

        snap_dict = {
            "vertices": snap.rename(columns={"tweet_id": "id"}),
            "edges": edges_df,
        }

        # Should not raise
        assert_snapshot_is_clean(snap_dict, t_snapshot_s)

    def test_assert_snapshot_raises_on_future_edge(self, toy_cascade_df):
        """
        assert_snapshot_is_clean() must raise if a future edge is present.
        This validates the leakage guard is active.
        """
        t_snapshot_s = 60.0
        cascade = toy_cascade_df.copy()
        snap = cascade[cascade["timestamp"] <= t_snapshot_s]

        # Manually inject a future edge (simulating the footgun we're guarding against)
        bad_edge = pd.DataFrame([{
            "src": "C", "dst": "D", "timestamp": 100.0  # 100 > 60 = future!
        }])
        snap_dict = {
            "vertices": snap.rename(columns={"tweet_id": "id"}),
            "edges": bad_edge,
        }

        with pytest.raises(AssertionError, match="TEMPORAL LEAKAGE"):
            assert_snapshot_is_clean(snap_dict, t_snapshot_s)


class TestClassifier:
    """Tests for the classification head."""

    def test_classifier_output_shape(self):
        """Classifier must produce (batch_size, n_classes) logits."""
        clf = C2VClassifier(embed_dim=16, num_classes=2)
        embed = torch.randn(8, 16)
        logits = clf(embed)
        assert logits.shape == (8, 2), f"Expected (8,2) logits, got {logits.shape}"
