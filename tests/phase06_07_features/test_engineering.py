"""
test_engineering.py — Phase 6 Feature Engineering Tests
=========================================================
Tests structured around 4 cascade shapes:
  - Singleton:       1 node, 0 edges
  - Chain:           root -> A -> B -> C (depth=3)
  - Star:            root -> A, B, C (depth=1, branching=3)
  - Disconnected:    root -> A, then orphaned B -> C (C unreachable from root)

Leakage tests:
  - Temporal cutoff: events before/after t are correctly included/excluded
  - assert_snapshot_is_clean() raises AssertionError on tampered snapshots
"""

import math
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Use a local Spark for unit tests (minimal config, no GraphFrames packages needed)
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, BooleanType, IntegerType
)
from graphframes import GraphFrame

from cascade2vec.phase02_ingestion.leakage_audit import assert_snapshot_is_clean


# ---------------------------------------------------------------------------
# Session-scoped Spark fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spark():
    import os
    jar_pkg = os.getenv("GRAPHFRAMES_PACKAGE", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
    s = (
        SparkSession.builder
        .master("local[2]")
        .appName("test-engineering")
        .config("spark.jars.packages", jar_pkg)
        .config("spark.sql.shuffle.partitions", "2")
        # Use local checkpoint dir instead of HDFS
        .config("spark.local.dir", "/tmp/spark-test-checkpoints")
        .getOrCreate()
    )
    s.sparkContext.setLogLevel("ERROR")
    s.sparkContext.setCheckpointDir("/tmp/spark-test-checkpoints")
    yield s
    # Graceful stop — don't force kill so JVM cleanup can complete
    try:
        s.stop()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Graph factory helpers
# ---------------------------------------------------------------------------

def _make_vertices(spark, rows):
    """rows: list of (id, cascade_id, parent_id, timestamp)"""
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("cascade_id", StringType(), False),
        StructField("parent_id", StringType(), True),
        StructField("timestamp", LongType(), True),
        StructField("label", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("text", StringType(), True),
        StructField("event_id", StringType(), True),
    ])
    data = [(r[0], r[1], r[2], r[3], "rumour", "u1", "t", "ev1") for r in rows]
    return spark.createDataFrame(data, schema)


def _make_edges(spark, rows):
    """rows: list of (src, dst, cascade_id)"""
    schema = StructType([
        StructField("src", StringType(), False),
        StructField("dst", StringType(), False),
        StructField("cascade_id", StringType(), False),
    ])
    return spark.createDataFrame(rows, schema)


def _make_graph(spark, v_rows, e_rows):
    v = _make_vertices(spark, v_rows)
    e = _make_edges(spark, e_rows)
    return GraphFrame(v, e)


# ---------------------------------------------------------------------------
# Test cascade shapes
# ---------------------------------------------------------------------------

class TestSingletonCascade:
    """Single node, zero edges."""

    def test_structural_features(self, spark):
        from cascade2vec.phase06_07_features.engineering import _compute_snapshot_structural
        cid = "singleton_cascade"
        v_rows = [("root", cid, None, 0)]
        e_rows = []
        snap = _make_graph(spark, v_rows, e_rows)
        snap_pd = {"vertices": snap.vertices.toPandas(), "edges": snap.edges.toPandas()}
        feats = _compute_snapshot_structural(snap_pd, t_seconds=3600, cascade_id=cid)

        assert feats["node_count"] == 1
        assert feats["edge_count"] == 0
        assert feats["max_depth"] == 0.0
        assert feats["avg_depth"] == 0.0
        assert feats["leaf_count"] == 1  # All nodes are leaves
        assert feats["leaf_ratio"] == 1.0
        assert feats["branching_factor"] == 0.0  # No internal nodes → convention = 0.0
        assert feats["root_degree"] == 0
        assert feats["reachable_ratio"] == 1.0  # Root is always reachable from itself
        assert feats["is_connected"] is True

    def test_singleton_temporal_features(self, spark):
        """n_intervals < 2 → std_interarrival=0.0, burstiness=0.0 policy."""
        from cascade2vec.phase06_07_features.engineering import _compute_snapshot_temporal
        cid = "singleton_cascade"
        v_rows = [("root", cid, None, 0)]
        v = _make_vertices(spark, v_rows)
        feats = _compute_snapshot_temporal(v.toPandas(), t_seconds=3600)

        assert feats["std_interarrival"] == 0.0
        assert feats["burstiness"] == 0.0
        assert feats["mean_interarrival"] == 0.0


class TestChainCascade:
    """root -> A -> B -> C, depth=3."""

    def test_structural_features(self, spark):
        from cascade2vec.phase06_07_features.engineering import _compute_snapshot_structural
        cid = "chain_cascade"
        # timestamps: root=0, A=60, B=120, C=180
        v_rows = [
            ("root", cid, None, 0),
            ("A", cid, "root", 60),
            ("B", cid, "A", 120),
            ("C", cid, "B", 180),
        ]
        e_rows = [
            ("root", "A", cid),
            ("A", "B", cid),
            ("B", "C", cid),
        ]
        snap = _make_graph(spark, v_rows, e_rows)
        snap_pd = {"vertices": snap.vertices.toPandas(), "edges": snap.edges.toPandas()}
        feats = _compute_snapshot_structural(snap_pd, t_seconds=3600, cascade_id=cid)

        assert feats["node_count"] == 4
        assert feats["edge_count"] == 3
        assert feats["max_depth"] == 3.0
        # avg_depth = (0+1+2+3)/4 = 1.5
        assert feats["avg_depth"] == pytest.approx(1.5, abs=0.01)
        assert feats["leaf_count"] == 1  # Only C is a leaf
        assert feats["leaf_ratio"] == pytest.approx(0.25, abs=0.01)
        # branching_factor = 3 edges / 3 internal nodes (root, A, B) = 1.0
        assert feats["branching_factor"] == pytest.approx(1.0, abs=0.01)
        assert feats["root_degree"] == 1  # Root only has A as child
        assert feats["reachable_ratio"] == 1.0
        assert feats["is_connected"] is True


class TestStarCascade:
    """root -> A, root -> B, root -> C."""

    def test_structural_features(self, spark):
        from cascade2vec.phase06_07_features.engineering import _compute_snapshot_structural
        cid = "star_cascade"
        v_rows = [
            ("root", cid, None, 0),
            ("A", cid, "root", 60),
            ("B", cid, "root", 120),
            ("C", cid, "root", 180),
        ]
        e_rows = [
            ("root", "A", cid),
            ("root", "B", cid),
            ("root", "C", cid),
        ]
        snap = _make_graph(spark, v_rows, e_rows)
        snap_pd = {"vertices": snap.vertices.toPandas(), "edges": snap.edges.toPandas()}
        feats = _compute_snapshot_structural(snap_pd, t_seconds=3600, cascade_id=cid)

        assert feats["node_count"] == 4
        assert feats["edge_count"] == 3
        assert feats["max_depth"] == 1.0
        assert feats["avg_depth"] == pytest.approx(0.75, abs=0.01)  # (0+1+1+1)/4
        assert feats["leaf_count"] == 3  # A, B, C are all leaves
        assert feats["leaf_ratio"] == pytest.approx(0.75, abs=0.01)
        # branching_factor = 3 edges / 1 internal node (root) = 3.0
        assert feats["branching_factor"] == pytest.approx(3.0, abs=0.01)
        assert feats["root_degree"] == 3
        assert feats["reachable_ratio"] == 1.0
        assert feats["is_connected"] is True


class TestDisconnectedCascade:
    """root -> A, then orphaned B -> C (C unreachable from root)."""

    def test_structural_features(self, spark):
        from cascade2vec.phase06_07_features.engineering import _compute_snapshot_structural
        cid = "disc_cascade"
        v_rows = [
            ("root", cid, None, 0),
            ("A", cid, "root", 60),
            ("B", cid, None, 90),   # Second orphaned root (simulates deleted parent)
            ("C", cid, "B", 120),
        ]
        e_rows = [
            ("root", "A", cid),
            ("B", "C", cid),  # This sub-tree is disconnected from root
        ]
        snap = _make_graph(spark, v_rows, e_rows)
        snap_pd = {"vertices": snap.vertices.toPandas(), "edges": snap.edges.toPandas()}
        feats = _compute_snapshot_structural(snap_pd, t_seconds=3600, cascade_id=cid)

        assert feats["node_count"] == 4
        # B and C are unreachable from root (since we use root's BFS only)
        assert feats["reachable_ratio"] < 1.0
        assert feats["is_connected"] is False


# ---------------------------------------------------------------------------
# Temporal cutoff tests (leakage)
# ---------------------------------------------------------------------------

class TestTemporalCutoff:
    """Verify that features only use pre-cutoff data."""

    def test_only_pre_cutoff_nodes_included(self, spark):
        """Snapshot at t=60s must exclude nodes with timestamp > 60."""
        from cascade2vec.phase04_05_graph.snapshots import get_snapshot
        cid = "cutoff_test"
        v_rows = [
            ("root", cid, None, 0),
            ("early", cid, "root", 30),
            ("future", cid, "root", 120),  # AFTER cutoff — must be excluded
        ]
        e_rows = [
            ("root", "early", cid),
            ("root", "future", cid),
        ]
        full_graph = _make_graph(spark, v_rows, e_rows)
        snap = get_snapshot(full_graph, t_minutes=1.0)  # t=60s

        node_ids = {r["id"] for r in snap.vertices.collect()}
        assert "root" in node_ids
        assert "early" in node_ids
        assert "future" not in node_ids, "Future node leaked into snapshot!"

    def test_feature_values_match_manual_pre_cutoff_computation(self, spark):
        """Features at t=60s must exactly match hand-computed values using only pre-cutoff events."""
        from cascade2vec.phase04_05_graph.snapshots import get_snapshot
        from cascade2vec.phase06_07_features.engineering import _compute_snapshot_structural
        cid = "manual_check"
        v_rows = [
            ("root", cid, None, 0),
            ("A", cid, "root", 30),
            ("B", cid, "A", 90),    # AFTER t=60s cutoff — excluded
        ]
        e_rows = [
            ("root", "A", cid),
            ("A", "B", cid),
        ]
        full_graph = _make_graph(spark, v_rows, e_rows)
        snap = get_snapshot(full_graph, t_minutes=1.0)  # t=60s

        # Manual computation using only pre-cutoff events: root + A
        snap_pd = {"vertices": snap.vertices.toPandas(), "edges": snap.edges.toPandas()}
        feats = _compute_snapshot_structural(snap_pd, t_seconds=60.0, cascade_id=cid)

        assert feats["node_count"] == 2    # root + A
        assert feats["edge_count"] == 1    # root -> A
        assert feats["max_depth"] == 1.0   # A is at depth 1
        assert feats["leaf_count"] == 1    # Only A is a leaf
        assert feats["root_degree"] == 1   # root has 1 child

    def test_assert_snapshot_is_clean_raises_on_tampered_graph(self, spark):
        """assert_snapshot_is_clean() must raise AssertionError when given a temporally invalid graph."""
        cid = "tamper_test"
        v_rows = [
            ("root", cid, None, 0),
            ("future_node", cid, "root", 9999),  # timestamp >> cutoff
        ]
        e_rows = [("root", "future_node", cid)]
        tampered_snap = _make_graph(spark, v_rows, e_rows)

        with pytest.raises(AssertionError, match="TEMPORAL LEAKAGE"):
            assert_snapshot_is_clean(tampered_snap, t_seconds=60.0)


# ---------------------------------------------------------------------------
# Numerical stability: n_intervals < 2 policy
# ---------------------------------------------------------------------------

class TestNumericalStability:

    def test_two_node_temporal_features(self, spark):
        """Two nodes: 1 interarrival interval → std=0, burstiness=0."""
        from cascade2vec.phase06_07_features.engineering import _compute_snapshot_temporal
        cid = "two_node"
        v_rows = [
            ("root", cid, None, 0),
            ("A", cid, "root", 60),
        ]
        v = _make_vertices(spark, v_rows)
        feats = _compute_snapshot_temporal(v.toPandas(), t_seconds=3600)

        # With exactly 1 interval, std_interarrival = 0.0 by convention
        assert feats["mean_interarrival"] == pytest.approx(60.0)
        assert feats["std_interarrival"] == 0.0
        assert feats["burstiness"] == 0.0

    def test_branching_factor_zero_for_no_internal_nodes(self, spark):
        """branching_factor must be 0.0 (not NaN) for a singleton."""
        from cascade2vec.phase06_07_features.engineering import _compute_snapshot_structural
        cid = "bf_stability"
        v_rows = [("root", cid, None, 0)]
        e_rows = []
        snap = _make_graph(spark, v_rows, e_rows)
        snap_pd = {"vertices": snap.vertices.toPandas(), "edges": snap.edges.toPandas()}
        feats = _compute_snapshot_structural(snap_pd, t_seconds=3600, cascade_id=cid)

        assert feats["branching_factor"] == 0.0
        assert not math.isnan(feats["branching_factor"])
