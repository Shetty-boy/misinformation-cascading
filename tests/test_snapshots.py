import pytest
from pyspark.sql import Row
from graphframes import GraphFrame

from src.graph.phase04_graph.loader import get_spark
from src.graph.snapshots import get_snapshot

@pytest.fixture(scope="session")
def spark():
    """Provides a SparkSession for testing."""
    return get_spark()

@pytest.fixture
def toy_cascade(spark):
    """
    Creates a toy cascade with 5 nodes:
    root (0s)
    |- reply1 (30s)
    |  |- reply1_1 (90s)
    |- reply2 (150s)
       |- reply2_1 (300s)
    """
    vertices = spark.createDataFrame([
        Row(id="root", timestamp=0.0),
        Row(id="reply1", timestamp=30.0),
        Row(id="reply1_1", timestamp=90.0),
        Row(id="reply2", timestamp=150.0),
        Row(id="reply2_1", timestamp=300.0)
    ])
    
    edges = spark.createDataFrame([
        Row(src="root", dst="reply1", cascade_id="root"),
        Row(src="reply1", dst="reply1_1", cascade_id="root"),
        Row(src="root", dst="reply2", cascade_id="root"),
        Row(src="reply2", dst="reply2_1", cascade_id="root")
    ])
    
    return GraphFrame(vertices, edges)

def test_get_snapshot_1_minute(toy_cascade):
    """Test snapshot at t=1 minute (60 seconds)."""
    # Should include root (0s) and reply1 (30s).
    snapshot = get_snapshot(toy_cascade, 1.0)
    
    assert snapshot.vertices.count() == 2
    assert snapshot.edges.count() == 1
    
    v_ids = [row.id for row in snapshot.vertices.collect()]
    assert set(v_ids) == {"root", "reply1"}

def test_get_snapshot_2_minutes(toy_cascade):
    """Test snapshot at t=2 minutes (120 seconds)."""
    # Should include root (0s), reply1 (30s), reply1_1 (90s).
    snapshot = get_snapshot(toy_cascade, 2.0)
    
    assert snapshot.vertices.count() == 3
    assert snapshot.edges.count() == 2
    
    v_ids = [row.id for row in snapshot.vertices.collect()]
    assert set(v_ids) == {"root", "reply1", "reply1_1"}

def test_get_snapshot_5_minutes(toy_cascade):
    """Test snapshot at t=5 minutes (300 seconds)."""
    # Should include all nodes (up to 300s).
    snapshot = get_snapshot(toy_cascade, 5.0)
    
    assert snapshot.vertices.count() == 5
    assert snapshot.edges.count() == 4

def test_get_snapshot_0_minutes(toy_cascade):
    """Test snapshot at t=0 minutes."""
    # Should include only the root.
    snapshot = get_snapshot(toy_cascade, 0.0)
    
    assert snapshot.vertices.count() == 1
    assert snapshot.edges.count() == 0
