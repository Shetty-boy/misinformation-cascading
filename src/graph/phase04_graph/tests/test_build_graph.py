"""
test_build_graph.py — Phase 4 Unit Tests
=========================================
Tests for the cascade graph construction pipeline.

All tests use a synthetic 5-node toy cascade built entirely from plain
Python dicts converted to Spark DataFrames — no file I/O required.

Toy cascade structure (reply tree):
    root (A)
     ├── B  (replies to A)
     │   └── D  (replies to B)
     └── C  (replies to A)
         └── E  (replies to C)

Node IDs:  A="tweet_001", B="tweet_002", C="tweet_003",
           D="tweet_004", E="tweet_005"
cascade_id = "tweet_001" for all nodes.

Expected results:
    vertices: 5 rows, column "id" present
    edges:    4 rows (B→A, C→A, D→B, E→C  as dst→src)
              i.e. src=A,dst=B | src=A,dst=C | src=B,dst=D | src=C,dst=E
    singletons: 0 (this cascade has edges)
"""

import logging
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

from src.graph.phase04_graph.build_graph import (
    to_vertices,
    to_edges,
    build_full_graph,
    get_cascade_subgraph,
    flag_singletons,
)


# ---------------------------------------------------------------------------
# Pytest fixture: shared SparkSession (created once per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .appName("cascade2vec-phase04-tests")
        .master("local[1]")
        .config(
            "spark.jars.packages",
            "graphframes:graphframes:0.8.3-spark3.5-s_2.12",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("/tmp/cascade2vec_test_checkpoints")
    yield spark
    spark.stop()


# ---------------------------------------------------------------------------
# Helper: build the standard 5-node toy cascade DataFrame
# ---------------------------------------------------------------------------

def _toy_cascade(spark: SparkSession, cascade_id: str = "tweet_001"):
    """Return the unified-schema DataFrame for the toy cascade."""
    data = [
        # tweet_id,    user_id,   ts,  text,  parent_id,    cascade_id, event_id,     label
        ("tweet_001", "user_A",   0,   "root", None,         cascade_id, "test_event", "rumour"),
        ("tweet_002", "user_B",  10,   "re:A", "tweet_001",  cascade_id, "test_event", "rumour"),
        ("tweet_003", "user_C",  20,   "re:A", "tweet_001",  cascade_id, "test_event", "rumour"),
        ("tweet_004", "user_D",  30,   "re:B", "tweet_002",  cascade_id, "test_event", "rumour"),
        ("tweet_005", "user_E",  40,   "re:C", "tweet_003",  cascade_id, "test_event", "rumour"),
    ]
    return spark.createDataFrame(
        data,
        schema="tweet_id string, user_id string, timestamp long, text string, "
               "parent_id string, cascade_id string, event_id string, label string",
    )


def _two_cascade_df(spark: SparkSession):
    """Return a DataFrame with two cascades for isolation tests."""
    cascade_A = _toy_cascade(spark, cascade_id="tweet_001")
    cascade_B = spark.createDataFrame(
        [
            ("tweet_101", "user_X",  0, "root2", None,        "tweet_101", "test_event", "non-rumour"),
            ("tweet_102", "user_Y", 15, "re:X", "tweet_101",  "tweet_101", "test_event", "non-rumour"),
        ],
        schema="tweet_id string, user_id string, timestamp long, text string, "
               "parent_id string, cascade_id string, event_id string, label string",
    )
    return cascade_A.union(cascade_B)


# ---------------------------------------------------------------------------
# Tests: to_vertices
# ---------------------------------------------------------------------------

class TestToVertices:
    def test_renames_tweet_id_to_id(self, spark):
        """tweet_id column must be renamed to 'id' for GraphFrames."""
        df = _toy_cascade(spark)
        v = to_vertices(df)
        assert "id" in v.columns, "Expected 'id' column after rename."
        assert "tweet_id" not in v.columns, "'tweet_id' should not remain after rename."

    def test_preserves_all_other_columns(self, spark):
        """All original columns (except tweet_id → id) must be preserved."""
        df = _toy_cascade(spark)
        v = to_vertices(df)
        expected = {"id", "user_id", "timestamp", "text", "parent_id",
                    "cascade_id", "event_id", "label"}
        assert expected.issubset(set(v.columns))

    def test_row_count_unchanged(self, spark):
        """Vertex count must equal input row count."""
        df = _toy_cascade(spark)
        v = to_vertices(df)
        assert v.count() == df.count()


# ---------------------------------------------------------------------------
# Tests: to_edges
# ---------------------------------------------------------------------------

class TestToEdges:
    def test_drops_root_tweet(self, spark):
        """Root tweet (parent_id IS NULL) must not appear as an edge."""
        df = _toy_cascade(spark)
        e = to_edges(df)
        # Root is tweet_001 — it should not be a dst
        dst_ids = {row["dst"] for row in e.select("dst").collect()}
        assert "tweet_001" not in dst_ids, "Root tweet_001 should not be an edge dst."

    def test_correct_edge_count(self, spark):
        """5-node cascade with 1 root → 4 edges expected."""
        df = _toy_cascade(spark)
        e = to_edges(df)
        assert e.count() == 4, f"Expected 4 edges, got {e.count()}"

    def test_edge_columns_are_src_dst(self, spark):
        """Edge DataFrame must have 'src' and 'dst' columns."""
        df = _toy_cascade(spark)
        e = to_edges(df)
        assert "src" in e.columns
        assert "dst" in e.columns

    def test_drops_duplicate_edges(self, spark):
        """Duplicate (src, dst) pairs must be reduced to one."""
        df = _toy_cascade(spark)
        # Manually duplicate one row
        df_duped = df.union(df.filter(F.col("tweet_id") == "tweet_002"))
        e = to_edges(df_duped)
        # Should still be 4 edges, not 5
        assert e.count() == 4, f"Expected 4 edges after dedup, got {e.count()}"

    def test_drops_self_loops(self, spark, caplog):
        """Self-loops (tweet_id == parent_id) must be dropped with a WARNING."""
        data = [
            ("tweet_001", "user_A", 0,  "root", None,        "tweet_001", "evt", "rumour"),
            ("tweet_002", "user_B", 10, "self", "tweet_002", "tweet_001", "evt", "rumour"),  # self-loop
        ]
        df = spark.createDataFrame(
            data,
            schema="tweet_id string, user_id string, timestamp long, text string, "
                   "parent_id string, cascade_id string, event_id string, label string",
        )
        with caplog.at_level(logging.WARNING, logger="src.graph.phase04_graph.build_graph"):
            e = to_edges(df)
        assert e.count() == 0, "Self-loop should be dropped → 0 edges."
        assert any("self-loop" in msg.lower() for msg in caplog.messages), \
            "Expected a WARNING log mentioning 'self-loop'."

    def test_drops_orphaned_edges(self, spark, caplog):
        """
        Orphaned edge: parent_id exists but belongs to a different cascade.
        Should be dropped with a WARNING log.
        """
        data = [
            ("tweet_001", "user_A",  0, "root",  None,        "tweet_001", "evt", "rumour"),
            # parent_id points to tweet_999 which is in a different cascade
            ("tweet_002", "user_B", 10, "reply", "tweet_999", "tweet_001", "evt", "rumour"),
            # tweet_999 exists but in cascade_999
            ("tweet_999", "user_Z",  0, "other", None,        "tweet_999", "evt", "rumour"),
        ]
        df = spark.createDataFrame(
            data,
            schema="tweet_id string, user_id string, timestamp long, text string, "
                   "parent_id string, cascade_id string, event_id string, label string",
        )
        v = to_vertices(df)
        with caplog.at_level(logging.WARNING, logger="src.graph.phase04_graph.build_graph"):
            e = to_edges(df, vertices=v)
        # tweet_002's parent (tweet_999) is in a different cascade → orphan → dropped
        assert e.filter(F.col("cascade_id") == "tweet_001").count() == 0, \
            "Orphaned edge should be dropped."
        assert any("orphan" in msg.lower() for msg in caplog.messages), \
            "Expected a WARNING log mentioning 'orphan'."


# ---------------------------------------------------------------------------
# Tests: flag_singletons
# ---------------------------------------------------------------------------

class TestFlagSingletons:
    def test_normal_cascade_not_singleton(self, spark):
        """The 5-node toy cascade has edges → must NOT appear as singleton."""
        df = _toy_cascade(spark)
        v = to_vertices(df)
        e = to_edges(df)
        singletons = flag_singletons(v, e)
        cids = {row["cascade_id"] for row in singletons.collect()}
        assert "tweet_001" not in cids

    def test_singleton_detected(self, spark):
        """A cascade with only one tweet and no edges → must be flagged."""
        data = [
            ("tweet_solo", "user_S", 0, "alone", None, "tweet_solo", "evt", "rumour"),
        ]
        df = spark.createDataFrame(
            data,
            schema="tweet_id string, user_id string, timestamp long, text string, "
                   "parent_id string, cascade_id string, event_id string, label string",
        )
        v = to_vertices(df)
        e = to_edges(df)
        singletons = flag_singletons(v, e)
        cids = {row["cascade_id"] for row in singletons.collect()}
        assert "tweet_solo" in cids, "Singleton cascade must be flagged."


# ---------------------------------------------------------------------------
# Tests: build_full_graph + get_cascade_subgraph
# ---------------------------------------------------------------------------

class TestGetCascadeSubgraph:
    def test_subgraph_isolates_correct_cascade(self, spark):
        """Subgraph for cascade_id A must contain only A's nodes/edges."""
        df = _two_cascade_df(spark)
        v = to_vertices(df)
        e = to_edges(df, vertices=v)
        graph = build_full_graph(v, e)

        sub = get_cascade_subgraph(graph, "tweet_001")
        vertex_cids = {row["cascade_id"] for row in sub.vertices.collect()}
        edge_cids   = {row["cascade_id"] for row in sub.edges.collect()}

        assert vertex_cids == {"tweet_001"}, \
            f"Expected only 'tweet_001' in vertices, got {vertex_cids}"
        assert edge_cids <= {"tweet_001"}, \
            f"Expected only 'tweet_001' edges, got {edge_cids}"

    def test_subgraph_preserves_all_vertex_columns(self, spark):
        """All original vertex columns must survive get_cascade_subgraph."""
        df = _toy_cascade(spark)
        v = to_vertices(df)
        e = to_edges(df, vertices=v)
        graph = build_full_graph(v, e)

        sub = get_cascade_subgraph(graph, "tweet_001")
        expected_cols = {"id", "user_id", "timestamp", "text", "parent_id",
                         "cascade_id", "event_id", "label"}
        assert expected_cols.issubset(set(sub.vertices.columns)), \
            f"Missing columns: {expected_cols - set(sub.vertices.columns)}"

    def test_subgraph_has_correct_vertex_count(self, spark):
        """Subgraph for 5-node cascade must have 5 vertices."""
        df = _toy_cascade(spark)
        v = to_vertices(df)
        e = to_edges(df, vertices=v)
        graph = build_full_graph(v, e)

        sub = get_cascade_subgraph(graph, "tweet_001")
        assert sub.vertices.count() == 5

    def test_subgraph_has_correct_edge_count(self, spark):
        """Subgraph for 5-node cascade must have 4 edges."""
        df = _toy_cascade(spark)
        v = to_vertices(df)
        e = to_edges(df, vertices=v)
        graph = build_full_graph(v, e)

        sub = get_cascade_subgraph(graph, "tweet_001")
        assert sub.edges.count() == 4

    def test_subgraph_no_dangling_edges(self, spark):
        """Every edge src/dst in subgraph must exist in subgraph vertices."""
        df = _two_cascade_df(spark)
        v = to_vertices(df)
        e = to_edges(df, vertices=v)
        graph = build_full_graph(v, e)

        sub = get_cascade_subgraph(graph, "tweet_001")
        vertex_ids = {row["id"] for row in sub.vertices.collect()}
        for row in sub.edges.collect():
            assert row["src"] in vertex_ids, f"Dangling src: {row['src']}"
            assert row["dst"] in vertex_ids, f"Dangling dst: {row['dst']}"

    def test_singleton_subgraph_has_empty_edges(self, spark):
        """Singleton cascade subgraph must have 1 vertex and 0 edges."""
        data = [
            ("tweet_solo", "user_S", 0, "alone", None, "tweet_solo", "evt", "rumour"),
            ("tweet_001",  "user_A", 0, "root",  None, "tweet_001",  "evt", "rumour"),
            ("tweet_002",  "user_B", 5, "reply", "tweet_001", "tweet_001", "evt", "rumour"),
        ]
        df = spark.createDataFrame(
            data,
            schema="tweet_id string, user_id string, timestamp long, text string, "
                   "parent_id string, cascade_id string, event_id string, label string",
        )
        v = to_vertices(df)
        e = to_edges(df, vertices=v)
        graph = build_full_graph(v, e)

        sub = get_cascade_subgraph(graph, "tweet_solo")
        assert sub.vertices.count() == 1
        assert sub.edges.count() == 0
