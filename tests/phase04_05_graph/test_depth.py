import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from cascade2vec.phase04_05_graph.depth import compute_depths

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("pytest-depth") \
        .getOrCreate()
    yield spark
    spark.stop()

def create_df(spark, data_v, data_e):
    schema_v = StructType([
        StructField("id", StringType(), True),
        StructField("parent_id", StringType(), True),
        StructField("cascade_id", StringType(), True)
    ])
    schema_e = StructType([
        StructField("src", StringType(), True),
        StructField("dst", StringType(), True),
        StructField("cascade_id", StringType(), True)
    ])
    v_df = spark.createDataFrame(data_v, schema_v)
    e_df = spark.createDataFrame(data_e, schema_e) if data_e else spark.createDataFrame([], schema_e)
    return v_df, e_df

def test_single_node(spark):
    v = [("root", None, "c1")]
    e = []
    v_df, e_df = create_df(spark, v, e)
    
    depths = compute_depths(v_df, e_df).collect()
    
    assert len(depths) == 1
    assert depths[0].depth == 0
    assert depths[0].reachable == True

def test_linear_chain(spark):
    v = [("n1", None, "c1"), ("n2", "n1", "c1"), ("n3", "n2", "c1")]
    e = [("n1", "n2", "c1"), ("n2", "n3", "c1")]
    v_df, e_df = create_df(spark, v, e)
    
    depths = compute_depths(v_df, e_df).orderBy("tweet_id").collect()
    
    assert len(depths) == 3
    results = {r.tweet_id: (r.depth, r.reachable) for r in depths}
    assert results["n1"] == (0, True)
    assert results["n2"] == (1, True)
    assert results["n3"] == (2, True)

def test_balanced_tree(spark):
    v = [
        ("r", None, "c"),
        ("a", "r", "c"), ("b", "r", "c"),
        ("a1", "a", "c"), ("b1", "b", "c")
    ]
    e = [
        ("r", "a", "c"), ("r", "b", "c"),
        ("a", "a1", "c"), ("b", "b1", "c")
    ]
    v_df, e_df = create_df(spark, v, e)
    
    depths = compute_depths(v_df, e_df).collect()
    results = {r.tweet_id: r.depth for r in depths}
    assert results["r"] == 0
    assert results["a"] == 1
    assert results["b"] == 1
    assert results["a1"] == 2
    assert results["b1"] == 2

def test_disconnected_subtree(spark):
    # A graph with a root and a disconnected chunk that lacks a path from root
    # Even if "d1" has a parent "d2", neither can reach the root (no parent_id IS NULL).
    # Wait, the algorithm identifies roots as `parent_id IS NULL`. 
    # If the disconnected subtree has a node with `parent_id IS NULL`, it's a second root, which is valid.
    # To make it unreachable, it must have a parent, but the parent doesn't exist in the graph, or it forms a cycle.
    # Let's make "d2" have parent "missing". Since "missing" is not in `v_df` and has no `parent_id IS NULL`, it's unreachable.
    v = [
        ("r", None, "c"), ("n1", "r", "c"), 
        ("d1", "missing", "c"), ("d2", "d1", "c")
    ]
    e = [
        ("r", "n1", "c"),
        ("d1", "d2", "c")
    ]
    v_df, e_df = create_df(spark, v, e)
    
    depths = compute_depths(v_df, e_df).collect()
    results = {r.tweet_id: (r.depth, r.reachable) for r in depths}
    assert results["r"] == (0, True)
    assert results["n1"] == (1, True)
    assert results["d1"] == (None, False)
    assert results["d2"] == (None, False)

def test_deep_chain(spark):
    v = [("n0", None, "c1")]
    e = []
    for i in range(1, 25):
        v.append((f"n{i}", f"n{i-1}", "c1"))
        e.append((f"n{i-1}", f"n{i}", "c1"))
        
    v_df, e_df = create_df(spark, v, e)
    depths = compute_depths(v_df, e_df).collect()
    
    assert len(depths) == 25
    results = {r.tweet_id: r.depth for r in depths}
    for i in range(25):
        assert results[f"n{i}"] == i

def test_missing_parent(spark):
    # A node whose parent_id points to something not in vertices
    v = [("n1", None, "c1"), ("n2", "missing", "c1")]
    e = [] # no edges since parent is missing
    v_df, e_df = create_df(spark, v, e)
    
    depths = compute_depths(v_df, e_df).collect()
    results = {r.tweet_id: (r.depth, r.reachable) for r in depths}
    assert results["n1"] == (0, True)
    assert results["n2"] == (None, False)
