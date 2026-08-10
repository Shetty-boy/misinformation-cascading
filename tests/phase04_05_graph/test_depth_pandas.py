import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from cascade2vec.phase04_05_graph.depth import compute_depths
from cascade2vec.phase04_05_graph.depth_pandas import compute_depths_pandas
from cascade2vec.phase04_05_graph.loader import get_spark, load_unified
from cascade2vec.phase04_05_graph.build_graph import to_vertices, to_edges

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("pytest-depth-pandas") \
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

def test_disconnected_subtree_pandas():
    # Pure Pandas test for disconnected subtree (same as in test_depth.py)
    v_data = [
        {"id": "r", "parent_id": None, "cascade_id": "c"},
        {"id": "n1", "parent_id": "r", "cascade_id": "c"},
        {"id": "d1", "parent_id": "missing", "cascade_id": "c"},
        {"id": "d2", "parent_id": "d1", "cascade_id": "c"},
    ]
    e_data = [
        {"src": "r", "dst": "n1", "cascade_id": "c"},
        {"src": "d1", "dst": "d2", "cascade_id": "c"},
    ]
    v_df = pd.DataFrame(v_data)
    e_df = pd.DataFrame(e_data)
    
    depths = compute_depths_pandas(v_df, e_df)
    
    assert len(depths) == 4
    results = depths.set_index('tweet_id').to_dict('index')
    
    assert results["r"]["depth"] == 0.0
    assert results["r"]["reachable"] == True
    assert results["n1"]["depth"] == 1.0
    assert results["n1"]["reachable"] == True
    
    assert pd.isna(results["d1"]["depth"])
    assert results["d1"]["reachable"] == False
    assert pd.isna(results["d2"]["depth"])
    assert results["d2"]["reachable"] == False

def test_regression_spark_vs_pandas(spark):
    """
    Permanent regression test: Ensure compute_depths (PySpark) and compute_depths_pandas
    produce exactly identical results for 5 known real cascades, plus 1 mock disconnected cascade.
    """
    # 1. Load real data
    df = load_unified(spark)
    vertices = to_vertices(df)
    edges = to_edges(df, vertices=vertices)
    
    # 5 test cascades from validation.py
    test_cascade_ids = [
        "552806610490646528",
        "544332847050670080",
        "552797154692300800",
        "498433651835940865",
        "499689349420568577"
    ]
    
    # Filter vertices and edges for test cascades
    v_real = vertices.filter(F.col("cascade_id").isin(test_cascade_ids))
    e_real = edges.filter(F.col("cascade_id").isin(test_cascade_ids))
    
    # 2. Add disconnected mock cascade
    v_mock = [
        ("root_disc", None, "disc_c"),
        ("A", "root_disc", "disc_c"),
        ("B", "missing", "disc_c"),
        ("C", "B", "disc_c")
    ]
    e_mock = [
        ("root_disc", "A", "disc_c"),
        ("B", "C", "disc_c")
    ]
    v_mock_df, e_mock_df = create_df(spark, v_mock, e_mock)
    
    # Ensure schemas match before union
    v_mock_df = v_mock_df.withColumn("timestamp", F.lit(0).cast("long")) \
                         .withColumn("label", F.lit("rumour").cast("string")) \
                         .withColumn("user_id", F.lit("mock_user").cast("string")) \
                         .withColumn("text", F.lit("mock_text").cast("string")) \
                         .withColumn("event_id", F.lit("mock_event").cast("string")) \
                         .select("id", "cascade_id", "parent_id", "timestamp", "label", "user_id", "text", "event_id")
    
    v_combined = v_real.unionByName(v_mock_df, allowMissingColumns=True)
    e_combined = e_real.unionByName(e_mock_df, allowMissingColumns=True)
    
    # 3. Run PySpark BFS
    spark_depths = compute_depths(v_combined, e_combined).toPandas()
    spark_depths = spark_depths.sort_values(["cascade_id", "tweet_id"]).reset_index(drop=True)
    
    # 4. Run Pandas BFS
    v_pd = v_combined.toPandas()
    e_pd = e_combined.toPandas()
    pandas_depths = compute_depths_pandas(v_pd, e_pd)
    pandas_depths = pandas_depths.sort_values(["cascade_id", "tweet_id"]).reset_index(drop=True)
    
    # 5. Assert Exact Match
    assert len(spark_depths) == len(pandas_depths)
    
    # PySpark returns nullable int for depth, Pandas might return float64 for depth because of NaNs.
    # Convert PySpark depth to float64 to match Pandas types for comparison, or compare values.
    spark_depths['depth'] = spark_depths['depth'].astype(float)
    pandas_depths['depth'] = pandas_depths['depth'].astype(float)
    
    pd.testing.assert_series_equal(spark_depths['tweet_id'], pandas_depths['tweet_id'], check_names=False)
    pd.testing.assert_series_equal(spark_depths['cascade_id'], pandas_depths['cascade_id'], check_names=False)
    pd.testing.assert_series_equal(spark_depths['reachable'], pandas_depths['reachable'], check_names=False)
    pd.testing.assert_series_equal(spark_depths['depth'], pandas_depths['depth'], check_names=False)
