import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from cascade2vec.phase04_05_graph.depth import compute_depths
from cascade2vec.phase04_05_graph.depth_pandas import compute_depths_pandas
from cascade2vec.phase04_05_graph.loader import load_unified
from cascade2vec.phase04_05_graph.build_graph import to_vertices, to_edges

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

def main():
    print("Starting Spark...")
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("standalone-regression") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    print("Loading unified data...")
    df = load_unified(spark)
    vertices = to_vertices(df)
    edges = to_edges(df, vertices=vertices)
    
    test_cascade_ids = [
        "552806610490646528",
        "544332847050670080",
        "552797154692300800",
        "498433651835940865",
        "499689349420568577"
    ]
    
    v_real = vertices.filter(F.col("cascade_id").isin(test_cascade_ids))
    e_real = edges.filter(F.col("cascade_id").isin(test_cascade_ids))
    
    # Add disconnected mock cascade
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
    
    v_mock_df = v_mock_df.withColumn("timestamp", F.lit(0).cast("long")) \
                         .withColumn("label", F.lit("rumour").cast("string")) \
                         .withColumn("user_id", F.lit("mock_user").cast("string")) \
                         .withColumn("text", F.lit("mock_text").cast("string")) \
                         .withColumn("event_id", F.lit("mock_event").cast("string")) \
                         .select("id", "cascade_id", "parent_id", "timestamp", "label", "user_id", "text", "event_id")
    
    v_combined = v_real.unionByName(v_mock_df, allowMissingColumns=True).persist()
    e_combined = e_real.unionByName(e_mock_df, allowMissingColumns=True).persist()
    
    print(f"Total vertices to process: {v_combined.count()}")
    print(f"Total edges to process: {e_combined.count()}")
    
    print("Running PySpark compute_depths...")
    spark_depths_df = compute_depths(v_combined, e_combined)
    spark_depths = spark_depths_df.toPandas()
    spark_depths = spark_depths.sort_values(["cascade_id", "tweet_id"]).reset_index(drop=True)
    
    print("Running Pandas compute_depths_pandas...")
    v_pd = v_combined.toPandas()
    e_pd = e_combined.toPandas()
    pandas_depths = compute_depths_pandas(v_pd, e_pd)
    pandas_depths = pandas_depths.sort_values(["cascade_id", "tweet_id"]).reset_index(drop=True)
    
    print("Comparing results...")
    if len(spark_depths) != len(pandas_depths):
        print(f"Length mismatch: {len(spark_depths)} vs {len(pandas_depths)}")
        sys.exit(1)
        
    spark_depths['depth'] = spark_depths['depth'].astype(float)
    pandas_depths['depth'] = pandas_depths['depth'].astype(float)
    
    try:
        pd.testing.assert_series_equal(spark_depths['tweet_id'], pandas_depths['tweet_id'], check_names=False)
        pd.testing.assert_series_equal(spark_depths['cascade_id'], pandas_depths['cascade_id'], check_names=False)
        pd.testing.assert_series_equal(spark_depths['reachable'], pandas_depths['reachable'], check_names=False)
        pd.testing.assert_series_equal(spark_depths['depth'], pandas_depths['depth'], check_names=False)
        print("SUCCESS! PySpark and Pandas BFS produce exactly identical results!")
    except AssertionError as e:
        print("MATCH FAILED!")
        print(e)
        sys.exit(1)
        
    spark.stop()

if __name__ == "__main__":
    main()
