"""
loader.py — Phase 4 Graph Construction
=======================================
Loads the unified Phase 2 processed dataset into a PySpark DataFrame,
validating that the expected schema columns are present.

Usage:
    from src.graph.phase04_graph.loader import load_unified, get_spark
    spark = get_spark()
    df = load_unified(spark)
"""

import logging
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType,
)

logger = logging.getLogger(__name__)

# Exact column names from the unified schema (schema.py, Phase 2)
REQUIRED_COLUMNS = [
    "tweet_id",
    "user_id",
    "timestamp",
    "text",
    "parent_id",
    "cascade_id",
    "event_id",
    "label",
]

DEFAULT_DATA_PATH = "data/processed/phase2_ingestion/unified.parquet"


def get_spark(app_name: str = "cascade2vec-phase04") -> SparkSession:
    """
    Build or retrieve the active SparkSession with GraphFrames JAR attached.

    The GraphFrames JAR is resolved via the GRAPHFRAMES_JAR env variable
    (optional) or from the default Maven coordinates used during project
    setup. When running interactively, you can pre-load the JAR by launching
    PySpark with:
        pyspark --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12
    """
    jar_package = os.getenv(
        "GRAPHFRAMES_PACKAGE",
        "graphframes:graphframes:0.8.3-spark3.5-s_2.12",
    )

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", jar_package)
        # Checkpoint dir is required by GraphFrames' connectedComponents
        .config("spark.graphx.pregel.checkpointInterval", "2")
        .getOrCreate()
    )
    spark.sparkContext.setCheckpointDir("experiments/logs/04_graph/checkpoints")
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_unified(
    spark: SparkSession,
    path: str = DEFAULT_DATA_PATH,
) -> DataFrame:
    """
    Read the unified Phase 2 parquet into a Spark DataFrame.

    Validates that all REQUIRED_COLUMNS are present and casts types
    to their expected kinds. Returns the DataFrame unchanged if schema
    already matches.

    Raises
    ------
    ValueError
        If one or more required columns are missing from the parquet.
    """
    logger.info("[loader] Reading unified dataset from %s", path)
    df = spark.read.parquet(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"[loader] Unified parquet is missing required columns: {missing}. "
            f"Found: {df.columns}"
        )

    # Ensure timestamp is Long (int64); parent_id stays nullable string
    df = (
        df
        .withColumn("timestamp", F.col("timestamp").cast(LongType()))
        .withColumn("tweet_id",  F.col("tweet_id").cast(StringType()))
        .withColumn("parent_id", F.col("parent_id").cast(StringType()))
        .withColumn("cascade_id", F.col("cascade_id").cast(StringType()))
    )

    n = df.count()
    logger.info("[loader] Loaded %d rows from unified dataset.", n)
    return df
