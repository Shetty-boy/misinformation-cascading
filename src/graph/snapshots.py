from graphframes import GraphFrame
from pyspark.sql import functions as F

def get_snapshot(graph: GraphFrame, t_minutes: float) -> GraphFrame:
    """
    Slices a cascade graph to include only vertices and edges that existed
    at or before time `t_minutes`.

    Args:
        graph: A GraphFrame of a single cascade (from get_cascade_subgraph).
        t_minutes: Time threshold in minutes.

    Returns:
        A new GraphFrame containing the temporal snapshot.
    """
    t_seconds = t_minutes * 60

    # Filter vertices where timestamp <= t_seconds
    snapshot_vertices = graph.vertices.filter(F.col("timestamp") <= t_seconds)

    # Filter edges to ensure both src and dst exist in the filtered vertices
    snapshot_edges = graph.edges.join(
        snapshot_vertices.select("id"),
        graph.edges["src"] == snapshot_vertices["id"],
        how="inner"
    ).drop("id").join(
        snapshot_vertices.select(F.col("id").alias("id2")),
        graph.edges["dst"] == F.col("id2"),
        how="inner"
    ).drop("id2")

    return GraphFrame(snapshot_vertices, snapshot_edges)
