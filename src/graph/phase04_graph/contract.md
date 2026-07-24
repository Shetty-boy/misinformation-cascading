# `get_cascade_subgraph` — Output Contract

> **For teammate building Phase 5 (time-based snapshot slicing).**
> This document describes the exact interface of `get_cascade_subgraph` so
> you can build against it without reading the implementation.

---

## Signature

```python
from src.graph.phase04_graph.build_graph import get_cascade_subgraph

subgraph = get_cascade_subgraph(graph, cascade_id)
```

| Parameter | Type | Description |
|---|---|---|
| `graph` | `GraphFrame` | The full combined GraphFrame from `build_full_graph()` |
| `cascade_id` | `str` | The cascade identifier string to extract |

---

## Return Value

A **GraphFrame** containing only the nodes and edges that belong to the
specified cascade.

### Vertex DataFrame (`subgraph.vertices`)

| Column | Type | Notes |
|---|---|---|
| `id` | str | Unique tweet/post ID (renamed from `tweet_id`) |
| `user_id` | str | Author ID |
| `timestamp` | long | Seconds since cascade root tweet (`t_root = 0`) |
| `text` | str | Tweet text content |
| `parent_id` | str | `None` for the root tweet; `id` of parent for replies |
| `cascade_id` | str | Always equal to the `cascade_id` argument you passed in |
| `event_id` | str | Event label (e.g. `"charliehebdo"`) |
| `label` | str | `"rumour"` or `"non-rumour"` |

### Edge DataFrame (`subgraph.edges`)

| Column | Type | Notes |
|---|---|---|
| `src` | str | Parent tweet `id` (the tweet being replied to) |
| `dst` | str | Child tweet `id` (the reply) |
| `cascade_id` | str | Always equal to the `cascade_id` argument you passed in |

---

## Guarantees

1. **All vertex columns are preserved.** No columns are dropped or renamed
   relative to the full graph's vertex schema.
2. **Edge endpoints are always in the vertex set.** No dangling edges —
   every `src` and `dst` in `subgraph.edges` has a corresponding row in
   `subgraph.vertices`.
3. **Singletons are valid.** If the cascade has only one post (no replies),
   `subgraph.vertices` will have one row and `subgraph.edges` will be empty.
   This is not an error.
4. **The returned GraphFrame is not cached.** Call `.cache()` on
   `subgraph.vertices` and `subgraph.edges` if you will reuse it across
   multiple Spark actions.

---

## Example Usage (for snapshot slicing)

```python
from src.graph.phase04_graph.loader import get_spark, load_unified
from src.graph.phase04_graph.build_graph import (
    to_vertices, to_edges, build_full_graph, get_cascade_subgraph
)
from pyspark.sql import functions as F

spark = get_spark()
df = load_unified(spark)
vertices = to_vertices(df)
edges = to_edges(df, vertices=vertices)
graph = build_full_graph(vertices, edges)

# Get the subgraph for one cascade
subgraph = get_cascade_subgraph(graph, "581287108607811584")

# Snapshot at T=1 hour (3600 seconds since root)
T = 3600
snapshot_vertices = subgraph.vertices.filter(F.col("timestamp") <= T)
snapshot_edges = subgraph.edges.join(
    snapshot_vertices.select("id"),
    subgraph.edges["src"] == snapshot_vertices["id"],
    how="inner"
).join(
    snapshot_vertices.select(F.col("id").alias("id2")),
    subgraph.edges["dst"] == F.col("id2"),
    how="inner"
).select("src", "dst", "cascade_id")
```

---

## Deviation Notes

- The unified dataset has `parent_id` as `None` (Python/Spark `null`) for
  cascade roots, NOT as `"None"` string or `""`. Downstream code must use
  `.isNull()` / `.isNotNull()` for filtering, not string equality checks.
- `cascade_id` equals the `tweet_id` of the root post in PHEME. It is a
  string (not an integer) throughout the pipeline.
