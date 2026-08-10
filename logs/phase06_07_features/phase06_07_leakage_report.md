# Phase 6-7: Leakage Report

For every feature, this document states the formula, which information it uses,
and whether it is guaranteed safe before time t.

**Leakage definition:** A feature is leakage-safe if and only if it uses
EXCLUSIVELY information from tweets whose timestamp <= t_seconds from the
cascade root.

All snapshot operations are guarded by `assert_snapshot_is_clean()` from
`cascade2vec.phase02_ingestion.leakage_audit`, which raises AssertionError
immediately on any violation.

---

## Structural Features

| Feature | Formula | Uses only info ≤ t? | Leakage-Safe |
|---|---|---|---|
| `node_count` | count(vertices in snapshot) | ✅ Snapshot already filtered to ≤ t | ✅ |
| `edge_count` | count(edges in snapshot) | ✅ Both src and dst must be ≤ t | ✅ |
| `max_depth` | max(BFS depth from root) over snapshot | ✅ BFS runs on snapshot graph only | ✅ |
| `avg_depth` | mean(BFS depth from root) over reachable nodes | ✅ Same BFS, same snapshot | ✅ |
| `leaf_count` | count(nodes with out-degree=0 in snapshot) | ✅ Out-degree computed on snapshot edges only | ✅ |
| `leaf_ratio` | leaf_count / node_count | ✅ Both computed on snapshot | ✅ |
| `branching_factor` | edge_count / (node_count - leaf_count) | ✅ All terms from snapshot | ✅ |
| `root_degree` | out-degree of root in snapshot | ✅ Root is at t=0; children counted up to t | ✅ |
| `reachable_ratio` | reachable_count / node_count | ✅ **CRITICAL**: denominator is snapshot node_count, NOT final cascade size — final size is future information | ✅ |
| `is_connected` | reachable_count == node_count | ✅ Per (cascade_id, t) snapshot, not globally | ✅ |

## Temporal Features

| Feature | Formula | Uses only info ≤ t? | Leakage-Safe |
|---|---|---|---|
| `cascade_age` | max(timestamp) over snapshot vertices | ✅ All timestamps ≤ t by construction | ✅ |
| `tweets_per_minute` | node_count / (cascade_age / 60) | ✅ Both terms from snapshot | ✅ |
| `growth_velocity` | node_count / cascade_age | ✅ Same as above in seconds | ✅ |
| `mean_interarrival` | mean(Δ between consecutive tweet timestamps ≤ t) | ✅ All timestamps ≤ t | ✅ |
| `std_interarrival` | std(Δ between consecutive tweet timestamps ≤ t) | ✅ All timestamps ≤ t; 0.0 if n<2 | ✅ |
| `burstiness` | (std - mean) / (std + mean) of interarrivals | ✅ Derived from std/mean above; 0.0 if n<2 | ✅ |

## Phase 6B: Velocity Features

| Feature | Formula | Uses only info ≤ t? | Leakage-Safe |
|---|---|---|---|
| `depth_velocity` | (max_depth(t) - max_depth(t-Δ)) / Δ | ✅ Only t and t-Δ snapshots; 0.0 if t-Δ < 0 | ✅ |
| `breadth_velocity` | (node_count(t) - node_count(t-Δ)) / Δ | ✅ Same as above | ✅ |
| `branching_velocity` | (branching_factor(t) - branching_factor(t-Δ)) / Δ | ✅ Same as above | ✅ |

## Rejected Features (NOT implemented)

| Feature | Reason rejected |
|---|---|
| `final_cascade_size` | Uses total nodes regardless of t — future information |
| `global_cascade_lifespan` | Uses final tweet timestamp — future information |
| `final_pagerank` | Requires full cascade graph — future information |
