# Phase 6-7: Feature Dictionary

Complete specification for every feature in the Phase 6 feature matrix.

**Source module:** `src/cascade2vec/phase06_07_features/engineering.py`

---

## Phase 6A: Structural Features

### `node_count`
- **Definition:** Number of vertices in the temporal snapshot
- **Formula:** `count(v for v in vertices where v.timestamp <= t)`
- **Units:** count (dimensionless)
- **Leakage-safe:** ✅
- **Required inputs:** snapshot vertices
- **Source module:** `engineering._compute_snapshot_structural()`
- **Depends on:** `get_snapshot()` from `phase04_05_graph.snapshots`
- **Stability sentinel:** N/A — always well-defined

---

### `edge_count`
- **Definition:** Number of edges in the temporal snapshot
- **Formula:** `count(e for e in edges where src.timestamp <= t and dst.timestamp <= t)`
- **Units:** count (dimensionless)
- **Leakage-safe:** ✅
- **Required inputs:** snapshot edges
- **Source module:** `engineering._compute_snapshot_structural()`
- **Stability sentinel:** N/A — always well-defined

---

### `max_depth`
- **Definition:** Maximum BFS depth from the cascade root over reachable nodes
- **Formula:** `max(depth_df["depth"][depth_df["reachable"]])`
- **Units:** hops (dimensionless)
- **Leakage-safe:** ✅
- **Required inputs:** snapshot vertices + edges → `compute_depths()`
- **Source module:** `engineering._compute_snapshot_structural()`
- **Depends on:** `compute_depths()` from `phase04_05_graph.depth` — **DO NOT REIMPLEMENT**
- **Stability sentinel:** if no reachable nodes, = 0.0

---

### `avg_depth`
- **Definition:** Mean BFS depth from the cascade root over reachable nodes
- **Formula:** `mean(depth_df["depth"][depth_df["reachable"]])`
- **Units:** hops (dimensionless)
- **Leakage-safe:** ✅
- **Required inputs:** same BFS result as `max_depth`
- **Source module:** `engineering._compute_snapshot_structural()`
- **Depends on:** `compute_depths()` from `phase04_05_graph.depth`
- **Stability sentinel:** if no reachable nodes, = 0.0

---

### `leaf_count`
- **Definition:** Number of nodes that have no outgoing edges in the snapshot
- **Formula:** `count(v for v in snapshot_vertices where v.id not in edges.src)`
- **Units:** count (dimensionless)
- **Leakage-safe:** ✅
- **Required inputs:** snapshot vertices + edges
- **Source module:** `engineering._compute_snapshot_structural()`
- **Stability sentinel:** if edge_count == 0, leaf_count = node_count (all nodes are leaves)

---

### `leaf_ratio`
- **Definition:** Fraction of nodes that are leaves in the snapshot
- **Formula:** `leaf_count / node_count`
- **Units:** fraction [0, 1]
- **Leakage-safe:** ✅
- **Required inputs:** `leaf_count`, `node_count`
- **Source module:** `engineering._compute_snapshot_structural()`
- **Stability sentinel:** if node_count == 0, = 0.0

---

### `branching_factor`
- **Definition:** Average out-degree of non-leaf (internal) nodes in the snapshot
- **Formula:** `edge_count / (node_count - leaf_count)`
- **Units:** edges per internal node (dimensionless)
- **Leakage-safe:** ✅
- **Required inputs:** `edge_count`, `node_count`, `leaf_count`
- **Source module:** `engineering._compute_snapshot_structural()`
- **Stability sentinel:** **if there are no internal nodes (i.e. snapshot is a singleton or all leaves), = 0.0**
  This is the documented convention. We treat a tree with 0 internal nodes as having 0
  branching, rather than undefined/NaN. This is important: the max_depth=0.0 bug
  was allowed to hide as "real" data for weeks because 0.0 is a valid-looking value.
  Users of this feature must interpret branching_factor=0.0 with node_count=1 as
  "singleton, no internal structure" rather than "zero branches per internal node."

---

### `root_degree`
- **Definition:** Out-degree of the cascade root node at time t
- **Formula:** `count(e for e in snapshot_edges where e.src == root_id)`
- **Units:** count (dimensionless)
- **Leakage-safe:** ✅
- **Required inputs:** snapshot edges, root node id
- **Source module:** `engineering._compute_snapshot_structural()`
- **Stability sentinel:** if root not found in snapshot (disconnected/missing), = 0

---

### `reachable_ratio`
- **Definition:** Fraction of snapshot nodes that are reachable from the root via BFS
- **Formula:** `reachable_count / node_count`
  **⚠️ CRITICAL**: denominator is snapshot node_count (nodes at time t), NOT final
  cascade size. Using final cascade size would leak future information.
- **Units:** fraction [0, 1]
- **Leakage-safe:** ✅
- **Required inputs:** BFS result from `compute_depths()`
- **Source module:** `engineering._compute_snapshot_structural()`
- **Stability sentinel:** if node_count == 0, = 0.0

---

### `is_connected`
- **Definition:** Whether all snapshot nodes are reachable from the cascade root at time t
- **Formula:** `reachable_count == node_count`
- **Units:** boolean
- **Leakage-safe:** ✅
- **Required inputs:** BFS result from `compute_depths()`
- **Source module:** `engineering._compute_snapshot_structural()`
- **Note:** Computed per (cascade_id, t) snapshot pair — NOT once per cascade globally.
  A cascade may be disconnected at t=15min and become connected at t=40min
  when a bridging reply arrives.
- **Stability sentinel:** N/A — always well-defined (True or False)

---

## Phase 6A: Temporal Features

### `cascade_age`
- **Definition:** Time elapsed from root tweet to last observed tweet at or before t
- **Formula:** `max(v.timestamp for v in snapshot_vertices)` (root is at t=0)
- **Units:** seconds
- **Leakage-safe:** ✅
- **Source module:** `engineering._compute_snapshot_temporal()`
- **Stability sentinel:** if no nodes, = 0.0

---

### `tweets_per_minute`
- **Definition:** Rate of tweets arriving up to observation time t
- **Formula:** `node_count / (cascade_age / 60)`
- **Units:** tweets per minute
- **Leakage-safe:** ✅
- **Source module:** `engineering._compute_snapshot_temporal()`
- **Stability sentinel:** if cascade_age == 0, = 0.0

---

### `growth_velocity`
- **Definition:** Rate of tweets per second of cascade age
- **Formula:** `node_count / cascade_age`
- **Units:** tweets per second
- **Leakage-safe:** ✅
- **Source module:** `engineering._compute_snapshot_temporal()`
- **Stability sentinel:** if cascade_age == 0, = 0.0

---

### `mean_interarrival`
- **Definition:** Mean time gap between consecutive tweets at or before t
- **Formula:** `mean([ts[i+1] - ts[i] for i in range(n-1)])` where ts is sorted timestamps
- **Units:** seconds
- **Leakage-safe:** ✅
- **Source module:** `engineering._compute_snapshot_temporal()`
- **Stability sentinel:** if n < 2 (fewer than 2 tweets), = 0.0; also logged at INFO

---

### `std_interarrival`
- **Definition:** Standard deviation of interarrival times at or before t
- **Formula:** `std([ts[i+1] - ts[i] for i in range(n-1)])`
- **Units:** seconds
- **Leakage-safe:** ✅
- **Source module:** `engineering._compute_snapshot_temporal()`
- **Stability sentinel:** **if n < 2, = 0.0** (logged at INFO as "Insufficient temporal observations")

---

### `burstiness`
- **Definition:** Burstiness coefficient of tweet interarrival times
- **Formula:** `(std_interarrival - mean_interarrival) / (std_interarrival + mean_interarrival)`
  Range: [-1, 1]. Negative = regular/periodic. Positive = bursty.
- **Units:** dimensionless ratio
- **Leakage-safe:** ✅
- **Source module:** `engineering._compute_snapshot_temporal()`
- **Stability sentinel:** **if n < 2 or (std + mean) == 0, = 0.0** (logged at INFO)

---

## Phase 6B: Velocity Features

All velocity features use default `delta_t = 5 minutes`.

### `depth_velocity`
- **Definition:** Rate of change of max_depth between t-Δ and t
- **Formula:** `(max_depth(t) - max_depth(t-Δ)) / Δ`  (Δ in seconds)
- **Units:** hops per second
- **Leakage-safe:** ✅ Only t and t-Δ snapshots used
- **Source module:** `engineering.compute_features()`
- **Stability sentinel:** if t <= delta_t (no prior snapshot), = 0.0

---

### `breadth_velocity`
- **Definition:** Rate of change of node count between t-Δ and t
- **Formula:** `(node_count(t) - node_count(t-Δ)) / Δ`
- **Units:** nodes per second
- **Leakage-safe:** ✅
- **Source module:** `engineering.compute_features()`
- **Stability sentinel:** if t <= delta_t, = 0.0

---

### `branching_velocity`
- **Definition:** Rate of change of branching factor between t-Δ and t
- **Formula:** `(branching_factor(t) - branching_factor(t-Δ)) / Δ`
- **Units:** edges per internal node per second
- **Leakage-safe:** ✅
- **Source module:** `engineering.compute_features()`
- **Stability sentinel:** if t <= delta_t, = 0.0
