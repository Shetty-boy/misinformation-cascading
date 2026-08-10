# Phase 4-5: Graph Construction & Depth Computation

**Status:** ✅ Complete
**Log:** [`../../logs/phase04_05_graph/graph_stats.md`](../../logs/phase04_05_graph/graph_stats.md)

### What Was Done
- Built propagation tree graph from `parent_id` structure
- Implemented BFS depth computation in PySpark (`compute_depths`) and Pandas (`compute_depths_pandas`)
- Verified both implementations produce **exactly identical results** on 5 regression cascades

### Implementation
- `src/cascade2vec/phase04_05_graph/build_graph.py` — graph construction
- `src/cascade2vec/phase04_05_graph/depth.py` — PySpark BFS
- `src/cascade2vec/phase04_05_graph/depth_pandas.py` — Pandas BFS (regression reference)

### Dataset Graph Statistics
| Metric | Value |
|---|---|
| Total cascades | 5,802 |
| Singleton cascades | 358 |
| Fully connected cascades | 5,193 |
| Mean node count | 17.7 |
| Mean edge count | 16.4 |
| Mean max depth | 3.4 |

### Key Decisions
- **DAG guarantee:** Temporal ordering (reply always chronologically after parent) + in-degree ≤ 1 → forest structure, no cycles possible. No cycle-detection needed in BFS.
- 1,460 orphaned edges dropped (parent_id not found in same cascade's vertex set)
- `persist()` used in BFS loop instead of `localCheckpoint()` to avoid Py4J re-entry in test environments

### Bugs Fixed
- BFS was originally returning flat depth=0.0 for all nodes — fixed by correcting frontier propagation logic
- Comment in `depth.py` now correctly states the DAG guarantee as "temporal ordering + in-degree ≤ 1 → forest"
