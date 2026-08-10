# Phase 6-7: Feature Approval Matrix

Source of truth for which features are approved for implementation.
No feature may be implemented in `engineering.py` unless Status = **Approved**.

Reference: [validated_features.md](file:///home/omen/projects/misinformation-cascading/docs/phase03_eda_leakage/phase03_validated_features.md)

| Feature | Previously Approved (Phase 3 EDA) | Newly Proposed (Phase 6) | Leakage Reviewed | Status |
|---|---|---|---|---|
| **STRUCTURAL** | | | | |
| `node_count` | ✅ (cascade depth proxy) | — | ✅ Uses only vertices at t | **Approved** |
| `edge_count` | ✅ (graph density proxy) | — | ✅ Uses only edges at t | **Approved** |
| `max_depth` | ✅ (cascade depth at T) | — | ✅ Computed via BFS on snapshot | **Approved** |
| `avg_depth` | — | ✅ | ✅ Derived from same BFS as max_depth | **Approved** |
| `leaf_count` | — | ✅ | ✅ Count of nodes with out-degree=0 at t | **Approved** |
| `leaf_ratio` | — | ✅ | ✅ leaf_count / node_count, both at t | **Approved** |
| `branching_factor` | ✅ (branching factor at T) | — | ✅ avg out-degree of non-leaf nodes at t | **Approved** |
| `root_degree` | ✅ (in/out-degree at T) | — | ✅ Out-degree of root at t | **Approved** |
| `reachable_ratio` | — | ✅ | ✅ Denominator is snapshot node_count, NOT final cascade size | **Approved** |
| `is_connected` | — | ✅ | ✅ Computed per (cascade_id, t), not globally | **Approved** |
| **TEMPORAL** | | | | |
| `tweets_per_minute` | ✅ (propagation speed) | — | ✅ node_count / cascade_age, both at t | **Approved** |
| `growth_velocity` | — | ✅ | ✅ Uses only timestamps <= t | **Approved** |
| `mean_interarrival` | ✅ (inter-arrival times) | — | ✅ Mean of deltas between events at t | **Approved** |
| `std_interarrival` | — | ✅ | ✅ Std of deltas; set=0.0 if n<2 | **Approved** |
| `burstiness` | — | ✅ | ✅ (std-mean)/(std+mean); set=0.0 if n<2 | **Approved** |
| `cascade_age` | — | ✅ | ✅ max(timestamp) - 0 (root is at t=0) | **Approved** |
| **HYBRID (Phase 6B)** | | | | |
| `depth_velocity` | — | ✅ | ✅ (depth(t) - depth(t-Δ)) / Δ | **Approved** |
| `breadth_velocity` | — | ✅ | ✅ (node_count(t) - node_count(t-Δ)) / Δ | **Approved** |
| `branching_velocity` | — | ✅ | ✅ (branching(t) - branching(t-Δ)) / Δ | **Approved** |
| **DEFERRED / NOT APPROVED** | | | | |
| `final_cascade_size` | ❌ (uses future info) | — | ❌ Leaks final cascade size | **Rejected** |
| `global_cascade_lifespan` | ❌ (uses future info) | — | ❌ Leaks total duration | **Rejected** |
| `final_pagerank` | ❌ (uses future info) | — | ❌ Requires full cascade | **Rejected** |
| `root_tweet_embedding` | ✅ (optional) | — | ✅ Root is always at t=0 | **Deferred to Phase 6C** |
| `reply_sentiment` | ✅ (optional) | — | ✅ If capped at t | **Deferred to Phase 6C** |
