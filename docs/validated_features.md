# Validated Features for CASCADE2VEC

This document serves as the formal specification for the features that will be fed into the CASCADE2VEC model. Every feature has been audited to ensure it does not leak future information when performing early detection (where cascades are observed only up to time $T$).

## Feature Leakage Table

| Feature | Uses Future Information? | Safe Before Time $T$? | Keep? |
|---------|---------------------------|---------------------|-------|
| Cascade depth at time $T$ | No | Yes | ✅ |
| Final cascade size | Yes | No | ❌ |
| Final PageRank | Depends | Usually No | ❌ unless recomputed on truncated graph |
| Mean propagation speed (up to $T$) | No | Yes | ✅ |
| Node degree at time $T$ | No | Yes | ✅ |
| Average branching factor at time $T$ | No | Yes | ✅ |
| Sentiment of root tweet | No | Yes | ✅ |
| Global cascade lifespan | Yes | No | ❌ |

## Final Validated Feature List

Based on the leakage audit, the following features are approved for implementation in Weeks 4-7:

### 1. Structural Graph Features
These features capture the topology of the cascade up to the observation window $T$:
- **In-Degree/Out-Degree**: Computed on the graph truncated at time $T$.
- **Cascade Depth**: The maximum path length from the root tweet within the observation window.
- **Branching Factor**: The average out-degree of all non-leaf nodes present before $T$.
- **Graph Density**: The ratio of actual edges to possible edges in the observed cascade.

### 2. Temporal Features
These features capture the speed and rhythm of the cascade's growth before time $T$:
- **Inter-Arrival Times**: The average time difference between consecutive retweets/replies within the observation window.
- **Propagation Speed**: The number of nodes in the cascade divided by the time elapsed between the root tweet and the last observed tweet (capped at $T$).

### 3. Content Features (Optional)
If text embeddings are used alongside the graph structure:
- **Root Tweet Embedding**: Text features from the root tweet are always safe.
- **Reply Sentiment**: Aggregate sentiment of all replies that occurred strictly before time $T$.
