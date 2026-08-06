# Distributed Graph Traversal Performance Audit

## 1. Benchmarks
- **Dataset Size**: 102440 vertices
- **Runtime (Mean)**: 23.86 seconds
- **Runtime (StdDev)**: 4.13 seconds
- **Max Iterations**: 47
- **Largest Frontier**: 52709 nodes
- **Average Frontier**: 1996.9 nodes
- **Total Reachable Visited**: 99657 nodes

## 2. Spark Execution Plan Audit
- **GraphFrame.shortestPaths() usage**: None (Verified)
- **Recursive SQL usage**: None (Verified)
- **Python recursion**: None (Verified)
- **collect() inside loop**: None (Verified, loop relies on .count() and .localCheckpoint())
- **driver-side graph traversal**: None (Verified)

### Logical Plan Summary
```text
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- Project [tweet_id#0, cascade_id#5, depth#13150, isnotnull(depth#13150) AS reachable#13154]
   +- SortMergeJoin [tweet_id#0, cascade_id#5], [tweet_id#13148, cascade_id#13149], LeftOuter
      :- Sort [tweet_id#0 ASC NULLS FIRST, cascade_id#5 ASC NULLS FIRST], false, 0
      :  +- Exchange hashpartitioning(tweet_id#0, cascade_id#5, 50), ENSURE_REQUIREMENTS, [plan_id=94176]
      :     +- FileScan parquet [tweet_id#0,cascade_id#5] Batched: true, DataFilters: [], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:/home/dr_shetty/misinformation-cascading/data/processed/phase02_i..., PartitionFilters: [], PushedFilters: [], ReadSchema: struct<tweet_id:string,cascade_id:string>
      +- Sort [tweet_id#13148 ASC NULLS FIRST, cascade_id#13149 ASC NULLS FIRST], false, 0
         +- Exchange hashpartitioning(tweet_id#13148, cascade_id#13149, 50), ENSURE_REQUIREMENTS, [plan_id=94177]
            +- Filter (isnotnull(tweet_id#13148) AND isnotnull(cascade_id#13149))
               +- Scan ExistingRDD[tweet_id#13148,cascade_id#13149,depth#13150]


```
