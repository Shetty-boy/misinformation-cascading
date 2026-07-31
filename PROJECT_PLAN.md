# CASCADE2VEC — End-to-End Project Plan (Solo + 1 Teammate, Multi-Node)

This is the single doc to work through top to bottom. It merges three things that were
previously scattered across conversations: (1) what's already built, (2) the research
pipeline/week-by-week plan, and (3) the distributed infrastructure track needed to make
this a genuine multi-node big-data project rather than a single-laptop demo.

---

## 0. Current Status (as of now)

Already done, working locally on `local[*]` Spark:

| Phase | Status | Where |
|---|---|---|
| 1. Design doc + data acquisition | ✅ Done | `docs/design_doc.md` |
| 2. Ingestion + schema unification | ✅ Done | `src/features/phase02_ingestion/` |
| 3. EDA + leakage audit | ✅ Done | `notebooks/03_eda_and_leakage/` |
| 4. Cascade graph construction (GraphFrames) | ✅ Done | `src/graph/phase04_graph/` |
| 5. Temporal snapshot slicing | ✅ Done | `src/graph/snapshots.py` |
| 6+. Feature engineering, baselines, embedding, adaptive stopping | ⏳ Not started | — |

Known issues to fix before scaling out:
- `GraphFrames` version mismatch (`stats.py` already documents `connectedComponents()`
  throwing `Py4JException` — pip package `0.9.0` vs Maven JAR `0.8.3-spark3.5-s_2.12` don't match).
- `get_spark()` never sets `.master(...)` → silently always runs `local[*]` regardless of
  what cluster exists.
- All output/checkpoint paths are local filesystem paths, not cluster-reachable.

**Nothing below assumes you throw away this work.** The infra track runs underneath it —
same code, same logic, just made configurable so it can point at a real cluster instead
of always defaulting to one laptop.

---

## 1. Two Parallel Tracks From Here

Run these as two tracks that meet back up: **Track A (Research/ML)** is the original
19-week plan. **Track B (Infra)** is new — it needs to land *before* Week 16-17
(scalability benchmarking) but the earlier it's done, the more of Track A actually
runs distributed instead of on one laptop, which strengthens every later result.

Recommended order: do **Sprint 0 (Infra bootstrap)** now, before touching Week 6+ of
Track A, since your data ingestion/graph/snapshot code (Phases 1-5) already exists and
just needs to be pointed at the cluster once it's up — cheap to validate now, expensive
to retrofit later.

---

## 2. Track B — Infrastructure (Sprint 0, do this first)

### Goal
Two real laptops (you + teammate) running as a genuine Spark + HDFS cluster, built
modularly enough that adding node #3, #4, etc. later is "add one line to a file and
run one script" — not a re-architecture.

### 2.1 Networking
- Install Tailscale on both laptops, join same tailnet.
- **Enable MagicDNS** in the Tailscale admin console — reference machines by name
  (`laptop-a`, `laptop-b`) everywhere from here on, never by raw IP. This is what
  makes the setup survive IP changes and scale cleanly to more nodes.
- Verify: `ping laptop-b` from `laptop-a` and vice versa, before touching Spark/Hadoop.

### 2.2 Canonical node list (single source of truth)
- `$HADOOP_HOME/etc/hadoop/workers` is the one file that lists every node. Don't
  maintain a second list anywhere else — everything else (Spark worker launch,
  any custom scripts) reads from this file.

### 2.3 Spark cluster
- Set `SPARK_LOCAL_IP=<this machine's tailscale name/IP>` on every machine, in
  `~/.bashrc`, every session.
- Master: `start-master.sh` on `laptop-a`.
- Worker: `start-worker.sh spark://laptop-a:7077` on `laptop-b`.
- Confirm both hosts show up under Executors in the Spark UI (`http://laptop-a:8080`)
  before trusting anything else.

### 2.4 HDFS (distributed storage)
- Install matching Hadoop version on both machines.
- `core-site.xml`: `fs.defaultFS = hdfs://laptop-a:9000`
- `hdfs-site.xml`: `dfs.replication = 2` (documented tradeoff below)
- NameNode on `laptop-a`, DataNode role on **both** machines.
- `hdfs namenode -format` once, then `start-dfs.sh` on `laptop-a`.
- Confirm 2 live DataNodes at `http://laptop-a:9870`.
- Load data: `hdfs dfs -put data/processed/... /cascade2vec/data/`

**Known limitation to document, not fix right now:** at exactly 2 nodes with
replication=2, every chunk lands on both machines anyway — you don't get real
"no single machine has everything" behavior until node #3 arrives with
replication < node count. This is expected, not a bug — write it down in
`cluster/README.md` (template below) so it's not rediscovered under deadline pressure.

### 2.5 Fix the GraphFrames version mismatch
Do this while still testing on `local[*]`, before it's distributed — much easier to
debug on one machine:
- Confirm which Maven JAR version actually matches `graphframes-py==0.9.0`.
- Update `GRAPHFRAMES_PACKAGE` default in `loader.py` accordingly.
- Re-test `graph_summary_stats()`'s `connectedComponents()` call actually works.

### 2.6 Make the code cluster-aware (small diffs, big payoff)
In `src/graph/phase04_graph/loader.py`:
```python
master = os.getenv("SPARK_MASTER_URL", "local[*]")
checkpoint_dir = os.getenv("SPARK_CHECKPOINT_DIR", "experiments/logs/04_graph/checkpoints")
# ...SparkSession.builder...appName(app_name).master(master)...
```
In `build_graph.py` and `synthetic_generator.py`: same treatment for output paths —
default to local, override via env var, point at `hdfs://laptop-a:9000/...` when
running against the cluster.

**Do not hardcode `local[*]` or local paths anywhere else going forward.**

### 2.7 Modularity scaffolding (build now, use later)
Create these in the repo even though you only have 2 nodes today:
- `cluster/nodes.txt` — mirrors `workers`, human-readable reference.
- `cluster/setup_node.sh` — one script, run on any new machine, brings it up as
  worker/DataNode. (Provided separately, see below.)
- `cluster/README.md` — the runbook: how to add a node, when to run the HDFS
  balancer, what replication factor means at what node count, current known
  limitations.

**Idempotency check**: before trusting this for a real 3rd node, re-run
`setup_node.sh` against your *existing* 2 nodes and confirm nothing breaks.
This is the single highest-leverage validation step in the whole infra track.

### 2.8 Exit criteria for Sprint 0
- [ ] Both laptops resolve by MagicDNS name, not IP, everywhere in configs.
- [ ] Spark UI shows 2 executors on a `spark.range(1_000_000).repartition(8).count()` job,
      with tasks landing on both hosts (check Executors tab).
- [ ] HDFS UI shows 2 live DataNodes; `hdfs dfs -put`/`-get` works from both machines.
- [ ] `spark.read.parquet("hdfs://laptop-a:9000/...")` returns correct row counts
      matching the original local run (correctness check, not just "didn't crash").
- [ ] GraphFrames `connectedComponents()` runs without the JAR mismatch error.
- [ ] `setup_node.sh` re-run against existing nodes is a no-op / doesn't break anything.
- [ ] `cluster/README.md` documents the replication/node-count tradeoff explicitly.

Once these pass, Phases 1-5 (already built) can be re-run pointed at the cluster to
confirm identical results to the local runs — that's your first real "it's actually
distributed and it's actually correct" checkpoint before building anything new on top.

---

## 3. Track A — Research Pipeline (resume after Sprint 0)

Condensed from the full week-by-week plan — see the original roadmap for full detail
per phase. Weeks are relative to *after* Sprint 0 completes, not calendar weeks.

| Phase | Weeks | What | Depends on Sprint 0? |
|---|---|---|---|
| 6-7 | 2 | Feature engineering + simple baselines (LR/RF/XGB), target F1 > 0.80 on PHEME | No — can run local or cluster |
| 8-10 | 3 | Reimplement RP-DNN, PGNN, Bi-GCN, KPG on identical splits | No |
| 11-12 | 2 | CASCADE2VEC time-weighted GraphSAGE + contrastive training | No |
| 13-14 | 2 | Adaptive early-stopping θ(t), cross-dataset validation (PHEME→Twitter15/16) | No |
| 15 | 1 | XAI (SHAP/LIME), error analysis | No |
| 16-17 | 2 | **Scalability benchmarking** — this is where cluster size is the independent variable | **Yes, hard dependency** |
| 18 | 1 | Ablations + significance testing | No |
| 19 | 1 | Paper assembly, full repro check from clean clone | Cluster setup should be scripted enough to be part of "repro" |

### 3.1 What changes about Weeks 16-17 specifically
This phase's entire point is measuring runtime vs. node count. Concretely:
- Use the **synthetic cascade generator** (`synthetic_generator.py`), not the real
  PHEME/Twitter15/16 data, for the scaling curve — the real datasets are too small
  (102K rows) to show meaningful multi-node speedup.
- Data points: 1 node (local), 2 nodes (real laptops), then simulate additional
  nodes via Docker containers on top of the same 2 physical machines if a 3rd real
  laptop isn't available — but get at least one genuinely 3-node HDFS data point if
  at all possible, since that's what actually demonstrates non-trivial chunking
  (see Sprint 0, section 2.4 caveat).
- Report both strong-scaling (fixed problem size, more nodes) and weak-scaling
  (problem size grows with node count) curves.
- State the replication factor and node count used, explicitly, in the write-up.

---

## 4. Fallback Decisions (decide now, not under deadline pressure)

Same spirit as the original plan's own risk notes — writing these down now means you
don't have to make the call while stressed later:

- **If HDFS setup eats too much time**: fall back to full replication (`scp`/NFS)
  across nodes, state this explicitly as a scope decision in the paper's infra
  section. Compute-parallelism results are still valid; the storage-locality claim
  is what's weakened — say so.
- **If a 3rd physical node never materializes**: use Docker-simulated additional
  workers on top of the 2 real machines for the rest of the scaling curve, and
  clearly label which data points are real hardware vs. simulated in any figure.
- **If adaptive threshold (H2) doesn't beat fixed windows by ~10%**: fall back to
  documented fixed threshold, per the original plan — unchanged by any of this.
- **If Weeks 8-10 (baseline reimplementation) run long**: cut to the 2
  strongest/most-cited baselines, cite published numbers for the rest with a caveat.

---

## 5. Experiment Log (start this from day 1 of Sprint 0, not just Track A)

Log every cluster run too, not just ML metrics — infra runs matter for the H4 claim:

```
date | phase | node_count | replication_factor | dataset | metric | git_commit_hash
```

This is your only defense against forgetting why an earlier scaling number looked
different from a later one (e.g. after adding a node, after changing replication).
