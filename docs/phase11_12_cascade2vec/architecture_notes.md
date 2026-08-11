# Phase 11-12: CASCADE2VEC Architecture Notes

**Status:** 🔧 In Progress
**Training Mode:** Option B — Snapshot-aware (all 8 time windows per cascade)

---

## 1. Time-Decay Formula — Exact Specification

### 1.1 Time Units

All timestamps in `unified.parquet` are stored as **integer seconds relative to
the cascade root** (root tweet always has `timestamp = 0`; all replies are
positive offsets in seconds). This is confirmed by the ingestion pipeline's
`normalize_timestamps()` in `ingest.py`.

The snapshot cutoff windows in `feature_matrix.parquet` are stored as
`t_minutes` ∈ {1, 2, 5, 10, 15, 30, 60, 120}. These convert to seconds as:

| t_minutes | t_seconds (cutoff) |
|---|---|
| 1 | 60 |
| 2 | 120 |
| 5 | 300 |
| 10 | 600 |
| 15 | 900 |
| 30 | 1800 |
| 60 | 3600 |
| 120 | 7200 |

**Within the CASCADE2VEC encoder**, both `t_snapshot` and `t_edge` are kept in
**seconds** (matching the raw data), NOT in minutes. The formula is:

```
w(e) = exp(-λ * (t_snapshot_s - t_edge_s))
```

where:
- `t_snapshot_s` = snapshot cutoff in seconds (e.g., 300 for the t=5min window)
- `t_edge_s` = timestamp of the reply tweet (child node) in seconds from root
- `λ` (lambda) = time-decay rate, measured in **inverse seconds** (s⁻¹)

### 1.2 Sweep Range for λ

Because timestamps range from 0 to ~7200 seconds (2 hours), the meaningful
decay range is:

- `λ = 0.0` → no decay (uniform weights, equivalent to plain GCN baseline)
- `λ = 0.001` → `exp(-0.001 * 7200) = 0.00073` — strong decay over full window
- `λ = 0.0001` → `exp(-0.0001 * 7200) = 0.487` — mild decay over full window

**Recommended sweep grid:** λ ∈ {0.0, 0.0001, 0.0005, 0.001, 0.005}

At λ=0.0 the model degenerates to an unweighted GCN (useful ablation: does
time-weighting actually help?). At λ=0.005 edges older than ~200s get weight
< 0.37 — effectively discarding long-range temporal signal.

### 1.3 Root Node Edge Weight

The root tweet has `t_edge_s = 0` (by normalization). At any snapshot cutoff
`t_snapshot_s`:

```
w(root→child_at_t0) = exp(-λ * (t_snapshot_s - 0)) = exp(-λ * t_snapshot_s)
```

**NOTE:** The root node itself has no incoming edge (it is the cascade source).
For edges *from* the root, the child node's `t_edge_s` is its reply timestamp,
not 0.

The edge weight equals exactly **1.0** when and only when `t_edge_s == t_snapshot_s`
(i.e., the most recently arrived reply at the exact snapshot cutoff). This means
no single static edge gets weight = 1.0 unconditionally — weights depend on
both the edge arrival time and the snapshot cutoff.

**Test requirement (Step 7):** A test must construct a single-edge graph where
the child reply timestamp equals the snapshot cutoff exactly, and assert that
the computed edge weight is `exp(0) = 1.0`. This is tested in
`tests/phase11_12_cascade2vec/test_cascade2vec.py::test_edge_weight_at_zero_delay`.

---

## 2. Temporal Safety: Future Edges Are EXCLUDED, Not Down-Weighted

Replies arriving **after** `t_snapshot_s` are **excluded entirely** from the
snapshot graph — they do not appear as nodes or edges with near-zero weight.

**Rationale:** Assigning near-zero weight `exp(-λ * large_value) ≈ 0` to a
future edge would still leak:
- The **existence** of the reply (edge count changes)
- The **graph topology** (branching structure)
- The **node's text features** (even if message-passing contribution is
  near-zero, the node's TF-IDF vector is still part of the initial feature
  matrix before aggregation)

This matches the identical principle enforced by `assert_snapshot_is_clean()`
in `phase02_ingestion/leakage_audit.py`, which raises a `TEMPORAL LEAKAGE
DETECTED` error if any vertex or edge timestamp exceeds the snapshot cutoff.

**Implementation:** Snapshot graph construction in `cascade2vec.py` filters
nodes and edges by `tweet.timestamp <= t_snapshot_s` **before** building the
PyG `Data` object. The `assert_snapshot_is_clean()` function is called on
every constructed snapshot graph as a runtime assertion.

**Test requirement (Step 7):** `test_no_future_leakage_in_snapshot` constructs
a cascade with known timestamps and verifies that the returned PyG `Data`
object contains zero nodes/edges with `timestamp > t_snapshot_s`.

---

## 3. GraphSAGE Encoder Architecture

### 3.1 How CASCADE2VEC Differs from Phase 8-10 GCN Baselines

| Aspect | Bi-GCN / PGNN (Phase 8-10) | CASCADE2VEC (Phase 11-12) |
|---|---|---|
| **Message weighting** | Uniform (all edges equal) | Time-decay: `exp(-λ * Δt)` |
| **Aggregation** | Sum/mean over all neighbors | Weighted mean, weights = decay |
| **Training regime** | Full cascades only | All 8 snapshot windows per cascade |
| **Graph scope** | Fixed final graph | Variable-size partial graphs |
| **Objective** | Cross-entropy classification | Contrastive (InfoNCE) + classification |
| **Propagation** | Top-down OR bottom-up (Bi-GCN) | Top-down only (propagation direction) |

The critical architectural distinction is that time-weighting is not a
post-hoc scaling applied to an existing GCN — it modifies the **message-passing
aggregation itself**:

Standard GCN aggregation:
```
h_v^(l+1) = σ( W · mean_{u ∈ N(v)} h_u^(l) )
```

CASCADE2VEC time-weighted aggregation:
```
h_v^(l+1) = σ( W · Σ_{u ∈ N(v)} w(u→v) · h_u^(l) / Σ w(u→v) )
```

where `w(u→v) = exp(-λ * (t_snapshot_s - t_u_s))` and `t_u_s` is the
timestamp of the **source node** `u` (the reply arriving at time t_u).

### 3.2 Contrastive Objective

Supervised InfoNCE loss (SupCon variant):
- **Positive pairs:** Two snapshot views of the same cascade (different t values)
- **Negative pairs:** Snapshots of different cascades within the same minibatch
- **Loss:** InfoNCE temperature-scaled dot product of L2-normalized embeddings

This shapes the embedding space so that the *same cascade at different
observation times* produces similar embeddings, while *different cascades* are
pushed apart — the key property needed for Phase 13-14's early-stopping
confidence estimation.

Combined training loss: `L = α·L_contrastive + (1-α)·L_classification`
where α is a hyperparameter (default 0.5, swept in Step 5).

### 3.3 Node Features

Same TF-IDF approach as Phase 8-10 (5000-dim, vocabulary built on train
split only). Applied to the root tweet's text only per cascade — replies'
text is not used as node features (cascade-level TF-IDF, not per-node).
This matches the Phase 8-10 baseline setup exactly, isolating graph structure
+ time-weighting as the variable.

---

## 4. Evaluation Protocol

- **Split:** Exact same `train_val_test_split.parquet` as Phase 8-10
- **Full-cascade H1 comparison:** Evaluate at `t=120` (7200s) snapshot
- **Metrics:** Macro F1 (primary), Weighted F1, Accuracy, Precision, Recall, ROC-AUC
- **Test split:** Used EXACTLY ONCE for final evaluation
- **Seed:** 42 throughout (matching Phase 8-10)
- **Results table:** Added as new row to `logs/phase08_10_sota_baselines/sota_comparison.md`
