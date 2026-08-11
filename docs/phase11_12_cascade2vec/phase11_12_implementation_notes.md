# Phase 11-12: CASCADE2VEC — Implementation Notes

**Status:** 🔧 In Progress (Hyperparameter sweep running)
**Training Mode:** Option B — snapshot-aware (all 8 time windows per cascade)
**Source:** `src/cascade2vec/phase11_12_cascade2vec/`

---

## What Was Done

- Implemented `TimeWeightedSAGEConv` — custom GraphSAGE layer with time-decay
  edge weights `exp(-λ * (t_snapshot_s - t_edge_s))`, modifying the message
  aggregation itself (not a post-hoc scaling)
- Implemented `CASCADE2VEC` encoder — 2-layer (sweepable) time-weighted SAGE
  with soft attention pooling and L2-normalised output embedding
- Implemented `SnapshotDataset` — one PyG Data per (cascade_id, t_minutes),
  all future nodes/edges excluded via hard temporal filter + `assert_snapshot_is_clean()`
- Implemented Supervised InfoNCE (SupCon) contrastive objective over
  same-cascade/different-cascade pairs
- Combined loss: `α * L_contrastive + (1-α) * L_classification`
- Implemented hyperparameter sweep (72 configs) over `embed_dim`, `λ`, `n_layers`, `α`
- Wrote 10 unit tests — all passing

---

## Architecture Summary

| Aspect | Value |
|---|---|
| Base architecture | GraphSAGE with attention pooling |
| Temporal mechanism | Time-decay edge weights `exp(-λΔt)` in message aggregation |
| Node features | TF-IDF 5000-dim (root tweet, train vocab only) |
| Training regime | Option B: all 8 snapshot windows per cascade |
| Loss | α * SupCon InfoNCE + (1-α) * CrossEntropy |
| Early stopping | Patience=10 on val Macro F1 |

---

## Key Decisions

- **Option B (snapshot-aware)**: Training on all 8 time windows per cascade
  so Phase 13-14 gets clean per-`(cascade_id, t)` embeddings without retraining.
  H1 comparison uses `t=120min` evaluation only.
- **Time units: seconds** — `t_edge_s` and `t_snapshot_s` are both in seconds
  (matching `unified.parquet` which normalises root to t=0 seconds).
  λ is swept over {0.0, 0.0001, 0.0005, 0.001} (units: s⁻¹).
- **Future edges excluded entirely** — not down-weighted. Assigning near-zero
  weight would still leak graph topology. `assert_snapshot_is_clean()` called
  at every snapshot construction.
- **TF-IDF on root tweet only** (not per-node) — isolates graph structure +
  time-weighting as the variable vs. Phase 8-10 baselines.

---

## Problems Encountered & Resolutions

- **`PYTHONPATH=src` breaks editable install**: Running scripts with
  `PYTHONPATH=src` caused Python to shadow the installed editable package with
  the raw `src/cascade2vec` directory (which is not a package on its own).
  - *Fix:* Run scripts as modules (`python -m cascade2vec.phase11_12_cascade2vec.sweep`)
    or without `PYTHONPATH=src` (rely on the editable install instead).
