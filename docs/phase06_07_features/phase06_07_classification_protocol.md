# Phase 6-7: Classification Protocol

## Label Spaces in PHEME

Two conceptually distinct classification tasks exist in the PHEME dataset. They
must NOT be conflated anywhere in the codebase.

---

### Task A — Rumour Detection (ACTIVE — this phase)

| Property | Value |
|---|---|
| Labels | `rumour` / `non-rumour` |
| Source | Folder structure: `rumours/` vs `non-rumours/` per event |
| Ingestion status | **ALREADY IN unified.parquet** (Phase 2 complete) |
| Label column | `label` in `unified.parquet` |

**Label counts (per cascade):**
| Label | Cascade count | Tweet count |
|---|---|---|
| `non-rumour` | 3,830 | 71,210 |
| `rumour` | 1,972 | 31,230 |
| **Total** | **5,802** | **102,440** |

**Class imbalance ratio:** 3830 / 1972 ≈ **1.94:1** (non-rumour : rumour)

**Why this task is frozen for Phase 6-7:**
- No additional ingestion work required — labels exist and are verified
- Matches exactly the task reported by published benchmarks:
  Bi-GCN, RP-DNN, PGNN, and KPG all evaluate Rumour Detection (rumour vs.
  non-rumour) as their primary detection task
- Cascade-level label is consistent (every tweet in a cascade shares the same
  label, since the label is a property of the thread, not the individual tweet)

---

### Task B — Veracity Classification (DEFERRED — future work only)

| Property | Value |
|---|---|
| Labels | `true` / `false` / `unverified` |
| Source | Per-thread `annotation.json` files in each cascade directory |
| Ingestion status | **NOT YET BUILT** — annotation.json is not read by pheme.py |
| Scope | Out of scope for Phase 6-7 entirely |

**Explicit deferral statement:**
Veracity classification requires a new ingestion adapter that reads
`annotation.json` from each cascade directory and maps the veracity field
to a label. This is Phase 2 / stretch goal work. It must NOT be started,
scaffolded, or referenced as an active task during Phase 6-7 feature
engineering or baseline evaluation.

Any model results from this phase are reported **only** on the Rumour Detection
(binary) task.

---

## Active Task for Phase 6-7

**Rumour Detection** — binary classification:
- Positive class: `rumour` (1)
- Negative class: `non-rumour` (0)
- Feature matrix: one row per `(cascade_id, t)` snapshot pair
- Label: cascade-level, copied from `label` column of `unified.parquet` —
  no new label-mapping logic needed
