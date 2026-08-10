# Phase 2: Data Ingestion

**Status:** ✅ Complete
**Log:** [`../../logs/phase02_ingestion/phase02_data_audit.md`](../../logs/phase02_ingestion/phase02_data_audit.md)

### What Was Done
- Parsed all PHEME JSON files into a unified flat parquet
- Output: `data/processed/phase02_ingestion/unified.parquet`

### Schema
| Column | Type | Description |
|---|---|---|
| `tweet_id` | str | Unique tweet identifier |
| `cascade_id` | str | Root tweet ID (cascade group key) |
| `parent_id` | str/NaN | Parent tweet ID (NaN for root) |
| `text` | str | Tweet text |
| `timestamp` | datetime | UTC timestamp |
| `label` | str | `rumour` / `non-rumour` (thread-level) |
| `event` | str | PHEME event name |

### Dataset Statistics
| Metric | Value |
|---|---|
| Total cascades | 5,802 |
| Total tweets | 102,440 |
| Rumour cascades | 1,972 (34.0%) |
| Non-rumour cascades | 3,830 (66.0%) |
| Class imbalance | 1.94:1 (non-rumour:rumour) |

### Key Decisions
- Labels come from PHEME folder structure (`rumours/` vs `non-rumours/`), not tweet content
- Cascade-level label only — every tweet in a cascade shares the same label
- Task B (veracity classification: true/false/unverified) deferred indefinitely — not the target task
