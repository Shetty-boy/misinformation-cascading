# Phase 1: Data Acquisition

**Status:** ✅ Complete
**Source doc:** [`phase01_design_doc.md`](phase01_design_doc.md)

### What Was Done
- Acquired PHEME dataset from Zubiaga et al. (2016)
- Events: charliehebdo, ebola, ferguson, germanwings-crash, gurlitt, ottawashooting, prince-toronto, putinmissing, sydneysiege
- Organised raw JSON into project directory structure

### Key Decisions
- Raw data kept intact under `data/raw/` — never modified in place
- All transformations go through Phase 2 ingestion pipeline

### Problems Encountered & Resolutions
- **Nested Raw Data:** The raw dataset came as heavily nested JSON structures spanning 9 different events, making it difficult to parse uniformly.
  - *Fix:* Reorganized raw JSON strictly into a unified project directory structure without modifying the raw data in-place, passing the burden of structuring to the Phase 2 ingestion pipeline.
