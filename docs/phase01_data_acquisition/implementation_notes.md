# Phase 1: Data Acquisition

**Status:** ✅ Complete
**Source doc:** [`design_doc.md`](design_doc.md)

### What Was Done
- Acquired PHEME dataset from Zubiaga et al. (2016)
- Events: charliehebdo, ebola, ferguson, germanwings-crash, gurlitt, ottawashooting, prince-toronto, putinmissing, sydneysiege
- Organised raw JSON into project directory structure

### Key Decisions
- Raw data kept intact under `data/raw/` — never modified in place
- All transformations go through Phase 2 ingestion pipeline
