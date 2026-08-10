# Phase 3: EDA & Leakage Audit

**Status:** ✅ Complete
**Doc:** [`validated_features.md`](validated_features.md)

### What Was Done
- Exploratory analysis of cascade structure, text, and temporal patterns
- Leakage audit: identified which features are safe to compute without lookahead

### Key Decisions
- All temporal features must be computed within a time window `t` — no using future timestamps
- Text features from replies cannot use reply-level labels (would be leakage)
- Reachability (`reachable_ratio`) and connectivity (`is_connected`) are safe per-snapshot metrics
