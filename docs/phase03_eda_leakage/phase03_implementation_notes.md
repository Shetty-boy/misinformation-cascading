# Phase 3: EDA & Leakage Audit

**Status:** ✅ Complete
**Doc:** [`phase03_validated_features.md`](phase03_validated_features.md)

### What Was Done
- Exploratory analysis of cascade structure, text, and temporal patterns
- Leakage audit: identified which features are safe to compute without lookahead

### Key Decisions
- All temporal features must be computed within a time window `t` — no using future timestamps
- Text features from replies cannot use reply-level labels (would be leakage)
- Reachability (`reachable_ratio`) and connectivity (`is_connected`) are safe per-snapshot metrics

### Problems Encountered & Resolutions
- **Temporal & Label Leakage:** Initial exploratory analysis highlighted risks of temporal leakage (using future timestamps) or label leakage (using reply-level labels for the whole cascade) inflating baseline performance.
  - *Fix:* Defined strict guardrails preventing future lookahead by enforcing dynamic time windows (`t_minutes`) and relying exclusively on cascade-level PHEME labels.
