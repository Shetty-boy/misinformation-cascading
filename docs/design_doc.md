# CASCADE2VEC Design Document

## Problem Statement

Misinformation on social media causes measurable harm in proportion to how long it circulates before being flagged — yet most rumor-detection systems either classify only after a cascade has fully played out (too late to intervene) or apply a fixe
d observation window regardless of how a specific rumor is actually spreading (wasteful for fast, obvious cases; premature for slow, ambiguous ones). Separately, existing rumor-detection models are near-universally evaluated and run on a single machine, leaving open whether they can process the cascade volumes real platforms generate. CASCADE2VEC addresses both gaps: an adaptive, per-cascade stopping mechanism for early detection, built on a distributed graph pipeline (Apache Spark GraphX/GraphFrames) that scales to large cascade volumes.

## Hypotheses

*   **H1 (Representation):** A time-weighted GraphSAGE embedding, where neighbor influence decays with recency of interaction, produces cascade representations that separate rumor from non-rumor cascades more effectively than static graph embeddings or sequence-based baselines (RP-DNN, PGNN, Bi-GCN, KPG).
*   **H2 (Adaptive stopping):** A learned, per-cascade confidence threshold θ(t) — conditioned on cascade properties like velocity, depth, and breadth — achieves equal or better classification accuracy at a lower mean time-to-detection than any fixed observation window.
*   **H3 (Generalization):** Both the embedding (H1) and the adaptive threshold (H2) trained on one dataset (PHEME) transfer with limited performance degradation to structurally different datasets (Twitter15/16), rather than overfitting to one platform's propagation patterns.
*   **H4 (Scalability):** The distributed graph pipeline achieves near-linear speedup in runtime as cluster size increases, across both real cascades and large synthetic cascade graphs.

## Dataset Strategy
*   **Primary Development (PHEME):** Self-contained (full text + structure already released, no re-hydration needed), well-labeled, and the standard benchmark.
*   **Generalization Check (Twitter15/16):** Used for cross-dataset checks. Missing re-hydration text is treated as dropped.
*   **Scalability Benchmark (Synthetic Cascades):** Replaces FakeNewsNet. SIR/SEIZ epidemic-style simulators will generate controllable, arbitrarily large graphs (10M+ nodes) to test distributed system scaling limits.
