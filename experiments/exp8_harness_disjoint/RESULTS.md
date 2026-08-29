# Exp8 — RESULTS

**Status: RUNNING** (embedding phase, ~70 min remaining at time of writing).
This skeleton pre-commits the interpretation framework; numbers will be filled
from `results/exp8_summary.md` and `results/exp8_results.json` when the run
completes. No post-hoc gate edits.

## Setup
3,000 Exgentic sessions (cap pre-declared in DESIGN.md), metadata intact:
5 harnesses (4 evaluable at n≥150), 3 benchmarks, 5 models, 52.6% success.
Features and protocol identical to exp7 (frozen nomic-embed-text,
LogisticRegression C=1.0, StandardScaler; PCA-8/Tfidf/tag-vocab fit on train
fold only). Splits: in_format 5-fold stratified CV (seed 42); LOHO/LOBO/LOMO
leave-one-group-out. Metrics: AUC, Acc, Brier, 10-bin ECE.

## Main table (to fill)

| feature | in_format AUC | LOHO AUC | LOBO AUC | LOMO AUC |
|---|---|---|---|---|
| static_full | — | — | — | — |
| static_pca8 | — | — | — | — |
| meanvel | — | — | — | — |
| tfidf | — | — | — | — |
| tags | — | — | — | — |
| length | — | — | — | — |
| tfidf+meanvel | — | — | — | — |

## Pre-registered gates (frozen; verdicts to fill)

- **Gate A (monitor transfers)**: static_full in_format ≥ 0.75 AND LOHO ≥ 0.70 → —
- **Gate B (geometry retains > lexical under shift)**: drop(tfidf) − drop(static_full) ≥ 0.03 → —
- **Gate C (velocity null recheck)**: (tfidf+meanvel) − tfidf ≤ 0.02 in_format → —
- **Gate D (low-dim transfers)**: drop(static_pca8) ≤ drop(static_full) + 0.02 → —

## Interpretation ladder (pre-committed)

- If A passes + B fails → monitor is deployable across harnesses but rides
  content; A-space must be *manufactured* in P2 from a zero floor. Claim
  language: "content-based trepidation transfers; geometric margin absent."
- If A passes + B passes → embedding retains something specifically beyond
  lexical form under format shift — the A-space bet has a floor. P2 trains
  from a standing start.
- If A fails → harness transfer needs per-harness calibration or the signal
  is harness-local; P1 deployment scoped to same-harness; A-space claim
  must overcome a measured *negative* floor.
- C reopens shape only if it FAILS (increment > 0.02).
- D failing while A passes → the 8-dim projection is format-sensitive;
  waymarker work (P3 Impl 3) needs higher-dim or crystallized features.

## Honest-scope reminders (will apply to whatever the numbers say)

1. One dataset (Exgentic), one encoder (nomic), 5 models — results scoped to
   "public OTel-ish agent logs, frozen small encoder."
2. LOHO trains on 4 harnesses and tests on the held-out one — it measures
   *transfer to unseen format*, not deployment on our own stack (P0 shim data
   is the other half of this gate).
3. ECE/Brier matter for P1 thresholds; AUC alone does not license gating.