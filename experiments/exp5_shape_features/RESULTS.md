# Exp5 — Differential Shape Features vs Static Probe (the pilot gate)

**Question:** do trajectory SHAPE features separate fallacy from valid reasoning
better than the static endpoint probe? This is the gate for the pilot architecture
(DESIGN.md).

**Data:** the 90 exp3 sessions (45 fallacy / 45 valid, 6 steps each, 20 topics),
step embeddings for both encoders. Same 5-fold CV protocol for all families.

## Results (5-fold CV, mean AUC)

| Method | nomic AUC | qwen3 AUC |
|---|---|---|
| A. Static baseline (centroid+final) | 0.817 | 0.956 |
| B. Hand-crafted shape stats (velocity/curv/loop) | 0.738 | 0.642 |
| C. **Relative-delta (PCA50)** — the pilot's core mechanism | **0.970** | **1.000** |
| B+C. Shape + Static | 0.835 | 0.956 |

## Reading

1. **The pilot's core mechanism is validated.** The relative-delta representation
   (concatenated velocities, translation-invariant) beats the static baseline on
   nomic (0.970 vs 0.817) and reaches ceiling on qwen3 (1.000). "Shapes as
   relative relationships" carries signal that static position alone misses —
   at least on the weaker encoder, where static is only 0.817.

2. **Hand-crafted shape stats are weak.** Velocity/curvature/loop-closure summary
   features (B) underperform static on both encoders. The signal lives in the
   *full* delta sequence, not in hand-picked scalar summaries. This argues for
   the learned trajectory encoder (which sees all deltas) over hand-crafted
   features — exactly the pilot's design.

3. **qwen3 is near-ceiling for both static and delta.** On the stronger encoder
   the static probe is already 0.956, so delta (1.000) is only marginally better.
   The delta advantage is clearest where static is weaker (nomic). This suggests
   the temporal signal is most valuable when the static signal is limited.

## Caveats (honest)

- **Stylistic-vs-logical not yet controlled.** This data has no paraphrases, so
  we cannot separate "shape of the reasoning" from "shape of the surface text."
  The delta advantage could be stylistic. Follow-up: paraphrase control.
- **Small N.** 90 sessions, 5-fold CV. The nomic delta advantage (0.970 vs 0.817)
  is large but should be confirmed with more data / leave-one-out.
- **6 steps only.** All sessions are exactly 6 steps. Real agentic flows are
  longer and more variable; the delta representation generalizes but this is a
  fixed-length test.

## Verdict

The pilot's core mechanism (relative-delta shapes) has legs: it beats static on
the weaker encoder and matches it at ceiling on the stronger one. The next step
is the learned trajectory encoder (Barlow Twins + joint JEPA training) that
operates on the full delta sequence — the hand-crafted summary features are not
the way.
