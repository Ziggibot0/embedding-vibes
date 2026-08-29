# Wave-interference framing for trajectory aggregation (TABLED 2026-08-28)

## Idea
Interpret aggregation of K predicted trajectory embeddings as wave
interference: constructive = alignment = confidence; destructive = divergence
= trepidation. Would want phase/frequency definitions for trajectories in
embedding space, possibly NPU-accelerated aggregation.

## Why parked
No operational definition of phase/frequency for embedding trajectories exists;
it risks re-cloaking a plain weighted sum ("reviewer attack #2" in
novelty_analysis). The aggregation MECHANISM is in scope (P3 Impl 2); the
interference INTERPRETATION/FRAMING is not.

## Un-park trigger
A measured quantity in K-candidate aggregation that the interference framing
predicts better than magnitude-weighted sums (e.g., a stable "beating pattern"
correlating with failure modes beyond outcome disagreement).
