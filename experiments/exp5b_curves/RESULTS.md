# Exp5b — Two design curves on existing exp3 embeddings (no training, no new embedding)

## Curve A — Dimensionality: where is the real elbow?

Task AUC (delta representation) vs PCA dimension, 5-fold CV:

| dim | nomic AUC | qwen3 AUC |
|---|---|---|
| 8 | 0.983 | 1.000 |
| 16 | 0.980 | 1.000 |
| 32 | 0.973 | 0.998 |
| 50 | 0.968 | 0.988 |
| 64 | 0.968 | 0.943 |
| full (2500/20480) | 0.995 | 1.000 |

(Caveat: dims ≥ 96 exceed the sample count (90), so PCA keeps everything and
degenerates — AUC ~0.45 is an artifact of n_components = 90 on 90 samples,
not a real signal loss. The meaningful range is 8-64.)

**Finding 1: even 8 dims preserve the shape signal.** The elbow is not 64 —
it's at or below 8 for the PCA'd raw-delta representation. Sean's low-dim
instinct ("Russian-doll" argument) is directionally right: the signal survives
extreme compression.

**Finding 2 — this contradicts exp6, and the contradiction is informative.**
exp6's learned Barlow-Twins encoder LOST signal at 64 dims (AUC 0.862/0.864),
but PCA preserves nearly all of it at 8-64 dims. So the problem was never
dimensionality — it's what the learned encoder optimizes. Barlow Twins
decorrelation actively suppresses low-variance directions, and task-relevant
info can sit exactly there (the SCALE 2026 hazard). If we build a learned
encoder, the loss design must not discard low-variance discriminative info —
or we simply use PCA/raw deltas as the shape space.

## Curve B — Early detection: how many steps do we need?

AUC using only the first N steps of each trajectory (deltas of the visible prefix):

| visible steps | nomic AUC | qwen3 AUC |
|---|---|---|
| 2 | 0.894 | 0.970 |
| 3 | 0.993 | 0.993 |
| 4 | 0.998 | 1.000 |
| 5 | 0.995 | 1.000 |
| 6 (full) | 0.995 | 1.000 |

**Finding 3: early warning is real.** At 3 of 6 steps (half the trajectory),
AUC is already 0.993 on both encoders; at 2 steps it's 0.89-0.97. The pilot's
early-abort use case has empirical support: you don't need to watch the whole
trajectory to catch the failure trajectory shape.

## Implications for the pilot architecture

1. **Shape space**: raw deltas + tiny PCA (8-32 dims) is a strong, free shape
   representation. Any learned encoder must beat THIS, at the same budget.
2. **Encoder loss design**: Barlow Twins' redundancy reduction is suspect — it
   may discard the exact low-variance directions that carry task signal.
   The stylistic-vs-logical control and a supervised outcome head become more
   important than raw representation-learning scale.
3. **Early-abort value**: confirmed at prefix=3. The JEPA predictor's job is to
   extrapolate from a 2-3 step prefix to full-trajectory shape — the target
   AUC to beat is ~0.99 from just 3 steps.

## Caveats

- 90 synthetic sessions (exp3 data), fixed 6-step trajectories.
- exp7 (running) tests the same curves on 2000 REAL sessions with outcome
  labels — that's the decisive number. These curves predict *what to look for*.