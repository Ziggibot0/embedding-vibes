# Exp7 — Delta vs Static on 2000 REAL agentic sessions (the decisive gate)

## Setup

2000 Exgentic sessions with real success/failure labels (818 success / 1182
failure), multi-harness, benchmarks incl. AppWorld. nomic-embed-text step
embeddings (all-cached, 30,841/30,844 texts). Same 5-fold CV protocol as
exp5. Includes the mandatory length-only control (length leakage warning
from related work).

## Result

| Method | AUC (5-fold) |
|---|---|
| Length-only control | 0.561 |
| **Static (centroid+final)** | **0.901** |
| Delta (concatenated velocities, PCA50, zero-padded) | 0.655 |
| Static + Delta | 0.903 |

## VERDICT: NEGATIVE — on real agentic data, the delta/velocity representation
is WORSE than the static representation for outcome prediction.

## What this means (honest)

1. **The exp5 result did not transfer.** On the synthetic 90-session fallacy
   data (fixed 6 steps, single-topic prompts), deltas beat static by +0.15.
   On 2000 real sessions with variable length (avg ~18 steps, range varies),
   static wins by -0.25. The temporal/velocity mechanism was NOT the reason
   exp5 worked — or the variable-length padding destroyed the delta signal
   (see below).

2. **Length control passed**: 0.561, so the static 0.901 is not a length
   artifact. Static geometry of real agentic sessions carries real outcome
   signal at 0.901 AUC with zero training.

3. **Two competing explanations, not yet separable:**
   a. **Padding dilution**: zero-padding short sessions' deltas to the longest
      session's length means the PCA space is dominated by trajectory-length
      differences; delta signal for short sessions gets smeared. The delta
      features are length-confounded in a way the static ones are not.
   b. **Real difference in mechanism**: for fallacy-vs-valid (same task
      format, 6 steps), WHERE the reasoning went (transitions) matters most.
      For success-vs-failure on real tasks (variable length, varied topics),
      WHERE IT ENDED UP (final state / centroid) carries more.

4. **The static 0.901 is itself the strongest number in the project.** An
   untrained linear probe on frozen nomic centroid+final embedding separates
   SUCCESS from FAILURE on real agent sessions at 0.901 AUC. No JEPA, no
   deltas, no learned encoder. That is the cheapest, most robust deployment
   path: embed the session prefix, probe the centroid.

## Follow-up: exp7b disambiguation (run same day, cache-only)

| Test | Result | Reading |
|---|---|---|
| mean-velocity, fixed-length (no padding) | AUC 0.808 | Recovers most delta signal — padding WAS part of the failure |
| last-velocity, fixed-length | AUC 0.693 | The *last* transition alone is weak |
| speed stats (mean/max speed, net disp — 3 dims) | AUC 0.532 | Magnitude features carry almost nothing; direction is the signal |
| length-stratified (18 steps) | n=58 only, skipped | Too few exact-length sessions |
| prefix curve (static) | 3 steps: 0.787 → 8: 0.821 → 16: 0.863 | Real early-warning, but slower ramp than exp5b's synthetic |
| centroid only | AUC 0.898 | The CENTROID is the signal carrier |
| final state only | AUC 0.858 | Final alone is less than centroid |

**Resolution:** explanation (a) was substantially right — the zero-padding
diluted the delta signal. Fixed-length mean-velocity recovers 0.808 (vs
0.655 padded), but STILL loses to static 0.901. So on real data: direction
of motion carries signal (mean velocity 0.808 >> speed 0.532), but aggregate
POSITION (centroid, 0.898) still wins. The exp5 full-delta advantage remains
unexplained by either — its 90-session setup (synthetic, single-topic, fixed
length, same-task pairs) differs from real sessions in every dimension.

## Honest bottom line

1. **Strongest result in the project: centroid, 0.898-0.901 AUC, zero
   training, on 2000 real sessions.** An untrained probe on frozen nomic
   centroid embeddings separates success from failure on real agent
   trajectories. This is the deployment anchor.
2. **The velocity/delta mechanism is real but secondary on real data**
   (0.808 vs 0.901). It beat the speed/magnitude features decisively
   (direction >> magnitude), supporting the "similarity is not relevance /
   position-vs-velocity" framing — but loses to position for outcome
   prediction on real sessions.
3. **The from-scratch encoder (exp6b) and the learned shape-encoder plans
   ON HOLD** — their justification rested on exp5's delta-dominance,
   which did not transfer. Any next encoder investment must beat the
   0.901 centroid baseline, not exp5's synthetic numbers.
4. **Paradigm implication**: on real agentic data, success/failure is mostly
   encoded in WHERE the trajectory sits (task/topic-state region), less in
   HOW it moved — the opposite of the fallacy-type result. Fallacy-type
   detection (a *style/structure* classification) may still favor shape;
   outcome prediction (a *content* regression) favors position. Both things
   can be true: exp5's claim survives scoped to same-format, same-length,
   structure-classification tasks.

## Status

The pilot's core mechanism (velocity/delta shape) FAILED to beat static on
real data in its first real test. The static representation at 0.901 AUC is
the strongest simplest result and should anchor the next iteration. All of
exp6b (from-scratch encoder) and the learned trajectory-encoder plans are now
ON HOLD — they were justified by the exp5 signal, which did not transfer.