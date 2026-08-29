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

---

## ADDENDUM (2026-08-28): content-vs-shape controls (session-disjoint folds, cache-only)

Question raised: is the 0.901 signal "path shape" or just session content? Does
velocity (the time-implied path shape) add anything beyond the words?

Controls, same session-disjoint 5 folds (seed 42), LogisticRegression(C=1.0):

| Feature set | AUC |
|---|---|
| Format/harness tags only (top-40 [tool_call]/[user]/... counts + rates) | 0.602 |
| mean-velocity only (768d, fixed-length) | 0.802 |
| Static centroid+final (same folds, ref) | 0.900 |
| TF-IDF bag-of-words on raw session text (2k) | **0.918 ± 0.010** |
| TF-IDF + mean-velocity | 0.918 (velocity increment +0.000) |
| TF-IDF + centroid | 0.929 |
| TF-IDF + centroid + velocity | 0.925 (velocity increment ~0) |

### Findings
1. **Frozen-nomic "geometry" ≈ lexicon on real sessions.** TF-IDF over the raw
   step texts MATCHES/BEATS the static centroid features (0.918 vs 0.900).
   The untrained-probe 0.901 is real signal, but it is mostly carried by WHAT
   the session says, not embedding-geometry-specific structure. Same lesson as
   exp1's TF-IDF control, now confirmed on exp7 data.
2. **Velocity/path-shape adds ZERO beyond content.** +0.000 over TF-IDF alone,
   and ~0 even stacked on TF-IDF+centroid. mean-velocity's 0.808 is fully
   contained in session content: trajectories do not carry direction-of-motion
   outcome signal beyond what the text already provides on this dataset.
   ("Velocity conveys failure with less confounding" is FALSIFIED here.)
3. **Harness formatting is not the signal** (tags-only 0.602) — supports
   format-invariance, i.e. features are not riding harness markers.

### Implication for the cross-harness "process signature" hypothesis
The strong claim — a time-implied path shape IDENTIFIABLE across harnesses and
formats distinct from content — is UNTESTED, and the current data cannot test
it: exp7's sessions.jsonl keeps no benchmark/model/harness fields (only
steps+success+source). Next experiment (exp8) requires re-streaming
Exgentic/agent-llm-traces-v2 KEEPING benchmark + model metadata, then
harness-disjoint CV (train benchmarks A-C, test D-F) on shape-only features
(velocities, curvatures, tag-free whitened steps). Falsification criterion:
shape-feature AUC under harness shift >= 0.70 with content controls matched.
Until then, the honest summary: on real sessions the deployable signal is
content position (lexical or centroid), not trajectory shape.
