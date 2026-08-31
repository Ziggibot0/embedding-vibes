# Exp10.1 — Outcome Crystallization: Barlow Twins on same-outcome session pairs

**Status:** DESIGNED (pre-registered). Not yet run.

## The idea (one paragraph)

exp10 proved matryoshka truncation preserves the delta signal (64-dim AUC
0.928 vs exp6's learned 0.862). Now we TRAIN a 64-dim space to crystallize
OUTCOME: two sessions that end in the same result (both satisfied the user's
request, or both failed/blundered) — however different their paths — should
map to the same point. This is Barlow Twins on OUTCOME pairs: the invariance
pair is "same outcome, different session," not "same action, different
format." Outcome crystallizes, path blurs.

## The claim-shift (honest, pre-registered)

This is **outcome supervision via contrastive pairs**, not self-supervised
learning. The pairs are defined by the outcome label. This is the honest
version of exp12 (per-split outcome supervision) done through the pair
structure instead of a supervised head. It is the direct path to the
trepidation monitor.

**Leakage discipline (pre-registered, like exp9's two-run):** pairs are formed
ONLY within the training split. Held-out sessions' outcomes NEVER inform a
training pair. Eval is on held-out sessions only. If the aligned space
transfers to held-out harnesses, it is real crystallization, not memorization.

## Data

exp8's 3,000 labeled sessions (5 harnesses, 3 benchmarks, 1579 success /
1421 failure). This is the project's real agentic data — the target for
harness-invariant trepidation. Re-embed each step with nomic-embed-text:v1.5
(matryoshka), truncate to 64 dims (exp10's fix), mean-pool steps to a session
vector (exp7b: centroid is the signal carrier).

## Architecture

- **Input:** 64-dim matryoshka session vector (mean of step embeddings).
- **Encoder:** small Barlow-Twins alignment (64→64, LayerNorm). The matryoshka
  truncation IS the base representation (exp10's lesson — no lossy learned
  projection); BT aligns it so same-outcome sessions cluster.
- **Pairs:** two sessions with the SAME outcome (both success, or both
  failure), different content/harness/benchmark.
- **Loss:** Barlow Twins cross-correlation → identity (invariance on diagonal,
  redundancy reduction off-diagonal). Same-outcome → same point.

## Eval (the exam)

**LOHO (leave-one-harness-out):** train BT on 4 harnesses, test outcome
prediction on the 5th. This is the core A-space claim — does outcome
crystallization generalize across harnesses? Linear probe on the aligned
64-dim session vectors.

Also: in-format 5-fold, and compare BT-aligned vs raw matryoshka 64-dim
(does the alignment add value over truncation alone?).

## Pre-registered gates (frozen)

- **G1 (outcome crystallizes):** LOHO mean AUC > 0.60 (above tags-only 0.602
  from exp7a — the same bar exp9 used).
- **G2 (BT helps):** BT-aligned LOHO AUC > raw matryoshka 64-dim LOHO AUC.
  Does the outcome-pair alignment beat truncation alone?
- **G3 (no collapse):** per-dim variance of aligned 64-dim ≥ 0.2 (exp9 G4 bar).
- **G4 (harness-invariant):** LOHO (train on 4 harnesses, test on 5th) — the
  A-space claim. If G4 passes, outcome crystallization is harness-invariant.

## Honest caveats

- **Binary outcome only** (success/failure). Collapsing all failures to one
  point may lose within-outcome structure (e.g., different failure modes).
  For a trepidation monitor this is acceptable (we want "resembles success or
  failure"), but it is a scope limit to state.
- **Agentic data, not fallacy data.** exp8 sessions are tool-call trajectories
  (26 steps), not CoT fallacy arguments. This is MORE aligned with the real
  goal (agent trepidation) but is a different substrate than exp1–exp6.
- **Mean-pooling** loses step-order (exp7b showed centroid is the carrier, but
  the delta/velocity signal from exp5 is discarded). A delta-based session
  vector is a candidate follow-up.

## Artifacts

- `embed.py` — re-embed exp8 sessions with v1.5, truncate to 64, mean-pool.
- `train_eval.py` — BT on same-outcome pairs, LOHO + in-format eval, gates.
- `RESULTS.md` — filled after the run.

## Why this is the right next move

exp10 fixed the representation. exp10.1 uses that sound 64-dim representation
to test the project's core claim directly: does OUTCOME crystallize in a
low-dim, harness-invariant space? If yes, the trepidation monitor is
well-founded. If no, we have cleanly falsified outcome crystallization on
real agentic data. Either way it is the decisive test, and it reuses the
matryoshka fix from exp10.
