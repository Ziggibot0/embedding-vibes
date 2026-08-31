# Experiment Register — Embedding Vibes

Single source of truth for the experiment sequence. Every future experiment has
a numbered home (exp10+), each with a status, the thing it depends on, and the
decision that determines whether we continue.

Status tags: NOT STARTED · DESIGNED · RUNNING · DONE · SUPERSEDED · BLOCKED

> **Convention:** an experiment is "DONE" only when its RESULTS.md carries the
> pre-registered gates and the honest verdict. Nobody is there to catch
> gotchas — the pre-registration IS the gotcha-catcher (see exp9 Run B leakage,
> caught by the A/B discipline). When in doubt, write the gates before running.

---

## The decision ladder (read top to bottom)

The sequence is ordered so each experiment DECIDES whether the next is worth
doing. Do not skip ahead. Each step's result feeds the next gate.

### exp10 — Matryoshka JEPA: the russian-doll fix   [DONE]
**File:** `experiments/exp10_matryoshka_jepa/DESIGN.md` + `RESULTS.md`
**What:** Fix exp6's 64-dim bottleneck. The source embedder was dense
(nomic v1), so crushing to 64 dims blurred the signal (0.960→0.862 AUC).
Swap to nomic-embed-text:v1.5 (matryoshka) and TRUNCATE to 64/128/256/512/768;
measure the task-AUC-vs-dimension elbow. Remove the learned Projector; keep
Barlow Twins on the predictor.
**Result (2026-08-30):** russian-doll hypothesis CONFIRMED. 64-dim truncated
AUC 0.928 vs exp6's learned 0.862 (G1 PASS), near raw 0.960 (G2 PASS). Elbow
not perfectly flat (768=0.990, drop 0.062>0.05, G3 FAIL). JEPA predictor still
weak (G4 FAIL, −40% at 64 dims) — a DATA problem, not representation.
**Decision:** The representation is fixed; the PREDICTOR is now the bottleneck.
Low-dim A-space is viable (0.928 at 64 dims). Next: more data for the
predictor (exp16), not another representation change. exp11 is now well-founded.

### exp10.1 — Outcome crystallization (BT on same-outcome pairs)   [DONE]
**File:** `experiments/exp10_1_outcome_crystallization/DESIGN.md` + `RESULTS.md`
**What:** Train a 64-dim space so two sessions with the SAME outcome (both
success or both failure), however different their paths, map to the same
point. Barlow Twins on OUTCOME pairs (outcome crystallizes, path blurs).
This is outcome supervision via contrastive pairs — the honest version of
exp12. Data: exp8's 3,000 real sessions (5 harnesses, 3 benchmarks).
**Result (2026-08-30):** outcome crystallization WORKS and is harness-invariant.
LOHO mean 0.808 (G1 PASS, >0.60 bar), G4 PASS across 5 held-out harnesses.
But BT alignment does NOT beat raw matryoshka truncation (0.808 vs 0.825,
G2 FAIL) — the raw 64-dim matryoshka rep is the strongest outcome predictor
(0.825 LOHO, 0.845 in-format, no training). smolagents_code regresses under
BT (0.721 vs 0.884).
**Decision:** The trepidation monitor is well-founded (0.825 LOHO,
harness-invariant). Drop the BT outcome-pair objective (G2 fail) — raw
matryoshka is the better, simpler baseline. Investigate smolagents_code
(exp11-style). The matryoshka-vs-dense difference (exp10) + at-scale outcome
signal (exp10.1) is the paper's centerpiece.

### exp11 — Appworld transfer: understand WHY it transfers   [NOT STARTED]
**File:** future `experiments/exp11_appworld_transfer/DESIGN.md`
**What:** exp9 attempt #3's one replicated positive: appworld LOBO
0.741 [0.71,0.77], twice in a row, above every frozen feature (0.44–0.62).
Mean LOBO (0.562) dragged down by browsecompplus (0.485) and swebench (0.460).
Why does the A-space transfer to appworld specifically and NOT those?
Investigate before treating the transfer as general.
**Depends on:** exp10 (representation soundness). If exp10 shows the space
was blurry, exp11's interpretation changes.
**Decision:** If the why is a benign property (e.g., appworld sessions are
more homogeneous), the transfer is a real A-space fact. If it's a confound
(e.g., appworld sessions share formatting/task language), it's not.

### exp12 — Honest outcome supervision (per-split encoder retraining)   [NOT STARTED]
**File:** future `experiments/exp12_outcome_supervision/DESIGN.md`
**What:** The ONLY lever that directly injects the outcome signal. Retrain
the encoder per split (5× cost) so the outcome head never sees held-out
labels. This is what exp9's Run B was contaminated by (leakage); the honest
version retrains per fold.
**Depends on:** exp10 (representation). Only worth it if a sound low-dim
representation exists. Expensive — only pursue if exp10 + exp11 justify it.
**Decision:** Expensive with no guarantee. If exp10 shows matryoshka recovers
signal, this becomes the viable path to outcome prediction.

### exp13 — A-space dim expansion to 16–32   [NOT STARTED]
**File:** future `experiments/exp13_dim_expansion/DESIGN.md`
**What:** exp9 attempt #3's bottleneck is no longer collapse — it's that
8 dims may be too few for outcome signal. Re-train the A-space encoder at
16–32 dims, then re-test the 8-dim projection.
**Depends on:** exp10 (the matryoshka lesson — dim count matters, not just
collapse). Cheap, but must be compared against exp10's elbow findings.
**Decision:** Only if the elbow in exp10 justifies it. "More dims" is not a
goal in itself.

### exp14 — From-scratch encoder (exp6b, scaled up)   [NOT STARTED]
**File:** future `experiments/exp14_from_scratch_encoder/DESIGN.md`
**What:** The 6M from-scratch transformer (exp9) and the 6.37M joint encoder
(exp6b) were too small / low-dim. Scale up the from-scratch encoder trained
WITH the JEPA predictor (MLM + Barlow + JEPA) at higher dims and capacity.
**Depends on:** exp10. If matryoshka truncation recovers the signal for FREE
(no training needed to make 64 dims valid), a from-scratch encoder must beat
that baseline to be worth it.
**Decision:** If exp10 gives us 64 dims for free, exp14 faces a high bar.
Only pursue if a TASK-SPECIFIC representation (format-invariance + outcome)
that truncation can't provide is needed.

---

## Downstream (ROADMAP) experiments — the project's rungs

These map to ROADMAP P0–P4. They come AFTER the exp10–14 ladder where noted.

### P0b — Canonical dataset re-stream with metadata   [NOT STARTED]
**File:** `docs/ROADMAP.md` P0
**What:** Re-stream Exgentic/agent-llm-traces-v2 keeping harness/benchmark/
model fields (the current extract dropped them — the exp7 defect). Build the
logging shim for the user's own stack.
**Status:** NOT STARTED. Unblocks all transfer claims (exp11 needs it to
verify harness-vs-task confound).

### P1b — Trepidation v0 + calibration   [NOT STARTED]
**File:** `docs/ROADMAP.md` P1
**What:** Live-use formulation: prefix summary at step t → 8–64 dim → τ,
with calibration curves (reliability of τ as a probability of eventual
failure) and threshold sweeps.
**Depends on:** a sound representation (exp10) and P0b metadata.

### P2b — Cross-harness retrieval gate for A-space   [NOT STARTED]
**File:** `docs/ROADMAP.md` P2
**What:** "same episode, different harness → same point" (crystallize) +
formatting nuisance directions discarded (blur). exp9 attempt #3 PROVED
format crystallization (margin 0.583). This extends to cross-harness RETRIEVAL
significantly above content baseline. **This is the paper's central claim.**
**Depends on:** P0b (need real same-episode cross-harness pairs or controlled
synthetic re-renderings). exp10 representation soundness feeds the quality.

### P3b — Intervention studies (pre-submit gate / ranking / waymarkers)
**[NOT STARTED / BLOCKED]**
**File:** `docs/ROADMAP.md` P3
**What:** A/B the pre-submit gate on live agent runs; rank K candidates by τ;
waymarker navigation (needs A-space from P2b).
**Depends on:** P1b (v0 exists) + P2b (invariance for waymarkers).

### P4b — Paper / dissertation   [NOT STARTED]
**File:** `docs/paper_skeleton.md`
**What:** the contribution chain: content-vs-geometry decomposition discipline,
calibrated low-dim trepidation (P1), A-space existence test (P2), intervention
evidence (P3).

---

## Parked / on deck (earlier ideas with a home, awaiting a gate)

These are not in the main ladder but must not be forgotten. Each has a trigger.

### exp1b — LOGICCLIMATE cross-domain   [NOT STARTED]
Closest to RUNNING: 1,079 climate-claim examples; the cross-domain candidate
for whether geometric fallacy signatures generalize beyond quiz text.
**Trigger:** after exp10. If signature is representation-bound, test across
domain.

### exp15 — Markov transitions in embedding space   [NOT STARTED]
The transition matrix T[i,j]=count(i→j) as the object of study (nobody has
modeled agentic step-to-step transitions as a Markov process over discrete
embedding cells). "Markov Vibes" paper angle.
**Trigger:** after exp10 establishes the representation.

### exp16 — Own-CoT generation (distribution-shift control)   [NOT STARTED]
The predictor currently learns failure patterns of LARGE models (AgentTrove is
GPT/Claude-class), but the user runs qwen3.5moe (35B-A3B). Generate the user's
own CoT trajectories, label by outcome. Addresses the transfer risk that
reasoning patterns won't transfer across model families.
**Trigger:** P0b shim built; needed before any production trepidation claim.

### RQ3 — Trajectory aggregation (trepidation across K candidates)
**[UNTESTED, promoted to ROADMAP P1/P3]**
The namesake signal: cross-candidate divergence as one operationalization of
trepidation. File: `docs/research_plan.md` RQ3.
**Trigger:** P1b (v0 exists) — this is how we generate candidates.

---

## Experiment map (all experiments, incl. past)

| # | What | Status | Home |
|---|---|---|---|
| 1 | Linear probe, fallacy types | DONE | exp1_linear_probe |
| 1.5/2 | Chunking ablation | DONE (neg) | exp2_chunking_ablation |
| 3 | Markov trajectories | DONE (in-sample flaw) | exp3_markov_trajectories |
| 4 | JEPA spec | SUPERSEDED | exp4_jepa |
| 5/5b | Delta vs static; dim elbow | DONE | exp5b_curves |
| 6 | Joint encoder + JEPA | DONE (64-d bottleneck) | exp6_joint_jepa |
| 7/7a/7b/7c | Real-data gates | DONE | exp7_real_data_gate |
| 8 | Harness-disjoint transfer | DONE | exp8_harness_disjoint |
| 9 | A-space encoder (8-dim) | DONE (3 attempts) | exp9_aspace_encoder |
| 10 | Matryoshka JEPA | **DONE** | exp10_matryoshka_jepa |
| 10.1 | Outcome crystallization | **DONE** | exp10_1_outcome_crystallization |
| 11 | Appworld transfer why | NOT STARTED | (exp11) |
| 12 | Honest outcome supervision | NOT STARTED | (exp12) |
| 13 | A-space dim expansion | NOT STARTED | (exp13) |
| 14 | From-scratch encoder | NOT STARTED | (exp14) |
| 15 | Markov transitions | NOT STARTED | (exp15) |
| 16 | Own-CoT generation | NOT STARTED | (exp16) |
| 1b | LOGICCLIMATE cross-domain | NOT STARTED | (exp1b) |

---

## Method check

- **exp10 is next.** It is the cheapest decisive test of the russian-doll
  hypothesis raised by exp6, and everything downstream depends on a sound
  representation.
- Every "future experiment we talked about" now has a numbered home: exp10
  (matryoshka), exp11 (appworld), exp12 (outcome supervision), exp13 (more
  dims), exp14 (from-scratch), exp15 (Markov), exp16 (own-CoT), exp1b
  (LOGICCLIMATE), plus ROADMAP P0b–P4b and RQ3.
- The order is a DECISION LADDER, not a to-do list: exp10 decides exp11–14's
  viability; P0b unblocks exp11 and P2b; exp16 gates the production claim.
