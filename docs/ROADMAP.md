# ROADMAP — refocused 2026-08-28

The project's goal, restated at the level that survived this session's audits:

> **Build a live, low-dimensional trepidation monitor for agentic sessions, and
> determine whether an agentic space (A-space) exists that makes it
> harness-invariant. Deploy it as a pre-submit gate, a candidate ranker, and a
> waymarker navigator.**

Everything else in this repo is scaffolding toward that or evidence about it.

## Where the evidence currently stands (one paragraph, honest)

On real sessions (2,000 Exgentic), a frozen-probe monitor is feasible at zero
training: 8-dim state summary → 0.757 AUC, 64-dim → 0.887, full → 0.900, not a
length artifact, insensitive to harness tags. But the equivalent is achievable
with TF-IDF (0.918), and velocity/shape features add **+0.000** beyond content;
format-shift generalization has never been tested because the dataset extract
dropped harness/benchmark metadata (P0 fixes this). On synthetic fixed-length
fallacy data, shape dominates (exp5) — the shape claim survives there, scoped.
"Agentic space" as a *trained, format-invariant, low-dimensional* space is
UNTESTED. Everything above is in S-space (frozen encoder, semantic space).

## The goal ladder (each rung needs the rung below)

### P0 — Instrumentation and metadata repair (gate: canonical dataset with metadata)
- Re-stream Exgentic/agent-llm-traces-v2 **keeping harness, benchmark, model
  fields**; same for one more public source if cheap. Produces the canonical
  dataset v1 (NOMENCLATURE) and fixes the exp7 addendum defect.
- Build the **logging shim** for our own stack (qwen35moe / qwen3.5 on Ollama;
  later the Jarvis agent): append-only JSONL of canonical sessions with
  outcome + harness metadata, from every run we do.
- Deliverable: `data/canonical/sessions.v1.jsonl` + shim in `src/shim/`.
- Gate check: ≥5 distinct harness/benchmark groups with ≥100 sessions each.
- Status: NOT STARTED.

### P1 — Trepidation v0 on real sessions (gate: calibrated early warning)
- Live-use formulation: features available at step t only (prefix summary →
  8–64 dim projection). Report prefix curves (have: 3 steps 0.787 → 16 steps
  0.863) and, critically, **calibration curves** (reliability of τ as a
  probability of eventual failure), threshold sweep → precision/recall at
  operationally plausible thresholds.
- Mandatory controls from day one (content, length, tags, session-disjoint).
- Deliverable: trepidation v0 module + calibration report.
- Gate to proceed: τ at ≤25% of mean session length must reach ≥0.70 AUC AND
  remain useful after content-control subtraction (i.e., prefix shape carries
  something the prefix words don't — if not, v0 is still useful, but we say
  "lexical early warning" honestly).
- Status: most ingredients exist from exp7/exp7b; calibration not yet computed.

### P2 — A-space existence test: format crystallization (the scientific core)
- Claim under test: a low-dim representation can be built where "same episode,
  different harness" → same point ("crystallize"), and formatting nuisance
  directions are discarded ("blur").
- Data: canonical v1, plus **synthetic re-renderings**: same sessions re-
  serialized through N formatting templates (chat-JSONL, OTel-ish, markdown,
  tag-permuted) → augmentation pairs without extra model calls.
- Training: contrastive/Barlow-Twins alignment on rendering pairs + outcome
  signal (small supervised head); 8–64 dims.
- Gate (pre-registered): cross-harness step/session retrieval from A-space
  significantly above content baseline; outcome AUC under harness-disjoint
  split ≥ frozen-probe baseline, at ≤64 dims. Fail either → A-space stays
  UNTESTED/refuted and the monitor remains harness-local (still deployable).
- Status: NOT STARTED. This is the make-or-break for the paper's central claim.

### P3 — Intervention studies (the engineering payoff)
- Pre-submit gate (Impl 1): A/B the gate on our own agent runs — task success
  rate and cost with gate on vs off, matched budget. Gate: significant Δsuccess
  per token spent, on live runs, not replay.
- Candidate ranking (Impl 2): K candidates from our base model, rank by τ,
  compare vs random/LLM-judge selection on the same tasks.
- Waymarkers (Impl 3): prototype only, after P2 — centroid templates require
  A-space to exist (Impl 3 is undefined if positions shift across harnesses).
- Status: NOT STARTED. Blocked by P1 (v0 exists) and P2 (invariance needed
  for templates).

### P4 — Paper (dissertation track)
- Working title: "Low-Dimensional Trepidation: Harness-Invariant Geometric
  Early Warning for LLM Agent Sessions."
- Contribution chain as it stands today:
  1. Existence + strength of zero-training session outcome signal, WITH the
     content/lexical decomposition nobody else reports (done, exp7 + addenda).
  2. Dimensionality result: outcome signal survives 768→8-d projection
     (0.757 AUC) — the low-dim agentic monitor is feasible in principle (done).
  3. Calibrated early warning at embed-only cost (P1).
  4. A-space existence test with format-crystallization (P2) — the novel
     scientific claim, positive or negative result both publishable.
  5. Intervention evidence from a live agent (P3) — the figure that makes it
     real for practitioners.
- What the paper is NOT claiming (mandatory honesty section): no NPU; no
  encoder-training on the working line; velocity-dominance on real data
  FALSIFIED; cross-harness invariance not yet demonstrated; frozen-encoder
  geometry at nomic scale is largely lexical for fallacy-type classification.

## Why this order

The audits taught the lesson that killed exp5→exp7's delta-dominance and would
have sunk the paper: every geometry claim dies or survives on its controls, and
public aggregates without metadata can't support transfer claims at all.
P0/P1 make the signal real and honest on OUR system. P2 is the first
experiment designed to test the actual idea (A-space) rather than a proxy.
P3 converts it into the three implementations the project exists for.
Academic framing: 1+2 are the solid engineering/evidence contribution even if
P2 fails; P2 positive is the high-novelty claim; P3 makes it a systems paper.

## Explicitly out of scope

See `ideas/` — parked concepts live there (wave interference, NPU, from-scratch
encoder exp6b, learned trajectory encoders, decoder attempts). They are not
forgotten; they are not in the way.