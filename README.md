# embedding-vibes

Trepidation monitor for LLM agent sessions: predict trouble ahead from the
geometry of a reasoning/act trajectory, in a low-dimensional, harness-invariant
"agentic space."

> **RE-FOCUSED 2026-08-28** — after a full audit session. The project goal is
> now a live trepidation score with three uses (pre-submit gate, candidate
> ranking, waymarker navigation) and a format-crystallization experiment that
> tests whether an agentic space exists at all. See `docs/ROADMAP.md`.

## Vocabulary first

**Read `docs/NOMENCLATURE.md` before anything else.** It defines: step,
session, trajectory, velocity, path shape, semantic space (S-space), agentic
space (A-space), trepidation, the mandatory controls (content / length /
format-tags / session-disjoint / harness-disjoint), and the claim-status tags
(PROVEN / SUGGESTED / FALSIFIED[scope] / UNTESTED) used in every results doc.
Claims are never upgraded by rephrasing — only by new experiments with
controls.

## Core idea

LLMs generate locally coherent reasoning that globally fails. A separate tiny
model watches the embedding trajectory of a running agent session and outputs
**trepidation** — a calibrated early-warning score meaning "this resembles
trajectories that went badly." No reasoning, no logic checking — geometric
pattern matching on the session's path through representation space.

Three deployments (all consume the same signal):
1. **Pre-submit gate** — τ above threshold → the model tries a different approach.
2. **Candidate ranking** — among K candidate next actions, take the lowest τ.
3. **Waymarkers** — centroid templates form a "yellow brick road" in A-space;
   steering keeps the live trajectory near it (no decoder — navigation only).

The scientific bet: **action crystallizes while formatting blurs** — the same
intent under different harness log formats maps to the same point in a
low-dimensional A-space. This is trainable (format-crystallization objective)
and is the paper's central testable claim (ROADMAP P2). It is currently
UNTESTED; the monitor itself is buildable regardless (P0→P1).

## Architecture (target)

```
live agent run (our stack: qwen35moe 35B-A3B, via Ollama)
    │  logging shim → canonical session format (steps + outcome + metadata)
    ▼
Step embedder (frozen; nomic-embed-text / qwen3-embedding)
    ▼
Prefix summary φ(S_1..t)  → 8–64 dim projection (dims are a measured curve)
    ▼
Trepidation τ(t)  ← calibrated against realized outcomes
    ├── Impl 1: threshold gate before text submission
    ├── Impl 2: rank K candidates by τ
    └── Impl 3: distance-to-waymarkers (after A-space exists, P2)
```

A-space (format-invariant, 8–64 dims) is manufactured in P2 via
format-crystallization training — contrastive alignment of same-episode
re-renderings across harnesses. Until then all measured signal is S-space.

## Evidence base (post-audit, claim-status tagged)

| Exp | What | Result | Status |
|---|---|---|---|
| 1 | Linear probe, fallacy types, dual encoder | qwen3 65.1% (real, >TF-IDF by +12pts, p~1e-42); nomic 50.8% ≈ TF-IDF 49.0%; site-disjoint 53.5%; binary AUC 0.987 (n=68) | PROVEN [LOGIC/quiz text, per-encoder], re-audited 2026-08-28 |
| 2 | Chunking ablation | Chunking hurts; signal is in transitions | PROVEN [synthetic] |
| 3 | Markov chains on trajectories | AUC 1.0 in-sample; leave-one-out needed | SUGGESTED, in-sample flaw |
| 4 | JEPA architecture spec | Design only | superseded by exp6 |
| 5 | Delta vs static, synthetic fallacy pairs | delta 0.970/1.000 vs static 0.817/0.956 | PROVEN [90 synthetic sessions, fixed 6-step] |
| 5b | Dim + early-curve | elbow ≤ 8 dims; AUC ~0.99 from 3 of 6 visible steps | SUGGESTED, synthetic |
| 6 | Learned 64-dim encoder + JEPA | Predictor works (beats mean 62–73%); 64-d learned projection loses signal | PROVEN [the loss]; learned-encoder plan paused |
| 7 | **Real-data gate: delta vs static, 2000 Exgentic sessions** | static 0.901 vs delta 0.655; exp7b: mean-velocity 0.808, centroid alone 0.898; length control 0.561 | PROVEN [Exgentic, nomic], NEGATIVE for velocity-dominance |
| 7a | **Content controls (this session)** | TF-IDF on session text 0.918; velocity increment over TF-IDF **+0.000**; harness tags alone 0.602 | PROVEN [Exgentic, session-disjoint] — signal is content-position, not path-shape; format tags are not the signal |
| 7c | **8-dim state curve (this session)** | PCA-8 → 0.757 AUC; 32 → 0.866; 64 → 0.887; full → 0.900; top dims ~uncorrelated with length | PROVEN [Exgentic] — useful low-dim state signal exists |

**The honest headline:** a zero-training frozen-probe monitor is feasible
(0.757 AUC from 8 dims, early-warning ramp confirmed), but its "geometry" is
currently content-dominated (TF-IDF-equivalent on real sessions), velocity is
falsified as an outcome predictor beyond content on real data, and cross-harness
invariance is untested (the dataset extract had no harness metadata — fixed as
ROADMAP P0). The synthetic shape result (exp5) survives scoped to fixed-length
structure classification.

## Datasets

- **Exgentic/agent-llm-traces-v2** — 10K agent runs, 6 benchmarks, outcome
  labels. Our extract (`exp6_joint_jepa/data/sessions.jsonl`, 2,000 sessions)
  currently LACKS harness/benchmark metadata → P0 re-stream.
- **open-thoughts/AgentTrove** — 1.7M trajectories (20,000 extracted,
  success-labels unreliable → self-supervised use only).
- **Jin et al. 2022 LOGIC** — 2,449 fallacy examples (exp1) + LOGICCLIMATE
  1,079 climate claims (unused; candidate exp1b).
- Our own runs via the logging shim (P0) — the outcome signal that matters.

## Hardware

Ryzen AI MAX 385 (Strix Halo): 8x Zen 5, Radeon 8050S iGPU, ~32GB unified.
**No NPU** (scope, 2026-08-28). Encoders served via Ollama (nomic ≈ fast,
qwen3-embedding for the strong end). All probes/predictors are tiny — CPU is
the right home for them.

## Repo map

- `docs/NOMENCLATURE.md` — vocabulary + mandatory controls + claim tags.
- `docs/ROADMAP.md` — the refocused goal ladder (P0 instrumentation → P1
  trepidation v0 → P2 A-space crystallization → P3 interventions → P4 paper).
- `docs/` — supporting analyses (why_jepa, related work, older plan docs,
  being caught up to this conversation).
- `experiments/exp1..exp7b` — the evidence base above, each with RESULTS.md
  carrying its controls and addenda (audit trail preserved on purpose).
- `ideas/` — parked out-of-scope concepts (interference, NPU, from-scratch
  encoder, templates/decoder explorations). Parked ≠ dead.

## Status

**P0 next action:** re-stream public sessions with metadata + build the
logging shim. Trepidation v0 (calibrated, controlled) is the first product
milestone. The paper claim that matters (A-space crystallization) is P2 and
has not been tested yet — everything said publicly before P2 is
monitor-feasibility science.