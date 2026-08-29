# Nomenclature

Project vocabulary. Every term below is used consistently across README, DESIGN,
ROADMAP, experiment docs, and (eventually) the paper. Definitions are
operational — each term says how it is measured, not just what it means.

## Claim-status legend

Claims in this repo are tagged with one of these. Never upgrade a tag by
rephrasing; upgrade it only with a new experiment + control.

- **PROVEN** — demonstrated on real data with the mandatory controls (below).
- **SUGGESTED** — observed on synthetic/small data; expected to generalize; unconfirmed on real data.
- **FALSIFIED [scope]** — failed a controlled test; the bracket states the scope within which it was falsified (may still hold elsewhere).
- **UNTESTED** — no experiment addresses it yet. Saying so is required, not shameful.

## The mandatory controls (referenced throughout)

Every claim that a *geometric/structural* feature carries signal must be paired,
under identical splits, with:

- **Content control** — TF-IDF (or equivalent bag-of-words) on the raw step
  texts. Established 2026-08-28: matched or beat embedding geometry in both
  exp1 (fallacy types) and exp7 (session outcomes). Motivation: frozen-encoder
  "geometry" is largely a lexicon in disguise until proven otherwise.
- **Length-only control** — a probe on session step-count alone (length leakage
  is documented in related work; exp7 length-only AUC 0.561).
- **Format-tags control** — probe on harness markup counts only
  (`[tool_call]`, `[user]`, ...). Exp7: 0.602 AUC. Guards against riding
  harness house style.
- **Session-disjoint splits** — GroupKFold by session (never iid row CV).
  Established exp1 addendum: quiz source leakage (quizizz.com = 67% of LOGIC).
- **Harness-disjoint splits** — group splits by harness/benchmark/model. The
  gate for any transfer claim (currently impossible on our Exgentic extract,
  which kept no metadata; see ROADMAP P0).

## Core objects

- **Step** — one atomic logged act: an assistant text chunk, a tool call, or
  (when logged) a tool result. The unit of time. Stored as plain text.
- **Canonical session format** — the normalized log schema this project
  analyzes: ordered list of step texts + outcome + full metadata (harness,
  benchmark, model, timestamps). v1 = exp6 `data_prep.py` output **plus the
  metadata it currently drops** (defect, fixed in P0).
- **Session** `S = (s_1, …, s_T)` — the ordered step sequence from one agent
  run, with its **outcome label** (success/failure). Use "session" for the
  data object; reserve "trajectory" for its geometry (below).
- **Step embedding** `z_t = E(s_t)` — frozen-encoder vector of step t.
  Current encoders: nomic-embed-text (768-d), qwen3-embedding (4096-d).
  E is never fine-tuned in this project's working line.
- **Trajectory** — the path traced by `z_1, …, z_T` through a representation
  space. Plural, strictly: sessions trace trajectories.
- **Velocity / delta** `v_t = z_{t+1} − z_t` — step-to-step displacement
  (run = 1 step). Informal name "semantic speed"; in the paper prefer
  "displacement". Storing raw (unnormalized) vectors preserves leap magnitude.
- **Path shape** — the variable-length sequence `(v_1, …, v_{T−1})`;
  translation-invariant description of a trajectory. Two sessions with the
  same shape but different absolute positions share it.
- **Session summary / shape vector** `φ(S)` — any fixed-length feature vector
  summarizing a session. Current instances: `[centroid; final]` (exp7 static,
  1536-d for nomic), mean-velocity, PCA-reduced variants.
- **Outcome label** `y ∈ {0,1}` — realized success/failure of the session.

## Spaces

- **Semantic space (S-space)** — the frozen encoder's raw embedding space,
  read as a *content/topic* similarity space. Everything measured so far
  (exp1 probes, exp7 centroid probes) lives here by default.
- **Agentic space (A-space)** — the representation we want: an embedding of
  sessions/step-sequences in which geometric relations correspond to
  *process* relations (steady progress, stalling, looping, thrash, recovery,
  derailment) and are *invariant to harness formatting*. A-space is trained
  (or distilled), not assumed: the format-crystallization objective (P2) is
  the attempt to manufacture it. Until P2 passes its gate, "we measured a
  signal in A-space" is UNTESTED and must not be claimed; static-probe
  results are semantic-space results.
- **A-space coordinates are low-dimensional by design** — the working target
  is 8–64 dims (evidence: real-session state AUC 0.757 at 8 PCA dims vs 0.900
  full; synthetic shape elbow ≤ 8 dims, exp5b). Not yet tested under format
  shift.

## The signal the project is about

- **Trepidation** `τ(S_{1..t})` — the live, calibrated scalar (or small vector)
  output by the monitor from an *in-progress* session prefix, meaning
  "resemblance to trajectories that went badly." Candidate operationalizations:
  (1) linear probe score on the prefix summary φ (works today),
  (2) distance from the predicted destination to the success region,
  (3) cross-candidate divergence (K sampled continuations disagree in
  representation space). For intervention use, report **calibration
  (reliability curves), not ranking metrics** — a threshold is only useful if
  τ is an honest probability-like score.
- **Trembling** (working word, not load-bearing) — within-trajectory variance
  of τ across successive prefixes; intended as a stall/oscillation indicator.
- **Velocity increment** — the AUC delta from *adding* velocity features to
  content-matched features. Measured: **+0.000 on 2,000 real Exgentic
  sessions** (FALSIFIED [Exgentic, nomic, outcome-prediction]) — hence the P0
  re-stream and the live-run redesign before the shape question is reopened.
- **Lexical equivalent** — the TF-IDF baseline matching a given geometric
  probe. If a probe's AUC ≈ its lexical equivalent, the "geometric signature"
  claim is not earned for that encoder/task. nomic on fallacy-type
  classification is currently at its lexical equivalent; qwen3 is ~+12 points
  above it; session outcome probes are ±0.02 around it.

## Steering vocabulary (the three implementations)

- **Pre-submit gate (Impl 1)** — before the agent commits text, compute τ; if
  τ > threshold, force a retry/different approach. Success metric: Δ(task
  success rate) vs no-gate under matched budget, not AUC.
- **Candidate ranking (Impl 2)** — at a decision point, score K candidate next
  actions, choose the lowest trepidation. Ranking need not be calibrated;
  must beat random candidate choice under matched budget.
- **Waymarkers / centroid template navigation (Impl 3)** — precomputed target
  points (centroids) in A-space forming a reference polyline through a task
  archetype ("yellow brick road"); steering keeps the live trajectory within a
  neighborhood of the road. No embedding decoder exists or is needed —
  navigation happens by reroll/re-prompt toward nearby logged exemplars, not
  by decoding. Known hazard: content confound — a topic centroid is not a
  process template; templates must be content-controlled like everything else.

## Training vocabulary (P2)

- **Format-crystallization objective** — any alignment training that pulls
  together A-space encodings of the *same underlying episode* rendered by
  different harnesses/formats (pairs via contrastive loss or
  cross-correlation/Barlow-Twins style objectives), while pushing apart
  different outcomes. The bet: "actions crystallize, formatting blurs."
- **Crystallization metrics** — cross-harness step/session retrieval accuracy,
  AUC drop under harness-disjoint transfer, before vs after crystallization
  training. These numbers decide whether A-space exists.
- **Harness** — the scaffolding system that formats a run's log (model +
  scaffold + serializer: OTel spans, chat-JSONL, markdown logs, …).
- **Rendering** — one harness's serialization of an episode.

## Falsification protocol (standing)

1. State the claim with its scope (space, dataset, encoder, task type).
2. Attach the mandatory controls: content, length, tags, session-disjoint
   splits — plus harness-disjoint splits for any transfer claim.
3. Pre-register the rejection threshold in the experiment README before
   running (exp7's verdict block is the house style).
4. Report the negative in RESULTS.md as a first-class result. The project's
   two most valuable artifacts are negative results (exp2 chunking, exp7
   velocity-dominance) precisely because they had controls attached.