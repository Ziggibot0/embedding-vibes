# Related-Work Addendum — 2026-08-28 scan (failure prediction on agent trajectories)

This scan focused on our direct application space: early failure/outcome
prediction for LLM agents from trajectory signals. Two papers are CLOSE —
one is nearly our concept realized in robotics.

## 1. Foresight (arXiv 2606.23085) — SAME IDEA, ROBOTICS — READ CLOSELY

Failure detection for long-horizon robotic manipulation using latents from an
**action-conditioned world model (V-JEPA 2-AC)**. Mismatch between predicted
and actual latent = failure signal. Trained ONLY on trajectory-level
success/failure labels (no per-step annotation). Causal sequence model over
latent tokens → failure score. Conformal-prediction thresholds.

- **This validates our core mechanism** ("predict latent, compare prediction
  vs reality, mismatch = trepidation") — it works, on real robots.
- **It also narrows novelty**: the "foresight-as-failure-signal" mechanism is
  NOT ours to claim. Our claim must be the REPRESENTATION (relative-delta
  velocities of text-embedding trajectories) + the LLM-agent application +
  format invariance across harnesses.
- Transfer claim they make ("policy-agnostic monitoring") parallels our
  format-invariance goal — we can cite as supports-the-thesis.

## 2. When Evidence is Sparse (arXiv 2606.05414) — CLOSEST ON LLM AGENTS

Weakly-supervised EARLY FAILURE ALERTING on LLM-agent trajectories
(evaluated on **AppWorld**, ALFWorld — same benchmark family as our Exgentic
data). Attention-based MIL failure predictor over prefix embeddings with
trajectory-level labels; learned stopping policy with accuracy–earliness
tradeoff selected at inference (no retrain per operating point).

- Their representation: frozen encoder over the PREFIX + attention pooling.
  **Not velocities, not deltas, no forward prediction.**
- We must cite and differentiate: our contribution is the relative-delta /
  velocity shape representation + JEPA forecast + whole-path shape as a
  first-class object, vs their prefix-classification.
- Their methodological rigor (sparse turn-level evidence, accuracy–earliness
  protocol) is a good template for our exp7 evaluation.

## 3. Trajectory Graph Copilot / Graph Debugger (arXiv 2607.27443)

Pre-execution error diagnosis: builds probabilistic graph over historical
trajectories, GNN flags actions that frequently lead to failure; +14.69% pass
ratio. Discrete/graph of typed actions — structurally similar goal, different
representation (no embedding-velocity geometry).

## 4. Confidence-trajectory thesis proposal (ACL 2026 SRW)

"Typed confidence trajectories" over tool-use agents; STL temporal pattern
mining over per-step confidence predicts success/failure; early-exit warnings.
Uses UQ/confidence signals, not embedding geometry. Notable: their SVD analysis
found tool-use trajectory state spaces are LOW-RANK (effective rank 3-7) —
independent support for our exp5b finding that ~8 dims suffice.

## 5. World models for LLM agents (WorldEvolver 2606.30639, ProPlay 2606.12780)

Foresight for planning via world models with episodic/semantic memory and
reliability-record embeddings on procedure graphs. Related to the pilot's
PLANNER role ("pilot" framing) — but discrete/procedural, not latent-geometric.

## 6. Latent Reasoning via Sentence Embedding Prediction (arXiv 2505.22202)

Autoregressive prediction of next-SENTENCE embeddings (latent reasoning in
embedding space, half the FLOPs of CoT). Conceptually adjacent to JEPA-on-
text-trajectories; useful citation for "reasoning in embedding space is
viable."

## 7. Practitioner signal: structural retrieval for failure prediction (episodiq, dev.to)

Retrieval over trajectories represented as sequences of TYPED cluster tokens
(embed→cluster→discrete tokens) beats cosine-RAG on averaged prefixes (AUC
0.705 vs 0.58-0.60 plateau) for failure prediction on SWE-bench trajectories.
**Directly supports "averaged content loses signal; transition structure
carries it"** — an empirical echo of exp5 from a different lab/practitioner.

⚠️ **Methodological warning from that post**: a length-only classifier reached
AUC 0.66 on their data — length leakage confounds naive global AUC. **exp7
must include a length-only baseline** (n_steps as sole feature). exp3/exp5
sessions were all exactly 6 steps so were safe by construction; the 2000
Exgentic sessions are NOT length-matched — this control is mandatory.

## 8. Position papers supporting the frame

- "LLM Reasoning Is Latent, Not the Chain of Thought" (arXiv 2604.15726) —
  argues reasoning should be studied as latent-state trajectory formation.
  Cites our entire framing.
- "LLM Reasoning as Trajectories" (arXiv 2604.05655, already in related_work)
  — unchanged: our static-vs-delta comparison extends it.

## Updated novelty statement (what remains OURS after this scan)

NOT claimable (prior work exists):
- world-model latent mismatch as failure signal (Foresight, robotics)
- early failure alerting on LLM agent trajectories (2606.05414)
- trajectory-geometry predicts correctness (Microsoft 2604.05655)
- low-rank / compressed trajectory state spaces (their SVD; our exp5b)

STILL OURS:
1. **Relative-delta (velocity) trajectory representation** of text-embedding
   sequences as the discriminative object — translation-invariant shapes vs
   static positions (exp5: beats static 0.82→0.97 nomic).
2. **Whole-path shape as a first-class learned object** (trajectory encoder →
   shape vector) with joint encoder+predictor training for LLM agents.
3. **From-scratch task-trained text encoder** for reasoning trajectories
   (not downstream of any frozen embedding model) — nobody does this for
   agent-log geometry, to our knowledge.
4. Format invariance across harnesses as an explicit design goal (trained
   multi-harness, cf. Foresight's policy-agnostic finding in robotics).

Framing rule going forward: cite Foresight as "our mechanism, proven in
robotics — we contribute the text-trajectory representation and the LLM-agent
instantiation." Do not claim the mechanism alone as novel.