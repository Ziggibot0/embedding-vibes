# Novelty Analysis — What's Actually New Here

## What already exists (be honest or reviewers will destroy you)

### The signal exists
- Microsoft (ACL 2026): late-step trajectory features predict correctness, AUC 0.87
- ICLR 2026: latent trajectory signals predict accuracy, 70% token reduction
- CraEG (ICML 2026): embedding-space crowding correlates with reasoning failure

### JEPA works for text
- LLM-JEPA: JEPA training objective for LLMs
- BERT-JEPA: JEPA reorganizes CLS embeddings
- UWM-JEPA: JEPA with belief-state latents (density matrices)

### Fallacy detection is hard for LLMs
- Jin et al. (2022): LLMs ~34% accuracy, structure-aware classifiers ~39.5%
- AID-LF (2025): decomposing into atomic steps helps

### Trajectory geometry in reasoning
- "Geometry of Reasoning" (2025): logical structure = higher-order geometry in representation space
- "Reasoning emerges from constrained manifolds" (2026): pathological dynamics = bad reasoning

## What does NOT exist (our actual contributions)

### 1. JEPA for trajectory PREDICTION (not training)
All existing text-JEPA work uses JEPA as a TRAINING OBJECTIVE to improve LLM representations.
Nobody uses JEPA to train a PREDICTOR that forecasts future trajectory states.
Microsoft used linear probes (supervised, static). We'd use JEPA (self-supervised, predictive).
**This is novel but incremental.** A reviewer might say "it's a linear probe with extra steps."

### 2. Trajectory aggregation across candidate trajectories
No existing work adds/superposes predicted trajectory embeddings from K candidates and interprets the resultant as confidence/trepidation.
Latent-Trajectory signals use individual trajectory metrics. Majority vote uses answer matching.
Aggregation in embedding space across reasoning candidates is genuinely new.
**This is a strong novelty claim.** Nobody has done this. The quantum cognition literature provides theoretical grounding but no implementation in LLM reasoning.

> **SCOPE NOTE (2026-08-28):** The wave-interference framing (constructive/destructive, phase/frequency) is TABLED — a separate project. This project keeps the aggregation mechanism but does not frame it as wave interference, and does not use the NPU.

### 3. Geometric fallacy detection (no reasoning)
Existing fallacy detection uses LLM reasoning or structure-aware classifiers.
Nobody has tried: embed the argument, check if its trajectory shape matches known fallacy shapes.
**Novel application.** The connection between "Geometry of Reasoning" (logical structure = geometry) and fallacy detection (fallacies = geometric deviations) has not been made operational.

### 4. ~~NPU as interference computation substrate~~ (TABLED)
~~Nobody has used the NPU for embedding-space operations during LLM inference.~~
~~**Novel hardware contribution.** But this is engineering, not science. A reviewer might not care.~~
**Removed from scope (2026-08-28).** The NPU is not used in this project. Compute runs on CPU/iGPU.

## Honest novelty assessment

| Contribution | Novelty | Risk |
|---|---|---|
| JEPA trajectory predictor | Medium — extends linear probe to learned predictor | Low (incremental but solid) |
| Trajectory aggregation | HIGH — no prior work | Medium (might not beat majority vote) |
| Geometric fallacy detection | HIGH — no prior work | High (fallacy shapes might not be distinct) |
| ~~NPU interference~~ (TABLED) | — | — |

## What a reviewer would attack

1. "JEPA predictor is just a learned linear probe. What does JEPA add over supervised prediction?"
   - Answer: self-supervised (no labels needed), multi-horizon prediction, Barlow Twins gives structured embedding space. But need ablation: JEPA vs supervised MLP vs linear probe.

2. "Trajectory aggregation is just weighted vector addition. Why is it novel?"
   - Answer: need to show it's MORE than weighted sum. The aggregation framing must provide additional signal beyond magnitude. If it's just Σ w_i z_i, a reviewer is right. (Note: the wave-interference framing that would add phase/frequency is tabled as a separate project.)

3. "Your fallacy detection results are on a small dataset. How do you generalize?"
   - Answer: need multiple datasets (Logic, AID-LF, 20K dataset) and multiple fallacy types.

4. "You use a frozen encoder. How do you know the encoder carries fallacy signal?"
   - Answer: gate experiment. But if it fails, this IS the failure mode.

5. "Distribution shift: you train on GPT-5.2 traces but test on Qwen. Why would it transfer?"
   - Answer: need to test both same-model and cross-model. If cross-model fails, that's a limitation.

## What would make this a STRONG paper (not just a publishable one)

1. **Show a fallacy shape that's interpretable.** Not just "AUC 0.7" but "circular reasoning traces literal loops in embedding space, here's the plot." A reviewer seeing a figure where begging-the-question makes a circle in PCA space would be convinced.

2. **Show aggregation beating majority vote by a clear margin in a specific regime.** Not overall, but in the hard cases (high disagreement, low confidence). "When 3/5 candidates agree, majority vote gets 60% but aggregation gets 80%."

3. **Show the predictor working at step 2, not step 5.** Microsoft's linear probe works at late steps. If our JEPA predictor works at EARLY steps (because it's predictive, not just classificatory), that's a real advantage.

4. **Show it on a small model.** Microsoft used large models. If our predictor helps a 4B model more than a 70B model (because the 4B needs more help), that's a useful finding for the small-model community.

## The contribution that would make reviewers care

The figure that would sell this paper: a 3D plot of CoT trajectories through embedding space, fallacious ones visibly looping or diverging, valid ones progressing smoothly toward the answer region. If that figure exists and is real, the paper writes itself.

---

## RE-SCOPE (2026-08-28, evening): novelty table after the audit session

Updated disposition of each claimed contribution:

| Claim | Was | Now | Evidence |
|---|---|---|---|
| JEPA trajectory predictor | Medium novelty | HOLD as P2 component; must beat frozen-probe + PCA-8 baselines (0.757–0.900 AUC) before it is a contribution | exp6 (predictor works, learned proj loses signal), exp7c (8-dim PCA strong) |
| Trajectory aggregation (K candidates → confidence/trepidation) | HIGH novelty | UNTESTED — promoted to the central signal ("trepidation"); now must be tested LIVE (K sampled continuations on real runs), not on frozen datasets | none yet on real runs |
| Geometric fallacy detection | HIGH novelty | PROVEN per-encoder with content caveat: qwen3 above lexical equivalent (+12pts), nomic AT it. Novelty now rests on the decomposition discipline (content vs geometry controls), not on "geometry detects fallacies" per se | exp1 + re-audit addendum |
| Cross-harness format invariance (A-space) | implicit wish | NEW central claim, UNTESTED: format-crystallization training + harness-disjoint gates (ROADMAP P2). No prior work does this for agent-session logs; also no prior work reports the content-control decomposition on agent trajectories | none yet — this is the paper's core testable novelty |
| Low-dim live monitor | implicit | NEW: 8-dim state summary 0.757 AUC (real sessions); early-warning ramp exists; calibration study pending (P1) | exp7c, exp7b prefix curve |
| ~~NPU interference~~ | TABLED (kept) | unchanged — `ideas/npu_wave_interference.md` | — |

Reviewer-attack section addition (2026-08-28): the strongest new attack is the
one we ran ourselves — "your embedding geometry matches TF-IDF." The paper must
lead with the decomposition (geometry vs content per encoder/task) and claim
only the margin above it. Second attack: "A-space is just TF-IDF/char-ngram
robustness in disguise" — answer: A-space gates are cross-harness *retrieval
and outcome transfer* tasks where lexical baselines are run side-by-side and
reported, win or lose.
