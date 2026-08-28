# Research Plan — Embedding Vibes

## Positioning

The core claim: **Logical fallacies and reasoning failures have geometric signatures in embedding space. A JEPA-trained predictor can learn these shapes without reasoning — detecting "vibes" about trajectory quality the way humans detect vibes about situations without articulating why.**

This sits at the intersection of three converging research threads:
1. **Reasoning trajectory geometry** (Microsoft ACL 2026, ICLR 2026) — proved the signal exists
2. **JEPA for text** (LLM-JEPA, BERT-JEPA, UWM-JEPA) — proved JEPA works for language
3. **Embedding geometry and reasoning quality** (CraEG, constrained manifolds) — proved geometry correlates with correctness

**Our novel contributions:**
- First JEPA predictor for CoT TRAJECTORY PREDICTION (not training, not vision)
- Trajectory aggregation across candidate trajectories (aligned/divergent = confidence/trepidation)
- Application to logical fallacy detection via geometric pattern matching (not reasoning)

> **SCOPE (2026-08-28):** No NPU. Wave-interference framing (constructive/destructive, phase/frequency) is TABLED as a separate project. Compute runs on CPU/iGPU only.

## What makes this a paper (not just an experiment)

### The gap
Microsoft (ACL 2026) proved that late-step trajectory features predict correctness with AUC 0.87 using LINEAR PROBES. They stopped there. They did not:
1. Learn a predictor that FORECASTS the trajectory destination (they classify current state, we predict future state)
2. Aggregate multiple candidate trajectories via trajectory aggregation (they use single-trajectory analysis)
3. Apply to fallacy detection (they use math reasoning)
4. Use JEPA as the training framework (they use supervised linear probes)

Latent-Trajectory Signals (ICLR 2026) proved hand-crafted signals work. They did not:
1. Learn the signals automatically via JEPA
2. Use trajectory aggregation for confidence
3. Apply beyond answer selection

### The contribution chain
1. **Existence proof**: Do logical fallacies have geometric signatures in embedding space? (gate experiment)
2. **Learned prediction**: Can a JEPA predictor forecast trajectory destination? (extends Microsoft's linear probe to learned predictor)
3. **Trajectory aggregation**: Does aggregating K candidate trajectories' predicted embeddings produce better signal than any individual trajectory? (novel mechanism)
4. **Fallacy detection**: Can the predictor detect fallacies without reasoning? (novel application)

## Research questions

### RQ1: Do logical fallacies have geometric signatures in embedding space?
- Embed 100 fallacious + 100 valid CoT traces step-by-step
- Measure: trajectory curvature, direction-change frequency, loop-back tendency, manifold dimensionality
- Hypothesis: fallacy traces have distinct geometric shapes (e.g., circular reasoning traces literal circles in embedding space)

### RQ2: Can a JEPA predictor forecast trajectory destination?
- Train predictor P(z_t, action) → z_{t+k} on AgentTrove trajectories with Barlow Twins anti-collapse
- Compare against Microsoft's linear probe (AUC 0.87) as baseline
- Hypothesis: learned predictor outperforms linear probe, especially at earlier prediction horizons

### RQ3: Does trajectory aggregation improve over single-trajectory prediction?
- Generate K candidate CoT paths, predict each trajectory, aggregate
- Compare: single-trajectory prediction vs K-trajectory majority vote vs K-trajectory aggregation
- Hypothesis: aggregation captures alignment/divergence patterns that majority vote misses

### RQ4: Can the predictor detect fallacies without reasoning?
- Train on fallacy-labeled CoT (Logic dataset, AID-LF, 20K fallacy dataset)
- Evaluate: fallacy detection accuracy vs LLM-based detection (~34% from Jin et al.)
- Hypothesis: geometric pattern matching beats LLM reasoning for fallacy detection

## Methodology

### Phase 1: Gate experiment (RQ1)
**Data**: Logic dataset (3,761 labeled fallacies), AID-LF, 20K fallacy dataset
**Encoder**: frozen Qwen3-14B late-layer hidden states (or nomic-embed-text for initial cheap test)
**Analysis**:
- Extract per-step embeddings for each CoT trace
- Compute trajectory metrics: cumulative angular change, path length, loop-back (autocorrelation), manifold dimensionality (intrinsic dim estimate)
- Statistical test: do fallacy and valid trajectories have significantly different geometric distributions? (KS test, permutation test)
- Visualization: PCA/t-SNE projection of trajectory shapes, colored by fallacy type
**Output**: Plot of trajectory geometry by fallacy type. If shapes differ → green light.

### Phase 2: Predictor training (RQ2)
**Data**: AgentTrove (1.7M trajectories with reward labels), Exgentic traces (10K)
**Architecture**:
- Encoder: frozen Qwen3 (target encoder, stop-gradient + EMA)
- Predictor: small transformer (4 layers, 256-dim, ~5M params)
- Action embedding: embed tool-call text with same encoder
- Loss: L1(z_pred, z_target) + λ * BarlowTwins(z_pred, z_target)
**Training**: 
- Extract (z_t, action_text, z_{t+k}) triples from trajectory data
- k = {1, 3, 5} (multi-horizon, inspired by VLWM variable-length prediction)
- Anti-collapse: Barlow Twins cross-correlation → identity matrix
**Evaluation**:
- Compare trajectory destination prediction accuracy vs Microsoft linear probe (AUC 0.87)
- Measure: at what step t can we predict final correctness? (earlier = better)

### Phase 3: Trajectory aggregation (RQ3)
**Setup**:
- Generate K=5 candidate CoT paths from base model (temperature sampling)
- Predict each trajectory's destination embedding z_{t+k}
- Aggregate: resultant = Σ w_i * z_i (weighted sum, weights from predictor confidence)
- Aligned: trajectories → large resultant magnitude → high confidence vibe
- Divergent: opposing trajectories → small resultant → trepidation
**Baselines**: majority vote, LT signals (ICLR 2026), single-trajectory prediction
**Evaluation**:
- Answer selection accuracy with K candidates
- Token efficiency (can we stop early on high-confidence vibes?)
- Compare aggregation vs majority vote vs LT signals

### Phase 4: Fallacy detection (RQ4)
**Data**: Logic dataset (fallacy labels), AID-LF (atomic instructions), 20K fallacy dataset
**Setup**:
- Embed fallacious and valid arguments step-by-step
- Train predictor on labeled fallacy trajectories
- Evaluate: can predictor classify fallacy type from trajectory shape alone?
**Baselines**: LLM zero-shot (~34% from Jin et al.), structure-aware classifier (39.5%)
**Hypothesis**: geometric pattern matching > LLM reasoning for fallacy detection

### Phase 5: ~~NPU implementation~~ (TABLED)
**Removed from scope (2026-08-28).** The NPU is not used in this project. Aggregation runs on CPU/iGPU. The wave-interference/NPU angle is a separate project.

## Paper outline

**Title**: "Embedding Vibes: Predicting Reasoning Trajectory Quality via JEPA and Trajectory Aggregation in Representation Space"

**Abstract**: LLMs generate locally coherent reasoning that globally fails. We show that reasoning failures have geometric signatures in embedding space — logical fallacies trace trajectories with distinct shapes that a predictor can learn without reasoning. We train a JEPA predictor with Barlow Twins anti-collapse to forecast trajectory destinations in representation space, and aggregate multiple candidate trajectories (aligned = confidence, divergent = trepidation). On [benchmark], our predictor achieves [X] AUC for correctness prediction at step [Y], outperforming linear probes (0.87) and majority voting. On fallacy detection, geometric pattern matching achieves [X]% accuracy vs [34]% for LLM zero-shot.

**Sections**:
1. Introduction — the big-picture gap in LLMs, vibes as geometric pattern detection
2. Related Work — reasoning trajectories (Microsoft), JEPA for text, embedding geometry, fallacy detection
3. Method — JEPA predictor, Barlow Twins, trajectory aggregation formulation
4. Experiments — RQ1-RQ4 with results
5. Analysis — what shapes do fallacies make? what does the predictor learn?
6. Discussion — limitations, when vibes fail
7. Conclusion

## Timeline (honest estimate)

- Phase 1 (gate): 1-2 weeks. If it fails, paper is dead. Cheap.
- Phase 2 (predictor): 2-3 weeks. Training is the bottleneck (data prep from AgentTrove).
- Phase 3 (aggregation): 1-2 weeks. Cheap once predictor works.
- Phase 4 (fallacy): 1-2 weeks. Depends on labeled data quality.
- Writing: 2-3 weeks.
- Total: 3-4 months, part-time.

## Target venues

- **ICML / NeurIPS** — if aggregation + fallacy results are strong. Main track.
- **ICLR** — if the method contribution (JEPA for trajectory prediction) is clean.
- **EMNLP / ACL** — if the fallacy detection results dominate.
- **Workshop** — ICML Workshop on Reasoning, NeurIPS Workshop on Efficient LLM Inference

## Risks and mitigations

1. **Fallacy traces don't have distinct geometry** → gate experiment catches this early. Mitigation: try multiple encoders, multiple layers, multiple problem types.
2. **JEPA predictor doesn't beat linear probe** → linear probes are already strong (AUC 0.87). Mitigation: the value may be in EARLIER prediction (step 2 vs step 5) and trajectory aggregation, not raw AUC.
3. **Trajectory aggregation doesn't add signal over majority vote** → this is a novel contribution, so this would weaken the paper. Mitigation: ablation shows when aggregation helps (high disagreement cases) vs when it doesn't.
4. **Distribution shift from large-model CoT to our model's CoT** → generate our own CoT traces with Qwen3-14B, label by test pass/fail.