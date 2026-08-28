# Why JEPA Instead of a Linear Probe — The Normalization Argument

## The problem with frozen embeddings

Exp1 showed that frozen embeddings (nomic-embed-text, qwen3-embedding) carry fallacy signal at the whole-statement level (AUC 0.87-0.99). Exp2 showed that chunking into reasoning steps DOESN'T HELP — it actually hurts (multiclass accuracy drops from 32.3% to 25.1%).

Why? Because frozen embeddings are trained for retrieval similarity, not reasoning structure normalization. Two ways of expressing the same reasoning step:

- "All birds can fly, therefore penguins can fly"
- "Since every avian species possesses the capability of flight, it follows that penguins share this ability"

...embed differently because the encoder responds to surface vocabulary. When you chunk a reasoning chain into steps, each fragment has less context and more surface noise, so the embedding becomes less informative about the underlying reasoning structure.

## What JEPA + Barlow Twins does differently

JEPA training with Barlow Twins anti-collapse learns an embedding space where:

1. **Invariance**: Different surface expressions of the same reasoning map to the same region (diagonal of cross-correlation matrix → 1)
2. **Redundancy reduction**: Each embedding dimension carries independent information (off-diagonal → 0)
3. **Trajectory prediction**: The predictor learns z_{t+k} = f(z_t, action), which forces the embedding space to preserve temporal structure

The encoder is TRAINED to normalize surface variation and expose reasoning structure. Frozen embeddings don't do this — they were trained for a different objective (retrieval/next-token-prediction).

## The progression

| Experiment | Encoder | Chunking | Signal | Limitation |
|---|---|---|---|---|
| Exp1 | Frozen (nomic/qwen3) | None (whole statement) | AUC 0.87-0.99 | Signal is holistic, detects surface patterns |
| Exp2 | Frozen (nomic) | Sentence/connective/clause | AUC 0.91-0.97 (no improvement) | Chunking loses context, surface noise dominates |
| Exp3 | JEPA-trained (Barlow Twins) | Step-level | Predicted: should exceed frozen | Needs training data + training compute |

## The reviewer answer

"Why JEPA instead of a linear probe on frozen embeddings?"

Answer: "Frozen embeddings lose signal when applied to reasoning fragments (Exp2: chunking hurts, not helps). This is because frozen encoders normalize for retrieval similarity, not reasoning structure. JEPA training with Barlow Twins learns an embedding space where surface variation is normalized (invariance term) and each dimension carries independent structural information (redundancy reduction). This recovers the step-level signal that frozen embeddings lose."

## Data sources for JEPA training

### Option A: Existing agentic trajectory datasets
- AgentTrove (1.7M trajectories, reward-labeled) — from large models (GPT-5.2, Claude), mostly successes
- Exgentic (10K runs, 6 benchmarks, success labels) — diverse tasks
- ISETrace (23K OS-agent trajectories, per-step success) — real tool execution

### Option B: Run a small model in sandbox, capture failure trajectories
- Run qwen3:0.6B or qwen2.5:1.5B on agentic tasks (coding, API verification, etc.)
- Small models will thrash: loop, retry, go down wrong paths, hallucinate
- Label each trajectory by outcome (did the task succeed?)
- These are REAL failure trajectories from a model that makes lots of mistakes
- Advantage: the trajectories are from the SAME model family that would use the predictor, so the embedding space is aligned
- Advantage: we control the task difficulty and can generate unlimited data

### Option C: Both
- Use AgentTrove/Exgentic for scale (large model trajectories, mostly success)
- Use sandbox-generated failures for negative examples (small model trajectories, mostly failure)
- Combined dataset has both success and failure trajectories from diverse sources

## Why Option B (sandbox generation) is the strongest for our claim

The claim is that the JEPA predictor can detect when reasoning is heading toward failure. To test this, we need:
1. Trajectories that FAIL (negative examples) — small model thrashing
2. Trajectories that SUCCEED (positive examples) — either large model traces or small model successes
3. The trajectories must be multi-step (5-20 steps) so the trajectory shape is visible

Running qwen3:0.6B on coding tasks with a test runner gives us:
- Multi-step trajectories (generate → test → fix → retest → fix → ...)
- Clear success/fail labels (tests pass or don't)
- The same model that would use the predictor in production
- Unlimited data generation capability

## The normalization experiment

To test whether JEPA training actually normalizes surface variation:

1. Take 100 reasoning steps, paraphrase each 3 ways (preserve logic, change surface form)
2. Embed all 300 with frozen encoder → measure pairwise distance between paraphrases
3. Train JEPA encoder on trajectory data
4. Embed same 300 with JEPA encoder → measure pairwise distance
5. If JEPA normalizes: paraphrase distance decreases
6. If it doesn't: JEPA isn't doing what we claimed

This is a concrete falsification test for the normalization argument.