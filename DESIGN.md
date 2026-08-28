# embedding-vibes — design notes

## Scope (2026-08-28)

- **No NPU.** The XDNA 2 NPU is out of scope for this project. All compute runs on CPU/iGPU.
- **Wave interference is TABLED.** The "vector addition of embeddings IS interference" idea is a separate project, not part of embedding-vibes. This project does not use interference aggregation; it uses trajectory prediction + aggregation on CPU/iGPU.

## The core claim

Logical fallacies and reasoning failures have geometric signatures in embedding space. A predictor can learn these shapes without understanding logic — the same way humans get "vibes" about something being off without articulating why.

## What this is NOT

- Not a logic checker. It doesn't verify arguments.
- Not a reasoning model. It doesn't think.
- Not a PRM (process reward model). PRMs score individual steps in token space. This scores the trajectory's predicted destination in embedding space.

## What it IS

A pattern detector. "This trajectory's shape resembles trajectories that failed." Swirls of numbers that arrange themselves into shapes that a different mechanism can reason about.

---

# The Pilot Architecture

A **pilot** model: given the first few steps of an agentic flow, it predicts the path most likely to succeed (and which paths to avoid), and the agentic model follows that path without deviating too far. This is a shift from *monitoring* (watching for failure) to *planning* (steering toward success).

## Core design decisions

### 1. Shapes are stored as RELATIVE relationships, not absolute positions

We do **not** store a path as a list of embeddings. We store it as a list of **deltas** (relative relationships):

```
path = [ z_1, z_2, ..., z_T ]          # absolute positions (NOT what we compare)
shape = [ z_2−z_1, z_3−z_2, ..., z_T−z_{T−1} ]   # relative deltas (what we compare)
```

Because each step is one time increment, each delta is a **velocity** — rise/run with run = 1. This is the "semantic speed" framing: a small delta = slow semantic movement, a large delta = fast movement.

**Why deltas:** translation invariance. Two paths that are the *same shape* but located in *different regions* of embedding space produce the *same* delta sequence. "Same shape, different place → still similar." This is the property that makes shape comparison meaningful.

**Two refinements:**
- **Keep the anchor.** Deltas alone lose *where* the path is. In reasoning, absolute location matters (what topic/state am I in). So: the shape encoder compares deltas (translation-invariant), but the predictor still conditions on the current absolute state. Position + velocity, not velocity alone.
- **Keep the magnitude.** Store raw delta *vectors*, not normalized unit vectors. A big semantic leap and a small one are semantically different; normalizing would erase that. Only normalize if we specifically want scale invariance (we don't).

### 2. The whole path is a first-class object

A **trajectory encoder** maps the full path (all step embeddings) to one fixed-length **shape vector**. This is what makes "compare shapes" operational:
- Similar-shaped paths map close together (Barlow Twins / contrastive loss lives on the shape vectors).
- The predictor can be conditioned on a target shape.
- At inference, generate candidate continuations, predict each one's full shape, and pick the one whose shape is closest to the success shapes.

This is the cleanest novelty claim: whole-trajectory shape encoding for agentic path selection.

### 3. Encoder and predictor are trained JOINTLY

The encoder and the JEPA pilot predictor train in the **same loop**, from the same experiences, so they coordinate: the encoder learns to produce shapes the predictor can actually work with, and "the shapes we say should be the same" become similar.

**Collapse risk is real here.** Joint training is exactly where encoder + predictor can collapse to a constant. Guardrails are mandatory:
- **Barlow Twins** redundancy reduction (cross-correlation → identity; off-diagonal → 0) packs independent info per dimension and blocks collapse.
- **Stop-gradient / EMA target** on the target encoder prevents the predictor from trivially matching a constant.

### 4. Multi-step lookahead

The forward model predicts 2, 3, 4 steps ahead (multi-horizon k = 1, 3, 5). Given the first few steps, it predicts the *rest of each candidate's shape*, then the shape encoder scores it. This is how the pilot "already knows the rest of each shape given the first few steps."

### 5. Action selection = rank candidates, don't generate

The predictor outputs a target embedding; we pick the candidate action whose embedding is closest to the predicted-success embedding. The pilot does not generate actions — it **ranks a pool of candidate next steps** by distance to the predicted-success shape. This is what makes "predict next action" actually steerable.

## Training data

- **Format invariance:** train on agentic workflows from MANY different LLM harnesses so the session-log format doesn't matter and overfitting is prevented. The model doesn't need to know what anything means — it predicts the next action's embedding.
- **Scale:** AgentTrove (1.7M reward-labeled trajectories), Exgentic (10K runs), ISETrace (23K), AgentTrace (1.4K). Use these for representation learning / format invariance.
- **Transferable signal:** our own model's trajectories are non-optional for the outcome signal that actually matters (large-model trajectories don't transfer to qwen35moe).

### Multiple-choice / contrastive objective

For each step, generate alternative actions (local LLM), then give the model a multiple-choice test: pick the action that leads down the path that succeeded.

**Label grounding (critical):** the positive must be the action that ACTUALLY happened in a successful trajectory; alternatives are negatives. If we just say "pick the best," the model learns to pick whichever alternative "looks most like the training distribution" rather than the one that actually succeeds. Ground the positive in a real successful continuation.

**Cheap negatives:** sample hard negatives from OTHER trajectories on the same task (no generation needed). Generate alternatives only for a subset, or only near decision points.

## Inference / steering

1. Given the first few steps, the forward model predicts the rest of each candidate's shape.
2. The shape encoder scores each candidate's full predicted shape.
3. Pick the candidate whose shape is closest to the success shapes (and farthest from failure shapes).
4. The agentic model follows that path without deviating too far.

**Early-warning / trepidation:** run the forward model from the first few steps, predict where it's heading, compare against success/fail centroids. High divergence = trepidation = early abort.

## Gate experiment (unchanged)

Take 100 CoT traces with known fallacies + 100 valid CoT traces. Embed each step. Plot the geometry:
- Do fallacy trajectories loop back on themselves? (circular reasoning literally circular?)
- Do they have different curvature?
- Different direction-change frequency?
- Different "frequency" of oscillation through embedding space?

If fallacy trajectories look geometrically different → the idea has legs.
If indistinguishable → embeddings don't carry the signal → dead.

**Cheapest validation first:** compute hand-crafted shape features (velocity, curvature, loop-closure) on the existing 90 exp3 sessions and test whether they separate success from failure better than the static probe. If shape signal doesn't exist on data we already have, no amount of custom encoding fixes it.

## Steering options (ranked by tractability)

1. **Beam search in embedding space**: generate K candidates, predict each trajectory, keep the one heading toward success cluster. No intervention in generation. Easiest.
2. **Flag and re-prompt**: "your reasoning appears to be heading toward a known failure pattern, reconsider." Simple, no gradient flow needed.
3. **Logit steering**: penalize tokens toward success-cluster direction. Needs gradient flow into base model. Hard with frozen encoder.
4. **Full steering**: intervene in generation at each step. Hardest, unsolved.

## Open questions

- Does embedding geometry actually distinguish fallacy from valid reasoning? (gate experiment)
- Does the signal transfer across models? (GPT-5.2 failure patterns vs qwen35moe failure patterns)
- Does trajectory aggregation add signal over static vector comparison? (needs ablation)
- Does the shape signal exist on the 90 existing exp3 sessions? (cheapest gate)
- What is the right shape-vector dimension? (task-AUC vs dimension curve — 64 is a hypothesis, not a law)
- What frequency/phase means for a trajectory through embedding space (not yet defined; tied to the tabled wave-interference project)
