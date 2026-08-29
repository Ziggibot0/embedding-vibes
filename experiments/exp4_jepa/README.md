# JEPA Training Architecture — Embedding Vibes

> **STATUS (2026-08-28):** This document is the ORIGINAL design spec (frozen base encoder + learned head). The current direction has moved:
> - **exp5/exp5b** validated the core mechanism differently: raw relative-deltas (translation-invariant velocities) beat static features, and the dimensional elbow is ≤ 8 dims.
> - **exp6** (joint encoder + predictor on frozen embeddings) showed the learned 64-dim Barlow-Twins projection LOSES signal vs raw deltas.
> - **exp6b** (`exp6_joint_jepa/train_from_scratch.py`) is the current build: a FROM-SCRATCH text encoder trained jointly with the JEPA predictor (MLM + Barlow Twins + JEPA), on multi-harness data (22K Exgentic+AgentTrove sessions), iGPU via torch-directml.
> - This spec remains as background/reference; see DESIGN.md for the current pilot architecture.

## Overview

A JEPA (Joint Embedding Predictive Architecture) that learns to predict future embedding states of reasoning trajectories. The encoder normalizes surface variation (Barlow Twins), the predictor forecasts where reasoning is heading (z_{t+k}), and the whole system runs on top of a frozen base encoder with a learned lightweight head.

## Architecture diagram

```
                    TRAINING
                    ========

  reasoning step t          reasoning step t+k
  ---------------          ------------------
  |  text_t      |          |  text_{t+k}   |
  ---------------          ------------------
        |                          |
        v                          v
  +-----------+              +-----------+
  | FROZEN    |              | FROZEN    |   (EMA copy, stop-gradient)
  | BASE      |              | BASE      |
  | ENCODER   |              | ENCODER   |
  | (qwen3-   |              | (qwen3-   |
  |  embedding)|              |  embedding)|
  +-----------+              +-----------+
        |                          |
   z_t (4096d)                z_target (4096d)
        |                          |
        v                          |
  +-----------+                    |
  | LEARNED   |                    |
  | PROJECTOR |                    |
  | (MLP)     |                    |
  +-----------+                    |
        |                          |
   e_t (256d)                e_target (256d)
        |                          |
        v                          |
  +-----------+                    |
  | PREDICTOR |---- action --------|
  | (small    |     embedding      |
  |  xformer) |                    |
  +-----------+                    |
        |                          |
   e_pred (256d)                   |
        |                          |
        v                          v
  +-----------------------------------+
  |           LOSS                    |
  |  L = L1(e_pred, e_target)        |   prediction loss
  |  + λ * BarlowTwins(e_pred,       |   anti-collapse
  |                     e_target)    |
  |  + β * L1(e_pred, e_target_k2)   |   multi-horizon (k=1,3,5)
  +-----------------------------------+
```

```
                    INFERENCE
                    =========

  current reasoning state
  ------------------------
  |  text_t              |
  ------------------------
        |
        v
  +-----------+     +-----------+
  | FROZEN    |     | LEARNED   |
  | BASE      |--->| PROJECTOR |---> e_t (256d)
  | ENCODER   |     +-----------+
  +-----------+          |
                         v
                   +-----------+
                   | PREDICTOR |--- e_pred (256d) = where this is heading
                   +-----------+
                         |
                    +----+----+
                    |         |
              e_pred     trepidation
              direction   = variance across
              vs success  k-horizon predictions
              centroid    (high spread = uncertain)
```

## Components

### 1. Frozen Base Encoder
- **Model**: qwen3-embedding (7.6B, 4096-dim output)
- **Why frozen**: we don't want to retrain an 8B model. We want a lightweight learned head on top.
- **Role**: produces a raw embedding of each reasoning step's text
- **Output**: 4096-dim vector per step
- **Access**: via Ollama API (localhost:11434) for data prep, via direct llama.cpp FFI for real-time inference

### 2. Target Encoder (EMA copy, stop-gradient)
- **Model**: same as base encoder, but weights are an exponential moving average of the base encoder
- **Why EMA + stop-gradient**: prevents representation collapse. The target is a slowly-updated version of the encoder, and gradients don't flow through it. This is the I-JEPA / BYOL recipe.
- **In practice**: since the base encoder is FROZEN, the EMA target is identical to the base encoder. The stop-gradient is still important — it prevents the projector from collapsing the representation by matching a constant target.
- **If we later unfreeze**: the EMA becomes meaningful (target encoder lags behind the online encoder, preventing collapse)

### 3. Learned Projector (the "encoder head")
- **Architecture**: 2-layer MLP
  - Layer 1: Linear(4096 → 512) + LayerNorm + GELU
  - Layer 2: Linear(512 → 256) + LayerNorm
- **Output**: 256-dim projected embedding e_t
- **Why 256-dim**: compressed enough to force structure learning, large enough to preserve signal. Can be tuned (try 128, 256, 512).
- **Role**: maps the frozen encoder's output into a "reasoning structure space" where surface variation is suppressed
- **Parameters**: ~2.1M (4096*512 + 512*256 + norms)
- **This is the Barlow Twins invariance/redundancy target**: the projector learns to map different surface expressions of the same reasoning to the same point

### 4. Predictor (the "world model")
- **Architecture**: 2-layer transformer
  - Input: e_t (256d) + action_embedding (256d, concatenated or cross-attention)
  - Layer 1: Linear(512 → 512) + LayerNorm + GELU
  - Self-attention: 4 heads, 512-dim, causal mask
  - Layer 2: Linear(512 → 256) + LayerNorm
- **Output**: predicted e_{t+k} (256-dim)
- **Multi-horizon**: predict k=1, 3, 5 simultaneously (3 output heads or shared trunk + 3 heads)
- **Parameters**: ~1.5M
- **Role**: given current reasoning state and what action is being taken, predict where the reasoning will be in k steps

### 5. Action Embedding
- **For agentic trajectories**: embed the tool call text (e.g., "run_tests", "search_docs", "write_code") with the same frozen base encoder + projector
- **For CoT trajectories**: embed the next reasoning step's first sentence (or the connective that leads to it) as the "action"
- **Output**: 256-dim, same space as e_t
- **Why**: the predictor needs to know WHAT the agent is about to do, not just WHERE it currently is

### 6. Trepidation Module
- At inference, run the predictor for k=1, 3, 5
- Trepidation = variance of predicted directions across horizons
  - If k=1, 3, 5 all predict similar destination → low trepidation (confident)
  - If predictions diverge → high trepidation (uncertain)
- Also: distance of predicted destination to "success centroid" vs "failure centroid"
  - Success centroid: mean embedding of known-successful trajectories' final states
  - Failure centroid: mean embedding of known-failed trajectories' final states
- Trepidation score = weighted combination of (directional variance + failure-centroid proximity)

## Loss function

```
L_total = L_pred + λ_bt * L_barlow + β * L_multi

L_pred = ||e_pred_k1 - e_target_k1||_1     (L1, k=1 horizon)
L_multi = ||e_pred_k3 - e_target_k3||_1    (k=3 horizon)
        + ||e_pred_k5 - e_target_k5||_1    (k=5 horizon)

L_barlow = BarlowTwins(e_pred, e_target)
         = ||C - I||_F^2
  where C = cross-correlation matrix between e_pred and e_target
  C_ij = correlation between dim i of e_pred and dim j of e_target across batch
  Diagonal → 1 (invariance: predicted matches target per-dimension)
  Off-diagonal → 0 (redundancy reduction: dims carry independent info)
```

**Hyperparameters**:
- λ_bt = 0.5 (Barlow Twins weight — tune between 0.1 and 1.0)
- β = 0.3 (multi-horizon weight — tune between 0.1 and 0.5)
- L1 vs L2: start with L1 (more robust to outliers), try L2 as ablation

## Training data format

Each training example is a trajectory:

```json
{
  "trajectory_id": "traj_001",
  "outcome": "success",  // or "failure"
  "task": "write a function that reverses a list",
  "steps": [
    {"text": "I need to write a function that reverses a list", "action": "think"},
    {"text": "Let me use Python's slice notation [::-1]", "action": "write_code"},
    {"text": "def reverse(lst): return lst[::-1]", "action": "run_tests"},
    {"text": "Tests passed: all 5 test cases correct", "action": "done"},
  ]
}
```

For each trajectory, generate training triples:
- (z_t, action_t, z_{t+1}) for k=1
- (z_t, action_t, z_{t+3}) for k=3 (if trajectory has ≥4 steps)
- (z_t, action_t, z_{t+5}) for k=5 (if trajectory has ≥6 steps)

## Training procedure

### Phase 1: Data preparation (offline, one-time)
1. Collect trajectories (AgentTrove + sandbox-generated failures)
2. For each step in each trajectory, embed with frozen qwen3-embedding → 4096-dim
3. Store as .npy arrays: (n_trajectories, max_steps, 4096)
4. Store outcome labels and action labels

### Phase 2: Train projector + predictor (the actual JEPA training)
1. Initialize projector (MLP, random init)
2. Initialize predictor (transformer, random init)
3. For each batch of trajectories:
   a. Sample (z_t, action, z_{t+k}) triples
   b. Project z_t → e_t via projector
   c. Project z_{t+k} → e_target via EMA projector (stop-gradient)
   d. Embed action → a_t via frozen encoder + projector
   e. Predict e_pred = predictor(e_t, a_t)
   f. Compute L_pred + L_barlow + L_multi
   g. Backprop through projector and predictor only (not frozen encoder)
4. Update EMA projector: θ_ema = 0.99 * θ_ema + 0.01 * θ_projector
5. Repeat for ~50 epochs or until loss converges

### Phase 3: Evaluate
1. Embed 50 successful + 50 failed trajectories with trained projector
2. Train a linear probe on the projector outputs → does it beat frozen-embedding probe?
3. Run predictor on partial trajectories → can it predict outcome before completion?
4. Trepidation test: do failed trajectories have higher trepidation scores?
5. Paraphrase test: does the trained projector reduce distance between paraphrased steps?

## Hardware plan

- **Training**: projector (~2.1M params) + predictor (~1.5M params) = ~3.6M total trainable
- Runs on CPU in minutes per epoch (small model, pre-computed embeddings)
- No GPU needed for training — the heavy compute (embedding) is done offline
- **Inference**: projector + predictor forward pass = <1ms on CPU
- The frozen base encoder (qwen3-embedding) dominates inference latency (~0.5s per step via Ollama)
- For real-time use: cache embeddings during generation, run projector+predictor on cached vectors

## Implementation plan

### File structure
```
experiments/exp4_jepa/
  README.md           ← this file
  data_prep.py        ← embed trajectories, build training triples
  model.py            ← projector, predictor, loss functions
  train.py            ← training loop
  evaluate.py         ← linear probe comparison, trepidation test, paraphrase test
  results/            ← saved models, metrics, figures
```

### Dependencies
- torch (already installed)
- numpy, sklearn (already installed)
- requests (for Ollama API, already available)
- qwen3-embedding via Ollama (already pulled)

### Key design decisions

1. **Frozen base + learned head**: don't retrain the 8B encoder. Learn a 3.6M-parameter head on top. This is tractable on CPU and doesn't require GPU.

2. **256-dim projected space**: the projector compresses 4096 → 256, forcing it to discard surface noise and keep structural signal. This is where normalization happens.

3. **Barlow Twins on projected space**: the cross-correlation matrix operates on the 256-dim projected embeddings, not the 4096-dim raw embeddings. This is where redundancy reduction is enforced.

4. **Multi-horizon prediction**: predicting k=1, 3, 5 simultaneously forces the predictor to learn both immediate transitions and longer-horizon trajectory shape. This is what enables trepidation (divergence across horizons = uncertainty).

5. **Stop-gradient on target**: the target projector is an EMA copy with no gradients flowing through it. This prevents collapse — the predictor can't trivially match a constant target.

6. **Action-conditioned**: the predictor takes both current state AND next action as input. This makes it a world model (what happens if I do X?) not just a sequence model (what comes next?).

## What this architecture does NOT do

- Does not retrain the base encoder (frozen)
- Does not generate text (pure embedding-space operation)
- Does not reason about logic (pattern detection only)
- Does not replace the base model (advisory/monitoring role)
- Does not require GPU for training (3.6M params on CPU is fine)

## Falsification tests for this architecture

1. **Normalization test**: does the trained projector reduce paraphrase distance vs frozen encoder? If no, Barlow Twins isn't normalizing.
2. **Chunking recovery test**: does the trained projector recover the signal lost by chunking in exp2 (chunking ablation)? If yes, normalization works. If no, the problem isn't normalization.
3. **Prediction test**: can the predictor forecast z_{t+k} better than a baseline (mean of training trajectories)? If no, JEPA prediction adds nothing.
4. **Trepidation test**: do failed trajectories have higher cross-horizon variance? If no, trepidation is not a real signal.
5. **Outcome prediction test**: does the projector+predictor beat the frozen-embedding linear probe (exp1) at earlier prediction horizons? This is the key comparison — if JEPA doesn't beat the linear probe, it's not worth the complexity.