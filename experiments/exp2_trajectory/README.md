# Experiment 2: Trajectory-Level Fallacy Detection

## The problem with exp1

Exp1 embedded whole statements as single points. This captures "this text is shaped like a fallacy" but misses the key insight: fallacies live in TRANSITIONS, not texts. The homeopathy example ("I took a remedy and my cold went away") failed as a single blob because the fallacy is in the implicit causal leap, not in any individual chunk.

## The insight

Microsoft (ACL 2026) found that correct and incorrect reasoning diverge at LATE steps in the trajectory through representation space. The signal is temporal, not static. A fallacy is a wrong TURN, not a wrong location.

## Temporal chunking approaches (ranked by fidelity)

### Option A: Hidden-state trajectory (highest fidelity, requires model internals)
- Extract hidden states at each token during generation from the base model
- The trajectory IS the model's reasoning process in its native representation space
- This is what Microsoft did — activations at "Step 1:", "Step 2:" markers
- Pro: most faithful to actual reasoning dynamics
- Con: requires access to model internals (llama.cpp fork can expose this)
- Con: trajectory is in model-specific hidden space, not a shared embedding space
- Shape: continuous path through high-dimensional space, one point per token

### Option B: Sliding-window embedding (high fidelity, model-agnostic)
- Embed every N tokens with a sliding window (e.g., N=20, stride=10)
- Produces a dense trajectory through the encoder's embedding space
- Pro: model-agnostic, works with any encoder
- Pro: no chunking decision — the trajectory is continuous
- Con: dense trajectory (hundreds of points), need to define "shape" for continuous paths
- Con: window size and stride are hyperparameters
- Shape: continuous path through embedding space, one point per window position

### Option C: Connective-based chunking (medium fidelity)
- Split on logical connectives: "therefore", "because", "so", "thus", "which means", "hence"
- Each chunk is a logical step; embed each step independently
- Pro: chunks align with logical structure
- Pro: discrete steps → easier to define trajectory shape (sequence of points)
- Con: not all reasoning uses explicit connectives
- Con: connectives are often used non-logically
- Shape: discrete sequence of points, one per logical step

### Option D: Sentence-level chunking (lowest fidelity, simplest)
- Split on sentence boundaries
- Pro: trivial to implement
- Con: logical steps don't map to sentences (one sentence can contain premise + conclusion)
- Shape: discrete sequence of points, one per sentence

### Option E: Step-marker chunking (for explicitly structured CoT)
- Split on "Step 1:", "Step 2:", "Let me think...", "Therefore..."
- Pro: aligns with reasoning model output format
- Con: only works for explicitly structured CoT
- Shape: discrete sequence of points, one per step

## What "shape" means for a trajectory

For a discrete trajectory (z_1, z_2, ..., z_n):
- **Path length**: sum of ||z_{i+1} - z_i|| — total distance traveled
- **Curvature**: angles between consecutive segments — how much the trajectory turns
- **Loop-back**: autocorrelation of position — does the trajectory return to where it started?
- **Direction changes**: how often the trajectory reverses direction
- **Manifold dimension**: intrinsic dimensionality of the trajectory
- **Terminal region**: where does the trajectory end up? (success vs failure cluster)

For a continuous trajectory (from hidden states or sliding window):
- All of the above, plus:
- **Velocity**: rate of change of position
- **Acceleration**: rate of change of velocity
- **Frequency**: oscillation rate through embedding space
- **Phase**: position in the oscillation cycle

## The experiment

### Phase 1: Build trajectory dataset
- Take 100 fallacious CoT traces + 100 valid CoT traces
- Sources: 
  - Fallacious: Jin et al. dataset examples, expanded into full reasoning chains
  - Valid: math proofs, logical arguments, scientific reasoning
- Apply chunking (start with Option C: connective-based, then try B: sliding window)
- Embed each chunk with both encoders
- Each trace becomes a trajectory: (z_1, z_2, ..., z_n)

### Phase 2: Trajectory shape analysis
- Compute trajectory metrics (path length, curvature, loop-back, direction changes)
- Statistical test: do fallacious and valid trajectories have different shape distributions?
- Visualization: 3D PCA projection of trajectories, colored by fallacy/valid
- THE FIGURE: if fallacious trajectories visibly loop or diverge and valid ones progress smoothly, that's the paper

### Phase 3: JEPA trajectory prediction
- Train predictor P(z_t, action) → z_{t+k} with Barlow Twins anti-collapse
- Can it predict where the trajectory is heading?
- Does predicted destination correlate with fallacy/valid outcome?
- This is the actual JEPA contribution — learned trajectory prediction, not just shape analysis

### Phase 4: Adversarial trajectory test
- The homeopathy example, chunked: ["I took a homeopathic remedy", "and my cold went away three days later", "therefore the remedy worked"]
- Does the THIRD chunk's predicted trajectory land near false causality?
- Does the TRANSITION from chunk 2 to chunk 3 show the fallacy signature?
- This tests whether trajectory-level detection catches what whole-statement missed

## Key question: how to chunk

The chunking strategy determines the trajectory granularity. Too coarse (whole statement) → misses the transition. Too fine (per-token) → noisy, need to define shape for continuous paths.

Best bet: start with connective-based chunking (Option C) because it produces discrete steps that align with logical structure. Then try sliding-window (Option B) for a denser, more continuous trajectory.

The homeopathy example chunked by connectives:
1. "I took a homeopathic remedy" → premise
2. "and my cold went away three days later" → observation  
3. "therefore the remedy worked" → conclusion (FALLACY: post hoc ergo propter hoc)

The fallacy is in step 3's transition from step 2. The whole-statement probe missed it. The trajectory probe should catch it because step 3's embedding should land near false causality, and the transition from step 2 to step 3 should show the characteristic "wrong turn" shape.

## Timeline
- Phase 1: 2-3 hours (build trajectory dataset, chunk, embed)
- Phase 2: 1-2 hours (shape analysis, visualization)
- Phase 3: 2-3 weeks (JEPA predictor training)
- Phase 4: 1 hour (adversarial test)