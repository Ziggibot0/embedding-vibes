# Experiment 2: Markov Chain Fallacy Detection on Agentic Embedding Trajectories

## Title
"Markov Vibes: Do Agentic Reasoning Trajectories Have Fallacy-Specific Transition Structure in Embedding Space?"

## Core hypothesis
Reasoning trajectories that contain logical fallacies have different Markov transition structure in embedding space than valid reasoning trajectories — and this transition signal is orthogonal to (and complements) the static geometric signal from exp1.

## Research questions

**RQ1**: Can a Markov chain on discretized embedding trajectories distinguish fallacious from valid agentic sessions better than a static linear probe on individual embeddings?

**RQ2**: Do fallacy types have distinct transition signatures (not just distinct locations)?

**RQ3**: Does the transition signal survive across encoders (emergent property)?

**RQ4**: What is the entropy structure of fallacious vs valid trajectories? (Do fallacious sessions have higher transition entropy — more "random-walk" reasoning?)

## Method

### Step 1: Generate agentic reasoning sessions
- Use Ollama to generate multi-step reasoning sessions
- Two conditions:
  - **Fallacious**: inject a specific fallacy type into the reasoning prompt
  - **Valid**: reasoning prompts that produce sound arguments
- Each session = sequence of 4-8 reasoning steps
- Embed each step with both encoders (nomic-embed-text, qwen3-embedding)
- Also embed the fallacy dataset examples as single-step "trajectories" for comparison

### Step 2: Discretize embedding space
- PCA to 50 dimensions (preserve structure, reduce curse of dimensionality)
- K-means with K=200 clusters (Voronoi cells = Markov states)
- Each embedding step → cluster index → Markov state sequence

### Step 3: Build transition matrices
- For each class (fallacy type, valid):
  - Count transitions: T[i,j] = count(state_i → state_j)
  - Normalize rows: T[i,j] /= sum(T[i,:])  (row-stochastic)
  - Laplace smoothing for unseen transitions
- Compute stationary distributions π (eigenvector of T)
- Compute per-state entropy H(i) = -Σ_j T[i,j] log T[i,j]

### Step 4: Markov classifier
- For a new trajectory s_0 → s_1 → ... → s_T:
  - Log-likelihood under T_fallacy: LL_f = Σ log T_fallacy[s_t, s_{t+1}]
  - Log-likelihood under T_valid: LL_v = Σ log T_valid[s_t, s_{t+1}]
  - Classify: fallacy if LL_f > LL_v
- Cross-validate against static probe baseline from exp1

### Step 5: Analysis
- KL divergence between T_fallacy and T_valid per fallacy type
- Transition entropy comparison (fallacy vs valid)
- Stationary distribution comparison
- Identify high-discriminability transitions (states where T_fallacy and T_valid diverge most)

## Falsification criteria

| Result | Interpretation |
|--------|---------------|
| Markov AUC > static probe AUC | Trajectory adds signal — transition structure is informative |
| Markov AUC ≈ static probe AUC | Trajectory is redundant — static signal suffices |
| Markov AUC < static probe AUC | Trajectory hurts — discretization loss dominates |
| Entropy(fallacy) > Entropy(valid) | Fallacious reasoning is more "random walk" — less directed |
| Entropy(fallacy) ≈ Entropy(valid) | No entropy signal — fallacies are locally coherent |

## Data sources

### Option A: Generated sessions (fast, controlled)
- Use Ollama to generate ~200 sessions (100 fallacious, 100 valid)
- Pros: controlled fallacy injection, known ground truth
- Cons: may not reflect real agentic behavior

### Option B: AgentTrove / existing datasets (real, noisy)
- 1.7M trajectories with reward labels
- Pros: real trajectories, large scale
- Cons: need to identify fallacy types, heavier compute

### Recommendation: Start with Option A for proof-of-concept, then validate with Option B

## Expected outcomes

If the Markov signal works:
1. Transition log-likelihood classifier beats static probe
2. Fallacious trajectories have higher transition entropy
3. Specific transitions are diagnostic of specific fallacy types
4. Signal persists across both encoders

This gives the paper a temporal dimension that exp1 lacks, and strengthens the "vibes" thesis: fallacies aren't just detectable from WHERE you are in embedding space, but from HOW YOU MOVE through it.