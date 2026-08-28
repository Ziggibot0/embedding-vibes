# Experiment 2: Markov Chain Fallacy Detection on Agentic Embedding Trajectories

## Quick start

```bash
# 1. Generate sessions + embed (5 per fallacy type, ~45 min)
python run_quick.py

# 2. Build Markov chains (PCA + k-means + transition matrices)
python build_mc.py

# 3. Run classifier (Markov LLR vs static probe)
python predict.py

# 4. Visualize results
python visualize.py
```

## For a larger run (50 per type, ~5 hours)

```bash
# Edit N_SESSIONS_PER_CLASS in simulate.py, then:
python simulate.py
python build_mc.py
python predict.py
python visualize.py
```

## Files

| File | Purpose |
|------|---------|
| DESIGN.md | Research design, RQs, method, falsification criteria |
| run_quick.py | Small-batch generation (5/type x 9 = 90 sessions) |
| simulate.py | Full generation (50/type x 9 = 900 sessions) |
| build_mc.py | PCA + k-means + transition matrices + entropy + KL |
| predict.py | Markov LLR classifier vs static probe baseline |
| visualize.py | All figures (entropy, KL, stationary, trajectories, heatmaps) |

## Key idea

Exp1 showed fallacy types are geometrically separable from *static* embeddings (AUC 0.70-0.96). This experiment asks: do *trajectories* through embedding space carry additional signal?

A Markov chain models P(state_{t+1} | state_t) — the probability of where reasoning goes next given where it is now. If fallacious reasoning has different transition structure than valid reasoning, the Markov chain is a fallacy detector that never reads the content.

## Falsification

- Markov AUC > static AUC → trajectory adds signal
- Markov AUC ≈ static AUC → trajectory is redundant  
- Markov AUC < static AUC → discretization loss dominates
- Entropy(fallacy) > Entropy(valid) → fallacious reasoning is more "random walk"

## Outputs (results/)

- sessions.json — raw session data
- session_labels.json — per-session labels
- trajectory_embeddings_*.npy — per-step embeddings
- trajectory_meta_*.json — session/step index mapping
- T_fallacy_*.npy, T_valid_*.npy — transition matrices
- pi_*.npy — stationary distributions
- H_*.npy — per-state entropies
- kl_fv_*.npy — per-state KL divergence
- states_*.npy — k-means state assignments
- markov_results.json — summary statistics
- predict_results.json — classifier comparison

## Figures (figures/)

- entropy_comparison_*.png
- kl_divergence_*.png
- stationary_*.png
- trajectories_pca_*.png
- transition_heatmap_*.png