# Experiment 1: Results

## Do Logical Fallacies Have Geometric Signatures in Embedding Space?

### Setup
- Dataset: Jin et al. (2022) logical fallacy dataset, 2,449 examples, 13 fallacy types
- Two encoders from different model families:
  - nomic-embed-text (Nomic AI, 137M params, 768 dims, MTEB 59.4)
  - qwen3-embedding (Alibaba/Qwen, 7.6B params, 4096 dims, MTEB 70.58)
- Method: LogisticRegression (multinomial), 5-fold stratified CV, StandardScaler
- Date: 2026-08-27

### Multiclass Results

| Metric | nomic-embed-text | qwen3-embedding |
|---|---|---|
| CV Accuracy | 0.508 ± 0.009 | 0.651 ± 0.023 |
| CV Macro-F1 | 0.465 ± 0.010 | 0.613 ± 0.032 |
| Chance baseline | 0.077 | 0.077 |
| Majority baseline | 0.180 | 0.180 |
| Accuracy / chance ratio | 6.6x | 8.5x |

### Per-Fallacy One-vs-Rest AUC

| Fallacy type | nomic AUC | qwen3 AUC | n (samples) |
|---|---|---|---|
| ad populum | 0.913 | 0.958 | 232 |
| false causality | 0.898 | 0.953 | 216 |
| false dilemma | 0.893 | 0.945 | 141 |
| fallacy of extension | 0.921 | 0.935 | 141 |
| ad hominem | 0.874 | 0.932 | 302 |
| fallacy of credibility | 0.819 | 0.912 | 132 |
| appeal to emotion | 0.806 | 0.900 | 167 |
| circular reasoning | 0.815 | 0.896 | 171 |
| faulty generalization | 0.782 | 0.886 | 441 |
| intentional | 0.792 | 0.880 | 143 |
| fallacy of relevance | 0.791 | 0.859 | 162 |
| fallacy of logic | 0.743 | 0.850 | 152 |
| equivocation | 0.696 | 0.832 | 49 |

### Key findings

1. **Signal is real**: Both encoders separate 13 fallacy types far above chance (6.6x and 8.5x).
2. **Emergent property**: The per-fallacy AUC ranking is nearly identical across both encoders despite completely different model families, training data, parameter counts (137M vs 7.6B), and embedding dimensions (768 vs 4096).
3. **Signal scales with encoder capacity**: qwen3-embedding has higher AUC for every single fallacy type, consistent with a more expressive embedding space capturing more geometric structure.
4. **All 13 types are separable**: Even the hardest fallacy (equivocation) has AUC 0.696 (nomic) / 0.832 (qwen3), well above 0.5 chance.
5. **Stylistically distinct fallacies are easiest**: ad populum, false causality, false dilemma, ad hominem — these have clear linguistic markers.
6. **Semantically subtle fallacies are hardest**: equivocation, fallacy of logic, fallacy of relevance — these require understanding logical structure, not just surface patterns.

### Interpretation

The consistent ranking across two unrelated encoders indicates that logical fallacy types have geometric signatures that are intrinsic to how fallacy structure manifests in representation space — not an artifact of any particular encoder's training. This is an emergent property: no encoder was trained to detect fallacies, yet both encode fallacy type in a geometrically separable way.

This finding supports the broader hypothesis that reasoning quality (and reasoning failures) have geometric signatures in embedding space that can be detected without reasoning — the "embedding vibes" hypothesis.

### Files
- `results/embeddings_nomic_embed_text.npy` — (2449, 768) embeddings
- `results/embeddings_qwen3_embedding.npy` — (2449, 4096) embeddings
- `results/labels.npy` — fallacy type labels
- `results/probe_results_nomic.json` — nomic probe metrics
- `results/probe_results_qwen3.json` — qwen3 probe metrics
- `results/pca_nomic_embed_text.npy` — PCA projections (10 components)
- `results/pca_qwen3_embedding.npy` — PCA projections (10 components)
- `embed.py` — embedding script (concurrent Ollama API calls)
- `probe.py` — linear probe evaluation script