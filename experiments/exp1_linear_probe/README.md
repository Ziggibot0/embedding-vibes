# Experiment 1: Do Logical Fallacies Have Geometric Signatures in Embedding Space?

## Mini-paper plan

### Title
"Do Logical Fallacies Have Geometric Signatures in Embedding Space? A Linear Probe Study"

### Abstract (target)
Logical fallacies are notoriously difficult for LLMs to detect via reasoning (~34% accuracy, Jin et al. 2022). We ask a simpler question: do fallacious arguments occupy distinguishable regions of embedding space compared to valid arguments? If so, a linear probe on frozen embeddings should classify fallacy type above chance — and the direction of separation would reveal whether fallacies have geometric structure that a predictive model could exploit without reasoning. We embed 2,452 labeled fallacy examples across 13 fallacy types using nomic-embed-text and train linear probes for binary (fallacy vs valid) and multiclass (fallacy type) classification. We find [RESULT].

### Research questions

**RQ1**: Can a linear probe on frozen embeddings distinguish fallacious arguments from valid arguments above chance?

**RQ2**: Can a linear probe distinguish between fallacy TYPES (ad hominem vs circular reasoning vs false causality, etc.)?

**RQ3**: Are specific fallacy types more or less geometrically separable? (Which fallacies have the strongest geometric signature?)

**RQ4**: Does the geometric separation depend on the encoder? (nomic-embed-text vs a larger model's hidden states)

### Why this experiment design

An experienced researcher would say: "Before you build a JEPA predictor, you need to know if the signal is linearly extractable from frozen embeddings. If a linear probe can't separate fallacy from valid above chance, then no learned predictor — JEPA or otherwise — will do better on the same embeddings. The linear probe is the ceiling for what's achievable with that encoder."

This is the "existence proof" that the entire research plan depends on. It costs one afternoon and uses tools already on the machine.

### Data

**Dataset**: Jin et al. (2022) logical fallacy dataset, already downloaded.
- 2,452 fallacy examples across 13 fallacy types
- Source: educational quiz data + climate articles
- Columns: source_article (text), updated_label (fallacy type)
- Labels: ad hominem (302), ad populum (232), appeal to emotion (167), circular reasoning (171), equivocation (49), false causality (216), fallacy of credibility (132), fallacy of extension (141), fallacy of logic (152), fallacy of relevance (162), false dilemma (141), faulty generalization (441), intentional (143)

**Problem**: This dataset has ONLY fallacious examples. There is no "valid argument" class.

**Solution**: We need valid arguments as a control. Two approaches:
1. **Synthetic valid arguments**: Use a model to generate valid arguments on the same topics (expensive, may introduce bias)
2. **Existing NLI datasets**: Use premises from SNLI/MNLI that are labeled as "entailment" (logically valid) as the "valid" class
3. **Simplest approach**: The dataset is multiclass across 13 fallacy types. RQ2 (can we distinguish fallacy TYPES) is answerable without a valid class. RQ1 (fallacy vs valid) requires a valid class — we can use a generic sentence embedding dataset or sample from a different dataset.

**Decision**: Run RQ2 first (multiclass fallacy type classification — no valid class needed). Then add a valid class from a second source for RQ1.

### Method

#### Step 1: Embed all 2,452 fallacy examples
- Use nomic-embed-text via Ollama API (already running on localhost:11434)
- POST to /api/embeddings with each source_article text
- Collect 768-dim embeddings (nomic-embed-text output dimension)
- Store as numpy array: (2452, 768)

#### Step 2: Linear probe — multiclass fallacy type (RQ2)
- X = embeddings (2452, 768)
- y = fallacy type labels (13 classes, drop "miscellaneous" with only 3 samples → 12 classes, 2449 samples)
- Train/test split: 80/20, stratified by label
- Model: LogisticRegression (multinomial, max_iter=1000, C=1.0)
- Report: accuracy, macro-F1, per-class F1, confusion matrix
- Baseline: majority class (faulty generalization, 18% accuracy)
- Baseline: random (1/12 = 8.3%)

#### Step 3: Binary fallacy vs valid (RQ1)
- Add valid arguments: sample 500 sentences from a general text dataset (e.g., Wikipedia first sentences, or NLI entailment pairs)
- Label: 0 = valid, 1 = fallacious (any fallacy type)
- Train/test split: 80/20
- Model: LogisticRegression (binary)
- Report: accuracy, F1, ROC-AUC
- Baseline: majority class

#### Step 4: Per-fallacy separability analysis (RQ3)
- For each fallacy type, train a one-vs-rest linear probe
- Report ROC-AUC for each fallacy type
- Rank fallacy types by separability
- Hypothesis: circular reasoning and equivocation will be HARDEST (most semantically subtle), ad hominem and appeal to emotion will be EASIEST (most stylistically distinct)

#### Step 5: Geometric visualization (RQ3, the figure)
- PCA projection to 2D and 3D
- Color by fallacy type
- Compute per-type centroid and average intra-type distance
- Compute inter-type centroid distances
- Plot: if fallacy types form visible clusters, that's the figure that sells the paper
- Additional: t-SNE and UMAP projections for comparison

#### Step 6: Encoder comparison (RQ4, optional)
- Repeat steps 2-3 with a different encoder
- Option A: use a larger model's hidden states (e.g., Qwen3-14B via Ollama)
- Option B: use sentence-transformers all-MiniLM-L6-v2 (if installable)
- Compare: does a larger/different encoder improve separability?

### Expected outcomes and what they mean

| Result | Interpretation | Next step |
|---|---|---|
| Multiclass accuracy > 25% (3x chance) | Fallacy types are geometrically separable | Proceed to JEPA predictor |
| Multiclass accuracy 15-25% | Weak signal, some types separable | Focus on most separable types, check if larger encoder helps |
| Multiclass accuracy < 15% (~chance) | Embeddings don't carry fallacy signal | Dead. Try different encoder or abandon |
| Binary AUC > 0.7 | Fallacy vs valid is clearly separable | Strong evidence for geometric signatures |
| Binary AUC 0.55-0.7 | Weak but present signal | May need trajectory analysis (multi-step), not single-embedding |
| Binary AUC < 0.55 | No signal | Dead for this encoder |
| Circular reasoning has LOWEST AUC | Semantically subtle, hardest to detect geometrically | Interesting finding for paper |
| Ad hominem has HIGHEST AUC | Stylistically obvious, easy to detect | Expected, less interesting |

### What this mini-paper contributes (standalone)

1. **First systematic study of fallacy-type separability in embedding space** — Jin et al. (2022) tested LLMs and structure-aware classifiers on fallacy detection. Nobody has tested whether frozen embeddings alone carry fallacy signal via linear probing.

2. **Per-fallacy geometric separability ranking** — which fallacies are easy to detect geometrically and which are hard. This is new and useful for the field.

3. **The figure** — if fallacy types form visible clusters in PCA/t-SNE space, that's a publishable visualization on its own.

4. **Baseline for JEPA predictor** — the linear probe's AUC is the number the JEPA predictor needs to beat. Without this baseline, there's no way to know if the JEPA predictor adds value.

### Technical details

**Embedding**: nomic-embed-text via Ollama
- API: POST http://localhost:11434/api/embeddings {"model": "nomic-embed-text", "prompt": "<text>"}
- Dimension: 768
- Rate: ~100 embeddings/sec on this hardware
- 2,452 examples → ~25 seconds

**Classifier**: sklearn LogisticRegression
- Standardize embeddings (StandardScaler) before fitting
- Stratified train/test split (80/20)
- 5-fold cross-validation for robustness
- Report: accuracy, macro-F1, weighted-F1, per-class F1, confusion matrix, ROC-AUC (for binary)

**Visualization**: matplotlib + sklearn PCA
- 2D and 3D PCA projections
- t-SNE (perplexity 30, 50)
- UMAP (if installable, otherwise skip)
- Save as PNG to docs/figures/

### Code structure

```
experiments/
  exp1_linear_probe/
    README.md          ← this file
    embed.py           ← embed all fallacy examples via Ollama
    probe.py           ← train and evaluate linear probes
    visualize.py       ← PCA/t-SNE plots
    results/           ← saved metrics, confusion matrices, plots
```

### Timeline

- Step 1 (embed): 30 min (API calls + caching)
- Step 2 (multiclass probe): 15 min
- Step 3 (binary probe): 30 min (need to find/create valid argument set)
- Step 4 (per-fallacy analysis): 15 min
- Step 5 (visualization): 30 min
- Step 6 (encoder comparison): 1 hour (optional)
- Total: 2-3 hours

### Paper output

This experiment produces a standalone mini-paper (~4 pages) that can be:
1. Submitted as a workshop paper (e.g., ICML Workshop on Reasoning)
2. Used as Section 4.1 (Existence Proof) of the full paper
3. Released as a technical report on arXiv

The mini-paper's contribution is: "We show that logical fallacy types are [partially/fully] separable in the embedding space of a frozen text encoder, establishing that fallacy detection does not require reasoning if the geometric signal is sufficiently strong."