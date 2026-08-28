# Experiment 1: Do Logical Fallacies Have Geometric Signatures in Embedding Space?

## Mini-paper plan (revised: dual-encoder design)

### Title
"Do Logical Fallacies Have Geometric Signatures in Embedding Space? A Dual-Encoder Linear Probe Study"

### Core hypothesis
Logical fallacy types have geometric signatures in embedding space that are detectable by a linear probe — and this signal is an EMERGENT PROPERTY of reasoning structure, not an artifact of any particular encoder. If two encoders from different model families (Nomic 137M and Qwen3 7.6B) both show the signal, the geometric structure is intrinsic to how fallacies manifest in representation space.

### Research questions

**RQ1**: Can a linear probe distinguish between fallacy TYPES from frozen embeddings alone?

**RQ2**: Does the signal persist across encoders from different families? (emergent property test)

**RQ3**: Which fallacy types have the strongest/weakest geometric signatures?

**RQ4**: Can a linear probe distinguish fallacious from valid arguments? (binary, requires control set)

### Dual-encoder design

| Encoder | Family | Params | Dims | MTEB | Purpose |
|---|---|---|---|---|---|
| nomic-embed-text | Nomic AI | 137M | 768 | 59.4 | Small, different family |
| qwen3-embedding | Alibaba/Qwen | 7.6B | 4096 | 70.58 | Large, different family |

Both are accessed via Ollama on localhost:11434. If both show fallacy-type separability above chance, the signal is encoder-independent.

### Data

Jin et al. (2022) logical fallacy dataset:
- 2,452 examples, 13 fallacy types
- Columns: source_article (text), updated_label (fallacy type)
- Distribution: faulty generalization (441), ad hominem (302), ad populum (232), false causality (216), circular reasoning (171), appeal to emotion (167), fallacy of relevance (162), fallacy of logic (152), intentional (143), false dilemma (141), fallacy of extension (141), fallacy of credibility (132), equivocation (49)
- Drop "miscellaneous" (3 samples) → 12 classes, 2,449 examples

### Method

#### Step 1: Embed all examples with both encoders
- POST to Ollama /api/embeddings for each source_article
- nomic-embed-text: 768-dim vectors (~25 sec for 2,452)
- qwen3-embedding: 4096-dim vectors (~5-8 min for 2,452)
- Cache to .npy files

#### Step 2: Multiclass linear probe (RQ1, RQ2)
- For each encoder:
  - X = embeddings, y = fallacy type (12 classes)
  - Standardize, stratified 80/20 split, 5-fold CV
  - LogisticRegression (multinomial, max_iter=2000)
  - Report: accuracy, macro-F1, per-class F1, confusion matrix
- Baselines: majority class (18%), random (8.3%)
- Compare: does the signal hold across both encoders?

#### Step 3: Per-fallacy separability (RQ3)
- For each encoder, for each fallacy type:
  - One-vs-rest binary probe, report ROC-AUC
  - Rank fallacy types by separability
- Compare rankings across encoders: do the same fallacies rank high/low?

#### Step 4: Visualization (the figure)
- For each encoder:
  - PCA 2D + 3D projection, colored by fallacy type
  - t-SNE projection (perplexity 30)
  - Per-type centroid + intra-type distance vs inter-type distance
- Save figures to results/

#### Step 5: Binary fallacy vs valid (RQ4)
- Construct valid argument set (use LLM to generate or use NLI entailment pairs)
- Binary probe: fallacy (any type) vs valid
- Report ROC-AUC for both encoders

### Expected outcomes

| Result | Interpretation |
|---|---|
| Both encoders >25% multiclass accuracy | Fallacy types are geometrically separable — emergent property |
| Both encoders 15-25% | Weak signal, some types separable |
| Both <15% | No signal in embeddings — dead |
| One strong, one weak | Signal is encoder-dependent, not emergent |
| Same fallacy types rank high/low in both | Geometric signature is consistent across encoders |
| Different rankings | Signal is encoder-specific, less interesting |

### Timeline
~1 hour total: embed (10 min), probe (5 min), visualize (15 min), analyze (30 min)