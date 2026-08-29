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

---

## ADDENDUM (2026-08-28): controls re-run — claim needs re-scoping

Re-analysis prompted by the question "is exp1 significant, and would more sampling help?" All numbers below reproduced fresh from the cached embeddings (`embeddings_nomic_embed_text.npy` / `embeddings_qwen3_embedding.npy`, 2449 rows, zero error rows) + `edu_all.csv` raw texts, same LogisticRegression(max_iter=2000, C=1.0) protocol.

### What reproduces
- Cached CV metrics match RESULTS.md exactly (nomic 0.508/0.465, qwen3 0.651/0.613).
- Random-label control (9.8%/10.3%) stands; no label leakage.
- Cross-encoder per-type AUC rank consistency is real: Spearman rho=0.945, p~1e-5, n=13.

### New control 1: TF-IDF lexical baseline (same CV protocol)
- TF-IDF (word 1-2 grams, min_df=2, sublinear) + same LogisticRegression:
  accuracy 0.490 +/- 0.014, macro-F1 0.420 +/- 0.014.
- nomic-embed-text (0.508 / 0.465) is ~1-2 points above a bag of n-grams.
  The "geometric signature" claim is NOT established for the small encoder.
- qwen3-embedding (0.651 / 0.613) beats TF-IDF by +0.121 accuracy
  (paired over 50 fold-fits: t=46.5, p=3.3e-42). Beyond-lexical signal is REAL,
  but only demonstrated for the large encoder.

### New control 2: repeated CV (50 fold fits, RepeatedStratifiedKFold 5x10)
- qwen3: 0.617 +/- 0.015   (original 0.651 was a favorable fold draw)
- nomic: 0.516 +/- 0.019
- tfidf: 0.495 +/- 0.016
Variance is already ~+/-1.5%. More sampling (LLN) cannot change any conclusion
here. The limiting factor is controls, not N.

### New control 3: split by source SITE (original_url), GroupKFold
-QUIZ SOURCE LEAKAGE: quizizz.com alone is 1645/2449 (67%) of examples. iid CV
 lets the probe memorize site style. Group-disjoint eval:
  - nomic: 0.384   (vs 0.508 iid)
  - qwen3: 0.535   (vs 0.617 repeated-iid)
  - tfidf: 0.344
- Note qwen3's edge GROWS under disjoint sites (+0.15..+0.19 over baselines):
  the embedding signal generalizes across sources better than lexical style.
  This strengthens the vibes hypothesis but was not the number reported.

### Binary RQ4 artifact (fallacy vs valid)
- 29 crafted valid + 39 sampled fallacious (n=68). Acc 92.4% -> Wilson 95% CI
  [0.836, 0.967]; AUC 0.987. Adversarial set is n=15 (CI ~[0.62, 0.96]).
- Stylistic confound (FALSIFICATION.md risk #1/#4) remains open and is now
  more pressing given the TF-IDF result. Superseded in practice by exp7's
  2000-session real-data gate (static centroid 0.901).

### Literature anchor
Jin et al. 2022 (same LOGIC dataset, their 300-sample held-out test):
finetuned Electra 53.3 micro-F1, +structure-aware 58.8; zero-shot models 8.6-13.7;
later LLM zero-shot ~35-36. Frozen qwen3 embedding + untrained logistic head is
therefore competitive with finetuned-precedent-SOTA at zero training cost.

### Revised claim
- HOLD: fallacy type is strongly, linearly decodable from frozen qwen3-embedding
  geometry of quiz-style text (>= finetuned BERT-era baselines, zero training).
- WEAKEN: "emergent property across encoders" — both encoders are decodable, and
  their rankings agree (rho=0.945), but nomic's decodable content is largely
  lexical/topical (TF-IDF-equivalent). Signal exceeds surface form only for
  qwen3-class encoders.
- REPORT: use site-disjoint numbers as the honest generalization estimate.
