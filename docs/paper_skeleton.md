# Paper Skeleton — "Low-Dimensional Trepidation: Harness-Invariant Geometric Early Warning for LLM Agent Sessions"

Working title. All sections outlined with figure/table slots. Results marked
[X] until exp9 attempt #2/#3 lands. Written to be filled, not to impress.

## Abstract (draft, ~150 words)

LLM agents produce rich multi-step trajectories whose success or failure is
often only determined at the end. We study whether a low-dimensional
geometric representation of a session trajectory — computed from frozen
embeddings or a from-scratch encoder — can predict failure early, transfer
across agent harnesses, and survive format variation. On 3,000 real agent
sessions from 5 harnesses and 3 benchmarks, we show: (1) frozen-embedding
probes achieve 0.87-0.92 AUC for outcome prediction and transfer across
harnesses, but (2) this signal is matched by TF-IDF, and velocity/trajectory-
shape features add +0.000 beyond content, (3) all representations collapse
to 0.44-0.62 AUC under benchmark-disjoint evaluation, revealing that outcome
semantics are benchmark-local, (4) a 6M-parameter from-scratch encoder trained
with Barlow Twins on synthetic format pairs achieves [X] AUC under
harness-disjoint evaluation, demonstrating [that format-invariant agentic
space is / is not] learnable at low cost. We release the first
content-vs-geometry decomposition, harness-disjoint evaluation protocol, and
label-leakage case study for agent-trace monitoring.

## 1. Introduction

- The problem: agents fail late, monitoring is expensive, LLM judges are
  costly and uncalibrated.
- The question: can a tiny geometric monitor detect trouble from session
  shape, independent of harness format?
- Contributions (numbered, concrete):
  1. Content-vs-geometry decomposition: TF-IDF as mandatory control for
     embedding-probe claims on agent traces (new methodology).
  2. Harness-disjoint evaluation: LOHO/LOBO/LOMO protocol with metadata-aware
     splits (new evaluation).
  3. Benchmark-disjoint collapse: outcome signal is benchmark-local (new
     finding, constrains all trace-monitor claims).
  4. From-scratch 8-dim A-space encoder with BT format-invariance training
     (new architecture, result pending).
  5. Label-leakage case study: transductive outcome-head contamination
     (methodological caution).

## 2. Related Work

(see docs/related_work.md — 5 clusters, ~40 refs target)

## 3. Method

### 3.1 Session representation
- Canonical session format: ordered step texts + outcome + metadata.
- Step embedding: frozen nomic-embed-text (768d) or from-scratch encoder.
- Session vector: mean of step vectors (centroid).
- Velocity: step-to-step displacement v_t = z_{t+1} - z_t.

### 3.2 Probes
- Logistic regression (C=1.0, standardized features, 5-fold stratified CV).
- Features: static (centroid+final), PCA-8, mean-velocity, TF-IDF, format
  tags, length.
- All transforms fit on train split only.

### 3.3 Evaluation protocol
- in_format: 5-fold stratified CV (seed 42).
- LOHO: leave-one-harness-out (metadata-aware, min 150 per held-out group).
- LOBO: leave-one-benchmark-out.
- LOMO: leave-one-model-out.
- Metrics: AUC, accuracy, Brier score, 10-bin ECE.
- Bootstrap 95% CIs (300x test-set resampling).

### 3.4 From-scratch A-space encoder
- 4-layer transformer, 256d, 6M params, own BPE tokenizer (8k vocab).
- Losses: MLM + Barlow Twins (all format-pairs, L2-normalized z) + prefix
  consistency.
- Synthetic format renderers: chat, xml, json, otel, terse (5 formats).
- Two-run discipline: Run A (no outcome), Run B (+outcome head) — B tests
  for transductive label leakage.

### 3.5 Controls (mandatory, per experiment)
- Content: TF-IDF on raw step text.
- Length: session step count only.
- Format tags: harness markup counts only.
- Session-disjoint splits (never iid row CV).

## 4. Experiments

### 4.1 Do fallacy types have geometric signatures? (exp1)
- Dataset: Jin et al. 2022 LOGIC (2,449 fallacy examples, 13 types).
- Finding: qwen3-embedding 65.1% multiclass (above TF-IDF 49.0%, p~1e-42);
  nomic 50.8% ≈ TF-IDF 49.0%. Cross-encoder rank rho=0.945.
- [Figure 1: per-fallacy AUC bar chart, both encoders + TF-IDF]

### 4.2 Does velocity/shape carry signal beyond content on real sessions? (exp7)
- Dataset: 2,000 Exgentic sessions, nomic-embed-text.
- Finding: static 0.901, TF-IDF 0.918, velocity increment +0.000.
- [Table 1: feature × AUC, content-controlled]

### 4.3 Does the signal transfer across harnesses? (exp8)
- Dataset: 3,000 Exgentic sessions, 5 harnesses, 3 benchmarks, 5 models.
- [Table 2: feature × {in_format, LOHO, LOBO, LOMO} AUC — THE table]
- [Figure 2: drop bar chart — in_format minus LOHO per feature]
- Finding: LOHO 0.87-0.92 (transfers), LOBO 0.44-0.62 (dies), tags 0.51.

### 4.4 Can a from-scratch encoder learn format-invariant A-space? (exp9)
- [Table 3: Run A vs Run B — LOHO, LOBO, cross-format margin, collapse check]
- [Figure 3: cross-format cosine matrix — within-step vs between-step]
- [Figure 4: per-dim variance (collapse check)]
- Result: [X — pending attempt #2/#3]

### 4.5 Label-leakage case study (exp9 Run B)
- [Table 4: Run A vs Run B, with the leakage diagnosis]
- Finding: Run B 0.985 LOHO = transductive contamination; excluded.

## 5. Analysis

### 5.1 What does the space look like?
- [Figure 5: 2D PCA of 8-dim session vectors, colored by outcome — if
  the space separates visually, this is the money figure]

### 5.2 Dimensionality curve
- [Figure 6: AUC vs PCA dimensions on real sessions — 8/32/64/full]

### 5.3 Early warning
- [Figure 7: prefix AUC vs fraction of session visible]

### 5.4 The "anything-space" generalization
- Formal structure: process domain D, format variants F, encoder E, test
  cross-format retrieval + cross-domain transfer.
- Agent sessions = first instance. Code traces, medical visits, robot
  paths = future instances. The architecture and controls are the same.

## 6. Discussion

- Limitations: one dataset family (Exgentic), one frozen encoder (nomic),
  3k-60k training sessions. LOBO collapse means "universal trepidation" is
  false; per-task calibration is required.
- The lexical bar: if frozen geometry ≈ TF-IDF, what is the embedding
  buying? Answer: transfer (LOHO), compression (8 dims), and the
  *potential* for format-invariance that TF-IDF cannot achieve.
- Implications for PRMs: if velocity carries no signal beyond content,
  what are step-level PRMs learning? (Open question, not answered here.)
- Honesty section: what we claim and don't claim (per NOMENCLATURE tags).

## 7. Conclusion

[3 sentences. Written last. Not yet.]

## Appendices

A. Full per-harness and per-benchmark tables (exp8_results.json).
B. Adversarial fallacy examples (exp1 FALSIFICATION.md).
C. Hyperparameters and training logs.
D. Reproducibility: all code, seeds, data sources documented.