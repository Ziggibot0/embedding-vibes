# Exp8 — Harness-Disjoint Transfer: does the session signal survive format shift?

Phase: ROADMAP **P0** (metadata repair) + first half of the P2 gate.
Pre-registered gates below are FROZEN before running. No post-hoc thresholds.

## Why this experiment exists (the "why" chain)

1. **NOMENCLATURE's central untested claim is A-space** — a session
   representation where actions crystallize and formatting blurs. Before we can
   *train* such a space (P2 crystallization), we must know how much of the
   signal already transfers across harnesses with zero invariance training.
   Exp8 measures the floor.
2. **The exp7 addendum found the blocking defect**: sessions.jsonl kept no
   harness/benchmark metadata, so no transfer claim could even be evaluated.
   The source dataset (Exgentic/agent-llm-traces-v2, schema v1.2) HAS these
   fields at row level (verified 2026-08-28: `harness`, `benchmark`,
   `benchmark_subset`, `models`, `success`, `score`, `steps`). Exp8 re-extracts
   with metadata intact.
3. **Why frozen probes, not a trained model, for exp8**: any *learned* model
   could be accused of learning to smooth over format differences. The
   cleanest measurement of "what transfers" is the same untrained
   LogisticRegression protocol as exp7, evaluated harness-disjoint. Training
   enters in P2 only.
4. **Why nomic-embed-text only**: comparability with exp7's 0.901/0.918
   numbers (same encoder, same probe family), and embedding-budget discipline
   (~100 texts/s local vs ~5/s for qwen3-embedding). qwen3 transfer = cheap
   follow-up once nomic result is known (cache keyed by model).
5. **Why both harness- and benchmark-disjoint**: harness is the *format*
   axis (the crystallization claim); benchmark is the *task-content* axis
   (secondary: does "trouble" look the same across task families). Different
   claims, different splits, both reported.

## Data

- Source: `Exgentic/agent-llm-traces-v2` (train split, streaming).
- Extraction: identical step-text logic to exp6 `data_prep.py` (assistant
  reasoning + `[tool_call]` prefixed calls + `[user]` context; consecutive
  dedupe) so exp7 numbers stay comparable, PLUS metadata kept per session:
  `harness, benchmark, benchmark_subset, models, success, score, steps,
  total_tokens, execution_time, run_id`.
- Inclusion: 3 <= n_steps_text <= 40 (exp6's filter — comparability).
- No session cap planned; if unique-text embedding ETA exceeds ~45 min we cap
  sessions at 3,000 and document it here.

## Features (all frozen, no training)

| Feature | Dim | Why included |
|---|---|---|
| static_full = [centroid; final] | 1536 | exp7's strongest result; the monitor's carrier |
| static_pca8 | 8 | A-space is 8–64d by design; does the low-dim projection transfer as well as full? |
| meanvel | 768 | exp7b's best velocity form; Gate C recheck |
| tfidf (1-2 gram, min_df=5, max 2000) | sparse | MANDATORY content control (NOMENCLATURE) |
| tags (top-40 format-tag counts + rate + length) | ~43 | MANDATORY format control |
| length only | 1 | MANDATORY length control |
| tfidf+meanvel | sparse+768 | velocity increment (Gate C) |

Scaler/PCA/Tfidf/token-vocab fit on TRAIN fold only, applied to test — no leakage.

## Splits

- **in_format**: stratified 5-fold CV over all sessions (seed 42) — exp7-comparable reference.
- **loho (leave-one-harness-out)**: hold out each harness with >= 150 sessions; train on rest. Smaller harnesses reported as skipped (with counts).
- **lobo (leave-one-benchmark-out)**: same for benchmark.
- (models-disjoint: reported only if >= 2 models have >= 150 sessions.)

## Metrics per (feature, split)

AUC, accuracy, Brier score, ECE (10-bin expected calibration error) on
out-of-sample predictions. Calibration is included because P1's gate consumes
τ as a threshold; transfer must be judged with calibration attached, not just
ranking.

## Pre-registered gates (frozen before run)

Let `drop(X) = in_format_AUC(X) - loho_mean_AUC(X)` (harness axis).

- **Gate A (monitor transfers without per-harness retraining)**:
  static_full in_format >= 0.75 AND loho_mean >= 0.70.
- **Gate B (geometry retains more than lexical under format shift)**:
  drop(tfidf) - drop(static_full) >= 0.03  →  evidence the embedding carries
  something beyond surface form specifically under format shift (A-space bet
  has a floor to stand on). If B fails, P2 must *manufacture* invariance from
  zero — still worth doing, but the claim changes.
- **Gate C (velocity null recheck)**: (tfidf+meanvel) - tfidf <= 0.02
  in_format. Expected pass-negative; if it exceeds, exp7's falsification was
  extraction-dependent and we reopen shape.
- **Gate D (low-dim transfers too)**: drop(static_pca8) <= drop(static_full) + 0.02.

Verdict logic and house-style verdict block auto-written to RESULTS.md.

## Artifacts / logging

- `results/sessions_meta.jsonl` — canonical extract w/ metadata (the P0 artifact, reusable by P1/P2)
- `results/emb_cache_nomic.npy` — text->vec cache (resumable, keyed by model)
- `results/exp8_results.json` — full metric dump (every feature × split × group)
- `results/exp8_summary.md` — auto-generated quick-read tables
- `exp8.log` — phase-progress log (extraction counts, embed ETA, split sizes)

## Explicit non-goals

No training (that's P2). No qwen3-embedding run (follow-up). No live agent
(shim is a separate P0 deliverable). No synthetic re-rendering (P2 data aug).