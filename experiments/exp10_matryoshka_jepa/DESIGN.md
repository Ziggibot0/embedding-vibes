# Exp10 — Matryoshka JEPA: the russian-doll fix for the 64-dim bottleneck

**Status:** DESIGN (pre-registered). Not yet run.

## The problem this fixes (one paragraph, honest)

exp6 trained a JEPA predictor on a **64-dim projection of dense embeddings**
(nomic-embed-text v1, 768-dim; qwen3-embedding, 4096-dim). The predictor itself
was strong (beats mean-of-training by 62–73%), but the 64-dim projection
**destroyed the delta signal**: raw-embedding deltas 0.960/0.995 AUC vs learned
64-dim deltas 0.862/0.864 AUC. The root cause is now identified: **the source
embeddings were dense, not matryoshka.** Truncating a dense vector to 64 dims
throws away 92% of it with no guarantee the kept part is meaningful. That is
not "compression" — it is destructive blur.

## The fix

Use **nomic-embed-text:v1.5** (137M, matryoshka-trained, Apache-2.0, already
pulled on this machine). v1.5 is explicitly trained so that **prefixes of the
output vector are valid embeddings** — the first 64/128/256 dims are each a
complete, usable doll. Truncation is done client-side per the HF/nomic recipe:

```
embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
embeddings[..., :matryoshka_dim]   # keep first N dims, drop the tail
```

This is the "find an embedder that natively does 64 dims" path. The 64-dim
representation now keeps the signal instead of blurring it.

## What changes vs exp6

| | exp6 (broken) | exp10 (this) |
|---|---|---|
| Source embedder | nomic v1 (dense 768) / qwen3 (dense 4096) | **nomic v1.5 (matryoshka 768)** |
| 64-dim path | learned Projector (MLP) crushes dense→64 | **matryoshka truncation** 768→64 (no learned crush) |
| JEPA predictor | same architecture | same architecture (reuse) |
| Barlow Twins | on predicted vs target | keep (anti-collapse + format-invariance) |

The **learned Projector is removed** for the low-dim arm. Matryoshka truncation
is the projection. (We keep Barlow Twins on the predictor — it is the
guardrail, not the bottleneck.)

## Data

Reuse exp3's 90 sessions (45 fallacy / 45 valid, 6 steps each) — the same
corpus exp6 used, so results are directly comparable. Re-embed with v1.5
(one-time cost; the existing caches are v1 and must not be reused).

## The dimension sweep (the russian-doll test)

The core question is the **task-AUC-vs-dimension elbow** that exp6 flagged and
never ran. For each matryoshka dim D ∈ {64, 128, 256, 512, 768}:

1. Embed each step with v1.5, truncate to D (layer-norm first).
2. Compute relative-delta representation (exp5 recipe).
3. Linear-probe AUC for fallacy/valid separation (5-fold CV).
4. Train the JEPA predictor on the D-dim deltas; report prediction L1 vs
   mean-of-training baseline.

This produces the elbow curve: where does truncation stop hurting? If v1.5's
matryoshka property holds, AUC should stay high down to 64 (or at least far
better than exp6's 0.862/0.864).

## Pre-registered gates (frozen)

- **G1 (matryoshka works):** 64-dim truncated deltas beat exp6's learned
  64-dim deltas (0.862 nomic / 0.864 qwen3) on the same 90 sessions. If
  truncation ≥ learned crush, the russian-doll property is real.
- **G2 (recovers toward raw):** 64-dim truncated AUC ≥ 0.90 (raw nomic deltas
  were 0.960). If truncation recovers most of the raw signal, the bottleneck
  was the crush, not the dimension.
- **G3 (elbow):** the AUC-vs-dimension curve is monotone-ish and the drop from
  768→64 is small (< 0.05). If 64 collapses, the elbow is higher and we report
  the honest minimum viable dim.
- **G4 (predictor still works):** JEPA on truncated deltas beats mean-of-training
  by ≥ 50% (exp6 was 62–73%). Confirms the forward model survives the fix.

## Honest caveats

- **Small N** (90 sessions, 5-fold CV) — same limitation as exp6. This is a
  mechanism test, not a production claim.
- **v1.5 needs task prefixes** (`search_document`, `classification`, etc.) to
  work well. We must pick a prefix and apply it consistently; the choice is a
  confound to note, not to hide.
- **v1.5 context is 2048 tokens** (Ollama reports 2048, num_ctx 8192). Steps
  longer than that get truncated — the exp9 corpus caps steps at 20000 chars,
  so we must cap step text to v1.5's window.
- **This tests the representation, not the whole A-space claim.** A positive
  result means "matryoshka truncation preserves the delta signal" — it does not
  by itself prove format-crystallization or outcome prediction. Those are
  separate gates (exp9's G3/G4).

## Artifacts

- `embed_v15.py` — re-embed exp3 sessions with v1.5, layer-norm, truncate to
  each D, cache per-D .npy.
- `sweep.py` — the AUC-vs-dimension elbow + JEPA prediction per D.
- `RESULTS.md` — filled after the run, with the gates above.

## Why this is the right next move

It is the cheapest decisive test of the russian-doll hypothesis that exp6's
failure raised. It reuses existing data and code, needs no new large-model
training, and directly answers whether the 64-dim bottleneck was the *crush*
(learned projection) or the *dimension* (too few). If truncation recovers the
signal, the whole low-dim A-space program is back on the table with a sound
representation. If it does not, we have cleanly falsified "64 dims is enough"
even with a matryoshka embedder, and the honest path is higher dims or the
from-scratch encoder.
