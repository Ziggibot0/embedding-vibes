# Exp6 — Joint Encoder + JEPA Predictor (the pilot's learned models)

**Goal:** train the two learned components of the pilot architecture (DESIGN.md)
TOGETHER — a Barlow-Twins encoder (64-dim) and a JEPA multi-horizon predictor —
and test whether the learned compact shape space helps or hurts.

**Data:** the 90 exp3 sessions (45 fallacy / 45 valid, 6 steps each), step
embeddings for both encoders. Joint training, EMA target, stop-gradient,
Barlow Twins anti-collapse. 200 epochs, CPU.

## Results

### 1. Collapse check — PASS
Projected 64-dim space has healthy variance (var ~0.95-0.97, std ~0.97-0.98) on
both encoders. No representation collapse. Barlow Twins + EMA target worked.

### 2. Delta separation — the learned 64-dim projection LOSES signal

| Method | nomic AUC | qwen3 AUC |
|---|---|---|
| Raw-embedding deltas (exp5) | 0.960 | 0.995 |
| **Learned 64-dim deltas** | **0.862** | **0.864** |

The learned 64-dim projection is **too lossy**. Compressing 768/4096 dims down
to 64 throws away delta signal that the raw representation preserved. This is a
direct falsification of the "64 dims is enough" hypothesis — the task-AUC-vs-
dimension curve matters, and 64 is below the elbow.

### 3. Prediction test — the JEPA predictor is strong

| Encoder | JEPA L1 | Mean baseline L1 | Improvement |
|---|---|---|---|
| nomic | 0.291 | 0.776 | −62.5% |
| qwen3 | 0.203 | 0.762 | −73.4% |

The JEPA predictor forecasts e_{t+k} far better than a mean-of-training baseline.
The forward model itself works.

## Reading

1. **The predictor works; the 64-dim encoder is the bottleneck.** The JEPA
   forward model is genuinely predictive (beats mean by 62-73%). But projecting
   through a 64-dim Barlow-Twins encoder destroys the delta signal that the raw
   embeddings carried. The compression is the problem, not the prediction.

2. **"64 dims" is falsified as a default.** This is exactly the task-AUC-vs-
   dimension curve I flagged earlier. 64 is below the elbow. The fix is to try
   higher projection dims (128, 256, 512) and find where the learned-delta AUC
   recovers toward the raw baseline.

3. **The pilot's core mechanism still holds** (exp5: raw deltas 0.960/0.995).
   The question is whether a learned encoder can match it at a higher dimension,
   or whether the raw delta representation is simply the better shape space.

## Next steps

- **Dimension sweep:** retrain the joint encoder at PROJ_DIM ∈ {128, 256, 512}
  and plot learned-delta AUC vs dimension. Find the elbow.
- **If learned never matches raw:** the pilot's shape space may be best as the
  raw relative-delta representation (no learned encoder), with the JEPA
  predictor operating on raw deltas. That's a legitimate, simpler architecture.
- **Stylistic-vs-logical control** still pending (paraphrase test).

## Caveats

- Small N (90 sessions, 5-fold CV).
- Fixed 6-step sessions.
- The learned encoder was trained on the same 90 sessions it's evaluated on
  (no held-out split for the projection itself) — the delta-separation numbers
  are optimistic, not pessimistic.
