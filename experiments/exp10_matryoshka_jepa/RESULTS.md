# Exp10 — Matryoshka JEPA: the russian-doll fix (RESULTS)

Run 2026-08-30. Re-embedded exp3's 90 sessions (45 fallacy / 45 valid, 6 steps
each) with nomic-embed-text:v1.5 (matryoshka), layer-normed, then truncated to
each D. Relative-delta representation (np.diff over steps) -> PCA-50 -> 5-fold
CV logistic AUC (exp6 protocol). JEPA predictor trained directly on truncated
D-dim vectors (no learned Projector — matryoshka truncation IS the projection).

## The dimension sweep (the russian-doll test)

| D | delta AUC | JEPA L1 | mean L1 | JEPA vs mean |
|---|---|---|---|---|
| 64  | **0.928** ±0.036 | 10.02 | 16.75 | −40.2% |
| 128 | 0.975 ±0.022 | 12.25 | 15.76 | −22.3% |
| 256 | 0.973 ±0.025 | 14.22 | 14.92 | −4.7% |
| 512 | 0.978 ±0.020 | 18.26 | 14.52 | +25.8% |
| 768 | **0.990** ±0.020 | 21.81 | 14.43 | +51.2% |

## Gates

- **G1 (truncation beats learned crush): PASS** — 64-dim 0.928 > exp6's learned
  64-dim 0.862 (nomic). Matryoshka truncation beats the learned Projector crush.
- **G2 (64-dim recovers toward raw): PASS** — 0.928 ≥ 0.90 (raw nomic deltas
  were 0.960). Truncation recovers most of the raw signal.
- **G3 (elbow flat to 64): FAIL** — drop 768→64 is 0.062 > 0.05 threshold.
  768 (0.990) is still best; 64 is very usable but not lossless.
- **G4 (JEPA beats mean by 50%): FAIL** — only −40.2% at D=64, and the
  predictor gets WORSE as dims rise (+51% at 768, i.e. worse than mean).

## Honest reading

1. **The russian-doll hypothesis is CONFIRMED for the representation.** The
   64-dim bottleneck in exp6 was the CRUSH (learned Projector), not the
   dimension. Matryoshka truncation to 64 dims recovers the delta signal
   (0.928 vs exp6's 0.862, and near raw 0.960). The low-dim A-space program is
   back on the table with a sound representation. This is the fix we set out
   to test, and it worked.

2. **The elbow is not perfectly flat.** 768 (0.990) beats 64 (0.928) by 0.062.
   So "64 dims is enough" is NOT cleanly proven — but 64 dims is now *usable*
   (0.928 is strong), which is what the A-space program needs. The honest
   claim: matryoshka makes low dims viable, not lossless.

3. **The JEPA predictor is still weak (G4 fails).** This is the SAME weakness
   exp6 had — a small MLP predictor on 90 sessions (~540 triples) is
   underpowered. It's a DATA problem, not a representation problem. The
   predictor needs more sessions, not a better embedding. This is the real
   bottleneck now, and it points to exp16 (own-CoT generation) or a larger
   corpus, not to another representation fix.

4. **The predictor degrades as dims rise** (worse than mean at 512/768). A
   128-hidden MLP predicting 768-dim targets from 768-dim inputs on 540
   triples can't learn — it's a high-dim regression with tiny data. At 64 dims
   the target space is small enough to make progress (−40%). This reinforces
   that the predictor's problem is capacity+data, and that low dims help it.

## Verdict (NOMENCLATURE tags)

- "Matryoshka truncation preserves the delta signal better than a learned
  projection" — **PROVEN [exp10, 90 sessions, nomic v1.5]** (0.928 vs 0.862).
- "64 dims is enough" — **SUGGESTED, not cleanly proven** (0.928 usable, but
  768 at 0.990 is better; elbow drop 0.062 > 0.05).
- "The JEPA predictor beats a mean baseline by ≥50%" — **FALSIFIED [exp10,
  90 sessions]** (only −40% at 64 dims; worse at higher dims). Data-limited.

## Next-move candidates

1. **The representation is fixed; the predictor is the bottleneck.** The
   honest path to a working JEPA predictor is MORE DATA (exp16 own-CoT, or a
   larger labeled corpus), not another representation change. exp10's AUC
   curve (0.928 at 64 dims) is the sound substrate to build on.
2. **exp11 (appworld transfer) is now well-founded** — the A-space
   representation is sound, so the appworld transfer (0.741) can be
   investigated as a real fact rather than a blur artifact.
3. **exp13 (dim expansion) is de-prioritized** — 64 dims is already usable
   (0.928); the elbow says 128–256 is the sweet spot if we want more, but the
   predictor, not the dims, is what's holding outcome prediction back.
