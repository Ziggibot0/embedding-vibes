# Exp9 — RESULTS (attempt #1, overnight run 2026-08-29)

Training: 4 epochs × 1500 steps × batch 24, on iGPU (ROCm torch 2.9.1, ~1.3s/step,
~2.7h per run). Run A pure (MLM+BT+prefix), Run B added the outcome head.
Eval: upgraded eval.py (batched, cross-format diagnostics, matched-dim lexical
control, bootstrap CIs) — all added BEFORE any exp9 numbers existed.

## Run A — the honest artifact (no outcome supervision anywhere)

| metric | value | reference |
|---|---|---|
| in_format AUC | 0.698 | nomic static 0.924, tfidf 0.943 |
| LOHO mean | **0.657** | tags-only 0.602, tfidf-pca8 **0.771**, nomic pca8 0.749 |
| LOHO per harness | cc 0.643 [.60,.68], os 0.723 [.69,.76], sm 0.769 [.74,.80], tc 0.493 [.45,.54] | |
| LOBO mean | 0.637 (appworld 0.722!) | best frozen feature was meanvel 0.602; appworld best was 0.622 |
| per-dim variance | 2 dims near-dead (0.01, 0.02) | G4 threshold 0.2 → **FAIL** |
| cross-format (step) | within 1.000 vs between 0.971, margin **0.029** | wanted: within >> between |

Gates: **G1 PASS** (0.657 ≥ 0.602) · G2 PASS (drop 0.041 ≤ tfidf drop +0.02) ·
**G3 FAIL** (0.657 < 0.699) · **G4 FAIL** (partial collapse).

Honest reading:
1. Signal exists but is WEAK and inconsistent (one harness below 0.5 —
   direction flip; the probe transfers worse than chance there).
2. **Does not beat the matched-dimension lexical control** (0.657 vs
   tfidf-pca8's 0.771). At equal compression, a bag of words is a better
   representation than our attempt #1. A-space claim: NOT YET.
3. Space is directionally degenerate: every step lands in nearly the same
   direction (within 1.000 / between 0.971). MLM dominates; Barlow Twins on
   8 dims did not open the space up. Crystallization margin is ~noise.
4. One genuine bright spot: **best benchmark-disjoint transfer in the
   project** (0.637 mean, appworld 0.722 vs every frozen feature dying at
   0.44–0.62). Mild, unbootstrapped, but it is the only thing anywhere that
   moved LOBO. Worth a targeted attempt #2.

## Run B — CONTAMINATED, caught by the two-run discipline

| metric | value |
|---|---|
| LOHO mean | 0.985 (!) |
| LOBO mean | ~0.94 (!!) |
| cross-format margin | 0.914 (between 0.064 ≈ orthogonal) |

These numbers are **not evidence of anything except label leakage**. The
outcome head's gradients flowed through the encoder while it trained on ALL
3,000 labeled sessions — including every session later used as eval test
sets. The encoder had already seen each test session's label. LOBO 0.94–0.99
against a world where every frozen feature gets 0.44–0.62 is the fingerprint
of transductive label memorization, not transfer. This is precisely the
failure mode the A/B design pre-registered as the leakage smell ("if B > A,
the difference is suspicious") — the discipline worked. Run B's checkpoint is
retained as a demonstration artifact; its eval numbers are excluded from all
claims.

## Attempt #3 (2026-08-30) — 60k corpus + VICReg anti-collapse hinge

Bet: attempt #1's two failures were (a) a collapsed space (G4) and (b) too
little data. Attempt #3 trained on a 60k-session corpus (10× attempt #1's
6k) and added a VICReg variance hinge (penalize any dim whose std < γ=1.0)
so Barlow Twins could not be satisfied by collapsing to a point. Run A,
pure (MLM + format-pair BT + VICReg + prefix consistency, no outcome head),
3 epochs × 2000 steps, iGPU.

| metric | attempt #1 | attempt #3 | reference |
|---|---|---|---|
| in_format AUC | 0.698 | 0.674 | tfidf 0.943 |
| LOHO mean | 0.657 | **0.614** | tfidf-pca8 0.771, nomic pca8 0.749 |
| LOHO per harness | cc .643 os .723 sm .769 tc .493 | cc .566 os .756 sm .609 tc .527 | |
| LOBO mean | 0.637 | 0.562 | best frozen 0.602 |
| LOBO appworld | 0.722 | **0.741 [.71,.77]** | best frozen 0.622 |
| per-dim variance | 2 dead (0.01, 0.02) | min 0.170 (dim5) | G4 threshold 0.2 |
| cross-format margin | 0.029 (degenerate) | **0.583** | wanted: within >> between |

Gates: **G1 PASS** (0.614 ≥ 0.60) · **G2 FAIL** (drop 0.060 > tfidf drop
0.024 + 0.02) · **G3 FAIL** (0.614 < 0.699) · **G4 FAIL** (dim5 var 0.170 < 0.2).

Honest reading:
1. **The anti-collapse fix worked mechanically.** VICReg hinge fired at
   start (vic=1.67) and drove to 0.000 by it200; the space opened up —
   cross-format margin went 0.029 → 0.583, the two dead dims are gone, and
   dims now carry real spread (min 0.17 vs attempt #1's 0.01). G4 fails by
   a hair (0.170 vs 0.2) but collapse is no longer the operative problem.
2. **Opening the space did NOT make it more informative.** LOHO got worse
   (0.657 → 0.614) and still does not beat the matched-dim lexical control
   (0.771). A healthier, better-separated representation is *less*
   predictive of outcome. The 10× corpus + VICReg bet is falsified.
3. **VICReg was non-binding after it200** (vic=0.000): variance already
   exceeded γ, so the hinge contributed nothing to shaping — real shaping
   came from MLM + BT, and that did not produce outcome signal. Healthy
   variance ≠ informative representation.
4. **The one replicated positive fact:** appworld LOBO 0.741 [0.71, 0.77],
   bootstrapped, second run in a row above every frozen feature (0.44–0.62).
   Mean LOBO (0.562) is dragged down by browsecompplus (0.485) and swebench
   (0.460), both at/below chance. The transfer signal is real but
   benchmark-specific, not general.

## Verdict (NOMENCLATURE tags)

- "A from-scratch 8-dim space carries outcome signal above format-tags" —
  **PROVEN [weak, attempts #1 & #3]** (0.657, 0.614 vs 0.602).
- "It beats matched-dim lexical" — **FALSIFIED [attempts #1 & #3, 8 dims]**
  (0.657, 0.614 vs 0.771). 10× corpus + VICReg did not close the gap.
- "No partial collapse" — FALSIFIED (attempt #1: 2 dead dims; attempt #3:
  min var 0.170 < 0.2, though collapse is no longer the operative problem).
- "Format crystallization achieved" — **PROVEN [attempt #3]** (margin 0.583,
  within 0.999 vs between 0.416). The space now separates steps cleanly.
- "Benchmark-disjoint transfer above frozen features" — **SUGGESTED,
  replicated [attempts #1 & #3]** (appworld 0.722, 0.741 [.71,.77] vs best
  frozen 0.622). Bootstrapped in #3; benchmark-specific, not general.
- "Outcome supervision transfers spectacularly" — artifact of leakage;
  excluded.

## Next-move candidates (attempt #4, not yet run)

1. The space is now healthy but uninformative — the bottleneck is no longer
   collapse, it is that MLM+BT shaping does not produce outcome signal.
   Options: (a) honest outcome supervision (retrain encoder per split, 5×
   cost) — the only lever that directly injects the target signal; (b) more
   dims (16–32) then re-test 8; (c) accept the A-space claim is about
   *format invariance* (now proven) rather than outcome prediction.
2. The appworld transfer (0.741) is the paper's one positive A-space fact —
   worth a dedicated attempt to understand *why* it transfers there and not
   on browsecompplus/swebench, before treating it as general.
3. If outcome prediction is the goal, the honest path is per-split encoder
   retraining (deferred, 5× cost) — the current architecture cannot claim
   outcome signal without it.