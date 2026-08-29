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

## Verdict (NOMENCLATURE tags)

- "A from-scratch 8-dim space carries outcome signal above format-tags" —
  **PROVEN [weak, attempt #1]** (0.657 vs 0.602).
- "It beats matched-dim lexical" — **FALSIFIED [attempt #1, 8 dims, 6000
  steps]** (0.657 vs 0.771).
- "No partial collapse" — FALSIFIED (2 dead dims, direction collapse).
- "Format crystallization achieved" — UNTESTED-in-effect (margin 0.029 ≈
  degenerate space; BT did not separate steps).
- "Benchmark-disjoint transfer above frozen features" — SUGGESTED (0.637 vs
  0.602 best; needs bootstrap + attempt #2 replication).
- "Outcome supervision transfers spectacularly" — artifact of leakage;
  excluded.

## Next-move candidates (for attempt #2, not yet run)

1. Open the space: λ_bt up (1.0 → 5.0) or BT on all rendering PAIRS per
   batch rather than one pair; add batch-variance floor; consider L2-
   normalizing z before BT so direction must carry info.
2. Beat the lexical bar (0.771): more steps (GPU headroom: 4x), the 22k
   unlabeled corpus for MLM, 16-32 dims first then re-test 8.
3. If outcome supervision is wanted honestly: encoder must be retrained
   per split (train folds only) — 5× cost, deferred.
4. Bootstrap the LOBO result; if appworld 0.72 replicates, that's the
   paper's first positive A-space fact.