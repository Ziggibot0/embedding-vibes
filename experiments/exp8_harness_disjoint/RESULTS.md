# Exp8 — RESULTS

Run completed 2026-08-29 01:14 (4779s wall: 65s extract, ~78min embed seeded
from exp7 cache, ~30s eval). 3,000 sessions, 4 evaluable harnesses, 3
benchmarks, 5 models, 52.6% success. All gates were pre-registered in
DESIGN.md before the run; nothing below was post-hoc.

## Main table (AUC)

| feature | in_format | LOHO (harness-disjoint) | LOBO (benchmark-disjoint) | LOMO (model-disjoint) |
|---|---|---|---|---|
| static_full (centroid+final, 1536d) | 0.924 | **0.866** | 0.508 | 0.786 |
| static_pca8 (8-dim projection) | 0.786 | **0.749** | 0.444 | 0.674 |
| meanvel (velocity) | 0.855 | 0.781 | 0.602 | 0.702 |
| tfidf (content control) | 0.943 | **0.919** | 0.567 | 0.846 |
| tags (format control) | 0.618 | 0.514 | 0.460 | 0.440 |
| length (control) | 0.502 | 0.441 | 0.453 | 0.447 |
| tfidf+meanvel | 0.945 | 0.915 | 0.617 | 0.841 |

Calibration (in_format): static_full ECE 0.100 / Brier 0.120; tfidf ECE
0.091 / Brier 0.104; static_pca8 ECE 0.066. LOHO per-harness detail in
`results/exp8_results.json` (weakest harness transfer for static_full:
openai_solo 0.799; strongest: claude_code 0.887, tool_calling 0.906).

## Gates (pre-registered)

- **Gate A (monitor transfers across harnesses): PASS** — static_full
  in_format 0.924 ≥ 0.75 AND LOHO 0.866 ≥ 0.70.
- **Gate B (geometry retains more than lexical under format shift): FAIL**
  — drop(tfidf) 0.024 < drop(static_full) 0.058; the embedding loses MORE
  under harness shift than the bag of words does (difference −0.034, gate
  required ≥ +0.03).
- **Gate C (velocity null recheck): PASS** — tfidf+meanvel 0.945 vs tfidf
  0.943: increment +0.002 ≤ 0.02. Velocity remains null beyond content.
- **Gate D (low-dim transfers): PASS** — drop(pca8) 0.037 ≤ drop(full)
  0.058 + 0.02. The 8-dim projection transfers as well as the full space.

## Findings (per the pre-committed interpretation ladder)

1. **The monitor transfers.** Frozen-probe trepidation trained on 4 harnesses
   and tested on a held-out 5th stays at 0.866 AUC (tags-only: 0.514). The
   harness-invariance *requirement* for a deployable monitor (P1/P3) is met
   by the frozen baseline. Claim: PROVEN [Exgentic, nomic, outcome AUC].
2. **B failed, so A-space starts from zero floor.** Under format shift the
   embedding loses MORE than TF-IDF (drop 0.058 vs 0.024). The frozen
   encoder's "geometry" advantage is content riding, and content survives
   format change better than geometry does. The user's bet — train an
   A-space where formatting blurs — is now the ONLY route to a
   beyond-lexical, format-invariant signal. That is exactly exp9's job.
3. **Surprise (not in the ladder): benchmark-disjoint is catastrophic.**
   LOBO 0.44–0.62 for everything, including TF-IDF (0.567). Success/failure
   signal in this dataset is benchmark-relative: "what trouble looks like"
   differs between AppWorld and SWE-bench so much that probes trained across
   benchmarks don't transfer at all. Scoping consequence: trepidation is
   per-task-family; cross-task claims need per-benchmark calibration or a
   task-conditioned design. This kills naive "universal trepidation" framing
   and must be stated in the paper.
4. Model-disjoint (LOMO) sits between (static_full 0.786, tfidf 0.846):
   failure signal is partly model-specific — consistent with the audited
   literature and with our own decision to calibrate on OUR model's runs.
5. Controls behaved: tags (0.514 LOHO) and length (0.441) carry almost
   nothing across harnesses; the signal is not format markers or verbosity.

## Scope statement (mandatory honesty)

One dataset (Exgentic), one encoder (nomic-embed-text 137M), 3,000 sessions,
5 frontier models, frozen probes only. "Harness transfer" here means
cross-harness LOG classification, not live intervention on our stack. LOBO
catastrophe suggests outcome semantics are benchmark-local; any deployment
claim must be same-benchmark or recalibrated. ECE 0.09–0.12 says calibration
work remains before thresholding (P1).

## What this means for exp9 (the from-scratch A-space encoder)

Run A's exam is now fully specified with baselines: its 8 dims must beat
LOHO 0.602 (tags) for G1, and G3 compares against static_pca8 LOHO 0.749.
Gate B's failure is the REASON exp9 exists: nothing frozen gives you
format-invariant geometry; it has to be trained (MLM + Barlow Twins on
format-rendering pairs + prefix consistency).