# Exp9 — A-Space Encoder: the from-scratch 8-dim session model (USER-DIRECTED)

This is the project's namesake artifact. NOT a probe on someone else's
embeddings — a small transformer we train ourselves, on raw session logs,
that maps a step (or session) into an 8-dimensional agentic space.

## What it is

- Trunk: 4-layer transformer, d_model 256, ~6M params (exp6b architecture,
  DML-tested). Our own BPE tokenizer (8k vocab) trained on our session corpus.
- Head: linear projection to **8 dims** = the agentic-space coordinate z_t.
- Session vector: mean of step vectors (centroid — exp7b's signal carrier).
- NO frozen encoder anywhere in the final artifact.

## Losses (all three user-specified)

1. **MLM (fill-in-the-blank)**: masked-language modeling on step text —
   language competence so the 8 dims aren't a random projection.
2. **Barlow Twins (anti-collapse)**: on FORMAT-RENDERING PAIRS — the same
   step text re-serialized into two different synthetic harness formats
   (chat / xml / json / otel-ish / terse). Cross-correlation → identity.
   This is the crystallization objective at step level: same action,
   different formatting → same point. Actions crystallize, formatting blurs.
3. **Prefix consistency (fill-in-the-blank, session level)**:
   mean(z_{1..t}) must match sg(mean(z_{1..T})) for random t — the user's
   own "hide part of the chain, same 8-dim embedding" requirement. This is
   what makes live monitoring (P1) well-defined.
4. **Outcome head (optional, --outcome)**: linear success/fail head on the
   session vector, trained on labeled sessions.

## The two-run discipline (pre-registered)

Label leakage risk: the outcome head sees ALL labeled harnesses, including
ones later held out for eval. So we train TWO encoders and report both:
- **Run A: pure MLM+BT+prefix** (no outcome supervision anywhere)
- **Run B: + outcome head**
If Run B transfers better than Run A on harness-disjoint eval, the difference
is suspicious (leakage smell); if equal, outcome supervision is neutral;
the honest artifact is Run A unless B wins by a margin we can defend.

## Eval (the exam — identical protocol to exp8, refit per split)

For each held-out harness h (n>=150): fit LogisticRegression on 8-dim session
vectors from the other harnesses' labeled sessions; test on h. Also in_format
5-fold. Same 3,000 labeled sessions as exp8 (its extract, already on disk).

## Pre-registered gates (frozen)

- **G1 (signal exists)**: Run A LOHO mean AUC >= 0.60 (above tags-only 0.602
  from exp7a — our model must at least match what format tags alone give).
- **G2 (transfers like lexical)**: Run A LOHO drop from its own in_format
  <= TF-IDF's LOHO drop (from exp8) + 0.02.
- **G3 (dream)**: Run A LOHO mean AUC >= (nomic static_pca8 LOHO from exp8)
  - 0.05 — a 6M from-scratch model within 5 points of a 137M pretrained
  encoder's low-dim projection would justify the whole program.
- **G4 (no collapse)**: per-dim variance of z >= 0.2 across eval corpus, and
  BT cross-correlation off-diagonal mean <= 0.3 at end of training.

## Why these choices (short chain)

- 8 dims: user spec + NOMENCLATURE A-space target + exp7c precedent.
- Synthetic format pairs: the dataset has no natural same-episode cross-harness
  pairs; ROADMAP P2 always planned synthetic re-rendering for exactly this.
- Mean-pool session vector: exp7b showed centroid > final as signal carrier.
- Two runs: the audits this session were all about confounds; the outcome-head
  leakage risk is the confound here, so it gets measured, not assumed away.

## Artifacts

- `renderer.py` — canonical steps -> N harness renderings (the pair factory)
- `train.py` — joint trainer (DML/CPU), checkpoints under `results/`
- `eval.py` — the exam (reads checkpoint + exp8's sessions_meta.jsonl)
- `results/` — checkpoints, training log, eval JSON, summary