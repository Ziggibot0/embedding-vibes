# Exp12 — Honest outcome supervision (per-split encoder retraining)

**Status:** NOT STARTED · placeholder (EXPERIMENTS.md)

Fill in before running. The ONLY lever that directly injects outcome signal.
Retrain the encoder PER SPLIT (5× cost) so the outcome head never sees
held-out labels. Exp9's Run B was the contaminated version (leakage, 0.985).
This is the honest retrain-per-fold version.

Gate: only pursue if exp10 + exp11 justify it (has a sound representation).
