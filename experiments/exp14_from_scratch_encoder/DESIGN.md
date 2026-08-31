# Exp14 — From-scratch encoder (scaled up)

**Status:** NOT STARTED · placeholder (EXPERIMENTS.md)

Fill in before running. The 6M from-scratch transformer (exp9) and 6.37M joint
encoder (exp6b) were too small / low-dim. Scale up the from-scratch encoder
trained WITH the JEPA predictor (MLM + Barlow + JEPA) at higher dims/capacity.

Bar: if exp10's matryoshka gives 64 dims for free, exp14 must beat that baseline
to be worth it. Only pursue if TASK-SPECIFIC representation (format + outcome)
that truncation can't provide is needed.
