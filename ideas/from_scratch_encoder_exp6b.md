# From-scratch text encoder + JEPA (exp6b — ON HOLD, inherited from exp7)

## Idea
Train a small from-scratch encoder (MLM + Barlow Twins + JEPA) on trajectory
text; use it as the A-space substrate. Already built + iGPU-trained once
(exp6_joint_jepa/train_from_scratch.py).

## Why parked
Its justification was exp5's delta-dominance, which did not transfer (exp7).
Any learned encoder must now beat: frozen nomic centroid probe (0.900), PCA-8
state (0.757), AND pass harness-disjoint gates — in that order (ROADMAP P2
gate). Until format-crystallization data exists (P0/P2), training it is
premature.

## Un-park trigger
P2's gate fails with frozen-encoder features; a trainable encoder becomes the
remaining hypothesis for A-space.
