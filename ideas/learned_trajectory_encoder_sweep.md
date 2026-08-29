# Learned trajectory/shape encoder + dimension sweep (parked)

## Idea
exp6 suggested sweeping PROJ_DIM in {128, 256, 512} for the learned encoder;
exp5b suggested the elbow is <= 8 dims for PCA'd raw deltas. Sweep properly
across both learned and PCA representations, on real data with controls.

## Why parked
Same as exp6b: the comparison baseline (frozen probes) must be established
under harness-disjoint splits first (P0-P2), or the sweep measures a confound.

## Un-park trigger
P2 passes its gate with PCA features and we need density (higher dims) for
retrieval/waymarking quality.
