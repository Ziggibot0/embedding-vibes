# Experiment 2a: Chunking Strategy Ablation

## Question
Which temporal chunking strategy yields the most signal for fallacy detection in embedding space?

## Design
Take the same fallacy examples + valid controls. Apply 4 chunking strategies. For each:
- Embed each chunk separately → trajectory of embeddings
- Measure: does the trajectory-level signal (max per-chunk AUC, trajectory shape metrics) exceed the whole-statement baseline?

## Chunking strategies to test

1. **Whole-statement** (baseline from exp1) — 1 embedding per example
2. **Sentence-level** — split on sentence boundaries, 1 embedding per sentence
3. **Connective-based** — split on logical connectives (therefore, because, so, thus, hence, which means, consequently, since, as a result)
4. **Clause-level** — split on commas, semicolons, and connectives (finer than sentences)

## Metrics per strategy
- Number of chunks per example (mean, std)
- Per-chunk fallacy-type classification accuracy (does any single chunk classify better than the whole?)
- Trajectory shape: path length, direction changes, terminal point distance to fallacy centroid
- Binary fallacy-vs-valid AUC using:
  a) Best single chunk
  b) Trajectory aggregate (mean of chunk embeddings)
  c) Trajectory shape features (path length, curvature)

## Prediction
Connective-based chunking should yield the most signal because fallacies live in the logical transitions marked by connectives. The "therefore" step in a false causality argument should embed closer to false causality than the whole statement.

## What would falsify this
If no chunking strategy beats the whole-statement baseline, the signal is truly holistic (in the whole text, not the parts) and trajectory analysis adds nothing.