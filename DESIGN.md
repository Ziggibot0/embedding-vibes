# embedding-vibes — design notes

## The core claim

Logical fallacies have geometric signatures in embedding space. A predictor can learn these shapes without understanding logic — the same way humans get "vibes" about something being off without articulating why.

## What this is NOT

- Not a logic checker. It doesn't verify arguments.
- Not a reasoning model. It doesn't think.
- Not a PRM (process reward model). PRMs score individual steps in token space. This scores the trajectory's predicted destination in embedding space.

## What it IS

A pattern detector. "This trajectory's shape resembles trajectories that failed." Swirls of numbers that arrange themselves into shapes that a different mechanism can reason about.

## Wave interference

Vector addition of embeddings IS interference. Not a metaphor — literal physics applied to vectors.

- K candidate reasoning paths → K predicted embedding trajectories
- Constructive interference (aligned trajectories) → large resultant → high confidence vibe
- Destructive interference (opposing trajectories) → small resultant → trepidation
- NPU computes the interference (batched matrix math, 50 TOPS sitting idle)

## Gate experiment

Take 100 CoT traces with known fallacies + 100 valid CoT traces. Embed each step. Plot the geometry:
- Do fallacy trajectories loop back on themselves? (circular reasoning literally circular?)
- Do they have different curvature?
- Different direction-change frequency?
- Different "frequency" of oscillation through embedding space?

If fallacy trajectories look geometrically different → the idea has legs.
If indistinguishable → embeddings don't carry the signal → dead.

## Steering options (ranked by tractability)

1. **Beam search in embedding space**: generate K candidates, predict each trajectory, keep the one heading toward success cluster. No intervention in generation. Easiest.
2. **Flag and re-prompt**: "your reasoning appears to be heading toward a known failure pattern, reconsider." Simple, no gradient flow needed.
3. **Logit steering**: penalize tokens toward success-cluster direction. Needs gradient flow into base model. Hard with frozen encoder.
4. **Full steering**: intervene in generation at each step. Hardest, unsolved.

## Open questions

- Does embedding geometry actually distinguish fallacy from valid reasoning? (gate experiment)
- Does the signal transfer across models? (GPT-5.2 failure patterns vs qwen35moe failure patterns)
- Is wave interference more informative than static vector comparison? (needs ablation)
- Can the NPU actually run the interference computation fast enough to be useful during generation?
- What frequency/phase means for a trajectory through embedding space (not yet defined)