# embedding-vibes — design notes

## Scope (2026-08-28)

- **No NPU.** The XDNA 2 NPU is out of scope for this project. All compute runs on CPU/iGPU.
- **Wave interference is TABLED.** The "vector addition of embeddings IS interference" idea is a separate project, not part of embedding-vibes. This project does not use interference aggregation; it uses trajectory prediction + aggregation on CPU/iGPU.

## The core claim

Logical fallacies have geometric signatures in embedding space. A predictor can learn these shapes without understanding logic — the same way humans get "vibes" about something being off without articulating why.

## What this is NOT

- Not a logic checker. It doesn't verify arguments.
- Not a reasoning model. It doesn't think.
- Not a PRM (process reward model). PRMs score individual steps in token space. This scores the trajectory's predicted destination in embedding space.

## What it IS

A pattern detector. "This trajectory's shape resembles trajectories that failed." Swirls of numbers that arrange themselves into shapes that a different mechanism can reason about.

## Aggregation (in scope)

K candidate reasoning paths → K predicted embedding trajectories. Aggregate on CPU/iGPU:
- Aligned trajectories → high confidence vibe
- Divergent trajectories → trepidation

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
- Does trajectory aggregation add signal over static vector comparison? (needs ablation)
- What frequency/phase means for a trajectory through embedding space (not yet defined; tied to the tabled wave-interference project)