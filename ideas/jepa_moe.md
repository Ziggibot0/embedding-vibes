# JEPA MoE — a language model made of tiny trepidation experts

## Idea
Instead of one big model, a collection of tiny 6M-param A-space encoders, each
specialized on a different process domain (agent sessions, code traces, math
reasoning, web browsing, medical conversations, robot plans). A router reads
the input and selects which expert(s) to activate. The experts communicate in
embedding space, not token space — each outputs its 8-dim A-space vector, and
a lightweight combiner aggregates.

## Why it's not just "another MoE"
Standard MoE routers select subnetworks that produce TOKENS. This MoE's
experts produce GEOMETRY — each is a JEPA predictor that knows one domain's
trajectory shapes. The output isn't the next token; it's the predicted
destination in that domain's A-space, plus a trepidation score. The "language
model" here is really a *process monitor* that switches lenses depending on
what kind of work it's looking at.

## What we already have
- exp6: JEPA predictor machinery (forecast future embedding from prefix,
  beat mean-baseline by 62-73% L1). This is the per-expert predictor.
- exp9: from-scratch 8-dim A-space encoder with BT format-invariance. This is
  the per-expert encoder.
- exp8: harness-disjoint transfer results showing when signal transfers and
  when it dies (benchmark-disjoint collapse = need per-domain experts).

## Architecture sketch
```
Input session prefix
      |
  [Router] — tiny classifier: which domain is this?
     / | \
   E1 E2 E3  ... En  (6M each, frozen after training)
   |  |  |
  8d 8d 8d           (each expert's A-space vector + trepidation)
   \ | /
 [Combiner] — weighted by router probabilities
      |
  trepidation τ + predicted destination φ
```

## Why it's cheap
- Each expert: 6M params, trained once on its domain corpus.
- Router: tiny (logistic on 8-dim embeddings, or a small MLP).
- Combiner: linear, 8×n params.
- Inference: activate 1-2 experts per query (sparse, like MoE).
- Total params: 6M × n_domains + router. For 10 domains = 60M. That's a
  small model that covers 10 different process types.
- Training is embarrassingly parallel — each expert trains independently.

## Why exp8's LOBO collapse supports this
The benchmark-disjoint collapse (0.44-0.62 for all features) says "universal
trepidation is false." That's the MoE's justification: you NEED multiple
experts because one space can't cover all domains. The collapse is the
evidence that the architecture is necessary, not optional.

## Relationship to the personalized trepidation idea
The MoE is the macro version (different domains), personalization is the micro
version (different users within a domain). Both are "the universal model is
wrong; specialize." The MoE is domain-specialization; the personalization head
is user-specialization. They compose: a user's agent runs are routed to the
right domain expert, then the personalization head calibrates for that user.

## Risks
- Router quality: if the router misclassifies the domain, the wrong expert
  fires. Need a fallback or an "I don't know" class.
- Expert coverage: need enough domains to be useful. 2-3 is a demo; 10+ is
  a product.
- Combiner: simple weighted average may not capture cross-domain interactions
  (e.g., a session that's half coding, half web browsing). Hierarchical
  routing could help but adds complexity.

## Un-park trigger
exp9 attempt #2 or #3 passes G1 (signal exists in A-space) AND we have
2+ trained experts from different domains. First test: does a 2-expert MoE
(appworld expert + swebench expert) beat a single generalist encoder on
benchmark-disjoint evaluation? If yes, the MoE architecture is validated and
the LOBO collapse becomes a feature, not a bug.

## Relationship to the dissertation
This is the architecture chapter. Chapter 1 (this paper) proves A-space
exists. Chapter 2 (generalization) shows it applies to any process domain.
Chapter 3 (MoE) shows the domains compose into a practical system. Chapter 4
(personalization) shows user-level calibration. Each chapter builds on the
last; each is independently publishable.