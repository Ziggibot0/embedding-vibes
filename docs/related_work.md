# Related Work — Annotated Bibliography

## 1. Direct precursors (the field is converging on this)

### Microsoft: LLM Reasoning as Trajectories (ACL 2026)
- arXiv: 2604.05655
- Code: github.com/slhleosun/reasoning-trajectory
- **KEY FINDING**: CoT reasoning traces structured trajectories through representation space. Each reasoning step occupies a distinct, linearly separable region. Early steps are correctness-invariant. Late steps diverge systematically between correct and incorrect solutions.
- **CORRECTNESS PREDICTION**: Linear classifiers on late-step trajectory features achieve ROC-AUC 0.87 for predicting final-answer correctness BEFORE the answer is emitted.
- **STEERING**: Trajectory-based interventions improve accuracy +7.6% on 6-step, +7.69% on 7-step problems with >97% preservation rate.
- **RELEVANCE**: This is the closest existing work. They proved the signal exists and used linear probes + steering. Our JEPA predictor would be a LEARNED version of their linear probe — predicting trajectory destination rather than just classifying current state. They did the existence proof; we'd be extending to learned prediction + trajectory aggregation.

### Latent-Trajectory Signals (ICLR 2026)
- arXiv: 2510.10494 (Microsoft Research)
- **KEY FINDING**: Three signals — total representational change, cumulative intermediate change, alignment with final state — predict solution accuracy. Reduces token usage by 70% while improving accuracy 2.6% over majority voting.
- **RELEVANCE**: They use hand-crafted trajectory signals. Our JEPA predictor would LEARN these signals automatically. Their work validates that trajectory geometry carries correctness signal. Direct precedent for the "beam search in embedding space" application.

### The Geometry of Reasoning: Flowing Logics (2025)
- arXiv: 2510.09782, 19 citations
- Code: github.com/MasterZhou1/Reasoning-Flow
- **KEY FINDING**: LLMs internalize logical structure as higher-order geometry in representation space. Same logical proposition with different semantic carriers traces similar geometric paths. Training via next-token prediction can internalize logical invariants as geometry.
- **RELEVANCE**: This is the theoretical basis for our claim. If logical structure = geometry, then logical fallacies = geometric deviations. Our predictor detects the geometric shape, not the logic.

## 2. JEPA for text (the method is new but not unprecedented)

### LLM-JEPA (2025)
- arXiv: 2509.14252, Code: github.com/rbalestr-lab/llm-jepa
- First JEPA objective for LLMs. Uses text/code view pairs (e.g., issue description + code diff). Outperforms standard LLM training across Llama3, Gemma2, OpenELM, OLMo.
- **RELEVANCE**: Proves JEPA works for text. But they use it for training, not for trajectory prediction. Different application.

### BERT-JEPA (2026)
- Uses JEPA to reorganize CLS embeddings into language-invariant "thought space"
- **RELEVANCE**: Shows JEPA can reshape text embedding space. Not trajectory prediction.

### UWM-JEPA (2026)
- arXiv: 2605.25313
- **KEY**: First JEPA where latent is a BELIEF (density matrix) not a point. Predictor is a unitary operator. Handles uncertainty natively.
- **RELEVANCE**: Directly relevant to trepidation. Their density-matrix latent IS the uncertainty representation we want. Could replace our heteroscedastic regression approach with something more principled.

### ThinkJEPA (2026)
- arXiv: 2603.22281
- VLM-guided JEPA with dual-temporal pathway: dense JEPA branch + VLM thinker branch
- **RELEVANCE**: Architecture pattern for combining a fast predictor with a slow reasoner. Analogous to our predictor + base model setup.

## 3. Embedding geometry and reasoning quality

### CraEG: Crowding-Aware Sampling (ICML 2026)
- arXiv: 2601.22536
- **KEY**: Embedding-space crowding (probability mass concentrating on geometrically close tokens) is statistically associated with reasoning FAILURE. Geometry-guided reweighting of next-token distribution improves reasoning.
- **RELEVANCE**: Validates that geometric properties of embeddings correlate with reasoning success/failure. Crowding is one such signal — our predictor would learn MORE signals from the trajectory shape.

### Reasoning emerges from constrained inference manifolds (2026)
- arXiv: 2605.08142
- **KEY**: Inference-time representations collapse onto low-dimensional manifolds. Effective reasoning requires adequate expressivity + spontaneous manifold compression + preservation of non-degenerate information volume. Models outside this regime show pathological inference dynamics.
- **RELEVANCE**: Theoretical framework for why trajectory geometry matters. Pathological dynamics = bad reasoning = detectable geometric signature. Directly supports our thesis.

## 4. Fallacy detection (the application domain)

### Logical Fallacy Detection (EMNLP 2022)
- Jin et al., arXiv: 2202.13758
- Dataset: tasksource/logical-fallacy (3,761 rows, HuggingFace)
- **KEY**: LLMs perform poorly at fallacy detection (~34% accuracy). Structure-aware classifiers outperform LLMs by 5.46% F1.
- **RELEVANCE**: Shows LLMs can't detect fallacies via reasoning — our approach bypasses reasoning and uses geometric pattern matching instead.

### AID-LF: Atomic Instruction Dataset for Logical Fallacies (2025)
- arXiv: 2510.09970, Code: github.com/olivianxai/AID-LF
- Decomposes fallacy detection into atomic binary decision steps
- **RELEVANCE**: Dataset resource for labeled fallacies. Could provide the labeled CoT traces for our gate experiment.

### Additional fallacy dataset: 20,036 examples, 13 fallacy types, 2015-2025 (scidb.cn)

## 5. PRMs (what we're NOT, and why we're different)

### Survey of Process Reward Models (2025)
- arXiv: 2510.08049
- PRMs score individual reasoning steps in token space. Our approach scores trajectory DESTINATION in embedding space. Fundamentally different mechanism.

## 6. Quantum cognition / wave interference (the speculative bridge) — TABLED

> **SCOPE (2026-08-28):** This section is the theoretical grounding for the wave-interference idea, which is TABLED as a separate project. It is retained here for reference only and is NOT part of the current embedding-vibes scope.

### Quantum Interference in Cognition (2012, Aerts et al.)
- Two superposed layers in human thought: classical logical + quantum conceptual. The quantum conceptual layer is responsible for deviations from classically expected reasoning.
- **RELEVANCE**: Theoretical precedent for wave interference in cognition. (Tabled — separate project.)

### Quantum-like Cognition and AI (2026, Springer)
- Superposition and interference can improve sampling efficiency. Entanglement enables encoding correlations difficult to represent classically.
- **RELEVANCE**: Motivates wave interference as a computational mechanism for AI, not just a metaphor. (Tabled — separate project.)

## 7. Agent trajectory datasets (training data)

### AgentTrove (open-thoughts)
- 1.7M rows, 219 source datasets, reward labels (1.0/0.0). ShareGPT format.
### Exgentic/agent-llm-traces-v2
- 10,057 agent runs, 6 benchmarks, OTel format, success/score labels.
### ISETrace
- 23,132 OS-agent trajectories, real tool execution, success flags per call.
### AgentTrace (pagarsky)
- 1,400 traces with execution telemetry.