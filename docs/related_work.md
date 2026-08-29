# Related Work — updated 2026-08-29

## Direct neighbors (must cite + differentiate)

### 1. Latent-Trajectory Signals (ICLR 2026, Microsoft)
"Tracing the Traces" — uses hidden-state trajectory geometry (net change,
cumulative change, aligned change) to predict reasoning correctness. AUC
~0.71-0.74, training-free, on math reasoning (AIME, GPQA, TSP).

**Our differentiation**: they probe frozen hidden states of the GENERATING
model (requires white-box access to internal layers). We embed session LOG
TEXT with a separate frozen/small encoder (black-box, harness-agnostic).
They study math reasoning; we study agent sessions. They don't test
cross-harness transfer or content controls. They don't train an encoder.
**Key gap they leave that we fill**: no TF-IDF lexical control, no
harness-disjoint evaluation, no from-scratch encoder.

### 2. Process Reward Models (PRMs) — survey 2025, multiple papers
SWE-TRACE (rubric PRM for SWE agents), Agent-RRM (reasoning reward model),
ClawTrack (trace-level process scoring), WebArbiter (web agent PRM),
PRInTS (long-horizon info-seeking PRM).

**Our differentiation**: PRMs score individual steps using an LLM or trained
verifier (expensive, requires reasoning). We score session-level geometry
with a tiny probe (cheap, no reasoning). PRMs operate in token/logit space;
we operate in embedding space. PRMs don't test format-invariance across
harnesses. **Our negative result (velocity = 0 beyond content)** is a
challenge to PRM assumptions: if trajectory shape doesn't carry signal
beyond words, what exactly are PRMs learning?

### 3. Agent Trajectory Analysis survey (2025-2026)
"A Survey for LLM Agent Trajectory Analysis" (Chen et al.) + AgentDiagnose
(EMNLP 2025 demos) + trajectory reduction (FSE 2026).

**Our differentiation**: these works analyze trajectory content (what the
agent did). We analyze trajectory GEOMETRY (where the embeddings moved).
They attribute failure to specific steps; we predict failure from shape.
They need full trajectories; we work from prefixes.

### 4. JEPA for text (LLM-JEPA, BERT-JEPA, UWM-JEPA — 2024-2025)
All use JEPA as a TRAINING OBJECTIVE for LLMs. None use JEPA to train a
PREDICTOR that forecasts trajectory destinations. T-JEPA (2024) does
trajectory similarity but for GPS traces, not agent logs.

### 5. Barlow Twins (Zbontar & LeCun 2021)
Original is vision. BT-SR (2025) applies it to sequential recommendation.
We apply it to format-invariance: same action, different rendering -> same
point. Novel application of the redundancy-reduction principle.

## What NOBODY has done (our actual contribution space)

1. **Content-vs-geometry decomposition for agent traces** — no prior work
   reports TF-IDF as a control against embedding-probe AUC on agent sessions.
   This is the methodological contribution: the decomposition discipline.
2. **Harness-disjoint evaluation** — no prior work evaluates trace monitors
   across harnesses with metadata-aware splits. Transfer is assumed, not
   tested.
3. **Benchmark-disjoint collapse** — no prior work reports that outcome
   signal dies at the benchmark boundary. This is a finding that constrains
   every PRM and trace-monitor paper that claims generality.
4. **From-scratch low-dim format-invariant encoder for agent sessions** —
   no prior work trains a tiny (6M) encoder with BT on synthetic format pairs
   for agent-trace representation.
5. **Label-leakage case study (Run B)** — no prior work documents the
   transductive leakage failure mode in trace-monitor training. This is
   a cautionary methodological contribution.

## The "anything-space" generalization (discussion section material)

If the method works for agent sessions, the formal structure generalizes:
- Define a "process domain" D with a set of sequential artifacts (sessions,
  code traces, medical visits, robot paths, conversations).
- Define "format variants" F — different serializations/renderings of the
  same underlying process.
- Train an encoder E: artifact -> R^k with BT on (same process, different
  format) pairs + MLM for domain competence.
- The claim: E learns a "D-space" where process-relevant structure is
  format-invariant.
- Test: cross-format retrieval, cross-domain transfer, prefix consistency.

This is a generalization from "agentic space" to "any process space where
format varies but intent doesn't." Each domain is a separate experiment; the
architecture and the controls are the same. The dissertation can frame
agent sessions as the first instance of a general method.

## Target venues (revised)

- **Main track**: ICLR (if A-space works — method + theory), NeurIPS (if
  the controls/decomposition is the headline — empirical methodology)
- **Findings/Workshop**: EMNLP Findings (if the fallacy/language angle
  dominates), ICML Reasoning Workshop (if the trajectory-geometry angle
  dominates)
- **Dissertation**: all of the above become chapters; the generalization
  is the concluding chapter.