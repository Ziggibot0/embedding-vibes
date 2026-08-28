# embedding-vibes

JEPA predictor for CoT / reasoning trajectories. Embedding-space pattern detection for logical fallacies and failure prediction.

> **SCOPE (2026-08-28):** This project does **NOT** use the NPU (XDNA 2). The wave-interference aggregation idea is **TABLED** — it is a separate project, not part of this one. Compute runs on CPU/iGPU only. See DESIGN.md for details.

## Core idea

LLMs generate locally coherent reasoning that globally fails. A separate tiny model watches the embedding trajectory of a reasoning chain and detects "vibes" — geometric patterns in embedding space that resemble known failure shapes. No reasoning, no logic, just pattern matching on shapes.

- **Observation**: CoT step t embedded into z_t
- **Predictor**: learns z_{t+k} = f(z_t, action) — predicts where the reasoning is heading
- **Collapse prevention**: Barlow Twins (cross-correlation → identity, off-diagonal penalty)
- **Trepidation**: high predictive variance = "I don't know where this goes"
- **Vibes**: predicted trajectory shape compared against success/failure clusters

## Architecture

```
Base model (35B-A3B on iGPU/Vulkan)
    │
    ├── generates K candidate CoT paths
    │
    ▼
Embed each step → z_t (frozen encoder)
    │
    ▼
JEPA predictor (tiny, few M params)
    │
    ├── predicts z_{t+k} for each path
    │
    ▼
Aggregate across K paths (CPU/iGPU)
    │
    ├── aligned paths → high confidence vibe
    ├── divergent paths → trepidation
    │
    ▼
Steering signal back to base model
```

## Datasets

- **AgentTrove** (open-thoughts) — 1.7M agent trajectories with reward labels
- **Exgentic/agent-llm-traces-v2** — 10K agent runs, 6 benchmarks, OTel format
- **ISETrace** — 23K OS-agent trajectories with real tool execution
- **AgentTrace** — 1,400 traces with execution telemetry

## Anti-collapse

Barlow Twins loss: cross-correlation matrix between predicted and actual embeddings → identity matrix.
- Diagonal → 1 (invariance: predicted matches actual)
- Off-diagonal → 0 (redundancy reduction: each dim carries independent info)
- Structurally blocks collapse without negative pairs.

Rust crate: `jepa_core::collapse::BarlowTwins` (docs.rs)

## Gate experiment

Before building anything: embed 100 CoT traces with known fallacies + 100 valid CoT traces. Plot trajectory geometry. Check if fallacy trajectories have different shapes (loops, curvature, direction-change frequency). If yes → the embeddings carry the signal. If no → dead.

## Hardware

- Ryzen AI MAX 385 (Strix Halo): 8x Zen 5, Radeon 8050S iGPU (32 CU RDNA 3.5), XDNA 2 NPU (50 TOPS)
- ~32GB unified LPDDR5X (~256 GB/s)
- Base model: qwen35moe 35B-A3B on Vulkan (60.6 t/s gen)
- Predictor: tiny, CPU
- Encoder: frozen qwen late-layer hidden states or nomic-embed-text

## Status

Pre-experiment. Gate: does embedding geometry distinguish fallacy from valid reasoning?