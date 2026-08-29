# NPU (XDNA 2) acceleration for embedding/predictor ops (TABLED 2026-08-28)

## Idea
Use the 50-TOPS NPU for encoder/predictor inference in the live monitor.

## Why parked
Compute is not the bottleneck (probes are tiny; nomic via Ollama is fast).
OnnxRuntime-XDNA tooling would consume weeks for no measured need.

## Un-park trigger
Live monitor end-to-end latency on iGPU cannot meet the agent's pre-submit
budget (target < ~200 ms per step) after P1 profiling.
