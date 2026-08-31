# Exp16 — Own-CoT generation (distribution-shift control)

**Status:** NOT STARTED · placeholder (EXPERIMENTS.md)

Fill in before running. The predictor learns failure patterns of LARGE models
(AgentTrove is GPT/Claude-class), but the user runs qwen3.5moe (35B-A3B).
Generate the user's own CoT trajectories, label by outcome. Addresses the
transfer risk that reasoning patterns won't cross model families. Requires
P0b's logging shim; gates any production trepidation claim.
