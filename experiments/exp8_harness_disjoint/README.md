# Exp8 — Harness-Disjoint Transfer

Read `DESIGN.md` first — it contains the pre-registered gates (A-D) and the
full "why" chain. This README is operational only.

## What this fixes
Exp7's extract (sessions.jsonl) dropped harness/benchmark/model metadata, so
the project's central question — does the session signal survive format shift
across harnesses — was untestable. Exp8 re-extracts with metadata and measures
the transfer floor with the same frozen probes as exp7 (no training).

## Data (extracted 2026-08-28, cap 3000 as pre-declared)
- 3,000 sessions from Exgentic/agent-llm-traces-v2 (schema v1.2)
- harness: claude_code 524, openai_solo 994, smolagents_code 774,
  tool_calling 604, tool_calling_with_shortlisting 104 (skipped in LOHO, <150)
- benchmark: appworld 893, browsecompplus 1485, swebench 622
- models: 5 × ~500-665 (DeepSeek-V3.2, Kimi-K2.5, claude-opus-4-5,
  gemini-3-pro-preview, gpt-5.2)
- success rate 1579/3000 (52.6%)

## Run
`python exp8_harness_disjoint.py` — phases: stream→extract (65s),
embed (seeded from exp7 cache: 30,841 reused, ~18k new, resumable),
evaluate (7 features × in_format CV / LOHO / LOBO / LOMO), gates, summary.

## Quick-read outputs
- `results/exp8_summary.md` — main table + drops + gate verdicts (read this)
- `results/exp8_results.json` — every metric incl. per-group, ECE, Brier
- `results/sessions_meta.jsonl` — the canonical-with-metadata P0 artifact
- `exp8.log` — full phase progress

## Status
RUNNING (embedding phase). Results will be written into RESULTS.md with the
gate verdicts, honest-scope language, and claim-status tags per NOMENCLATURE.