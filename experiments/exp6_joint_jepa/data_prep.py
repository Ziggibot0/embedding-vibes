"""
Exp6 data prep — extract per-step TEXT from multi-harness agentic trajectories.

The from-scratch encoder takes TEXT, not embeddings. This script streams the
multi-harness trajectory datasets and extracts, for each session, the ordered
list of step texts (assistant reasoning + tool calls) plus the success label.

Output: a tokenized, packed corpus for training the from-scratch encoder
(MLM + Barlow Twins + JEPA objectives). Saved incrementally so we never hold
the whole dataset in memory.

Sources (multi-harness -> format invariance):
  - Exgentic/agent-llm-traces-v2   (10K runs, 6 benchmarks, 5 models, OTel spans)
  - open-thoughts/AgentTrove       (1.7M trajectories, reward labels)
"""
import os, json, argparse, time
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_exgentic_session(row):
    """Pull ordered step texts from an Exgentic row's spans.

    Each span's gen_ai.input.messages / gen_ai.output.messages contain the
    actual text. We extract assistant text parts (reasoning) and tool calls.
    Returns (steps: list[str], success: bool).
    """
    steps = []
    for sp in row.get("spans", []):
        attrs = sp.get("attributes") or {}
        # assistant output (reasoning + tool calls)
        out = attrs.get("gen_ai.output.messages")
        if out:
            try:
                msgs = json.loads(out) if isinstance(out, str) else out
            except Exception:
                msgs = []
            for m in msgs:
                for part in m.get("parts", []):
                    if part.get("type") == "text" and part.get("content"):
                        steps.append(part["content"].strip())
                    elif part.get("type") == "tool_call":
                        steps.append(f"[tool_call] {part.get('name','')}: {part.get('arguments','')}")
        # user input (task context / tool results) — keep as context
        inp = attrs.get("gen_ai.input.messages")
        if inp and not steps:
            try:
                msgs = json.loads(inp) if isinstance(inp, str) else inp
            except Exception:
                msgs = []
            for m in msgs:
                for part in m.get("parts", []):
                    if part.get("type") == "text" and part.get("content"):
                        steps.append(part["content"].strip())
    # dedupe consecutive identical (span repeats)
    deduped = []
    for s in steps:
        if not deduped or deduped[-1] != s:
            deduped.append(s)
    return deduped, bool(row.get("success", False))


def extract_agenttrove_session(row):
    """Pull ordered step texts from an AgentTrove row.

    AgentTrove uses 'conversations' (list of {role, content}). We extract
    assistant + tool messages in order. Outcome is often unlabeled (None);
    for the from-scratch encoder (MLM/Barlow/JEPA are self-supervised) the
    label is secondary — scale is what matters.
    Returns (steps, success).
    """
    steps = []
    msgs = row.get("conversations") or row.get("messages") or []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) for c in content if isinstance(c, dict))
        content = str(content).strip()
        if not content:
            continue
        if role in ("assistant", "tool", "function"):
            steps.append(f"[{role}] {content}")
        elif role == "user":
            steps.append(f"[user] {content}")
    # outcome: try reward/result/judgment, else unknown
    success = bool(row.get("reward", 0) > 0 or row.get("success", False))
    return steps, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["exgentic", "agenttrove", "both"], default="both")
    ap.add_argument("--max-sessions", type=int, default=0, help="0 = all")
    ap.add_argument("--min-steps", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()

    from datasets import load_dataset

    all_sessions = []
    counts = {"exgentic": 0, "agenttrove": 0}

    if args.source in ("exgentic", "both"):
        print("Streaming Exgentic/agent-llm-traces-v2 ...")
        ds = load_dataset("Exgentic/agent-llm-traces-v2", split="train", streaming=True)
        for row in ds:
            steps, success = extract_exgentic_session(row)
            if args.min_steps <= len(steps) <= args.max_steps:
                all_sessions.append({"steps": steps, "success": success, "source": "exgentic"})
                counts["exgentic"] += 1
            if args.max_sessions and len(all_sessions) >= args.max_sessions:
                break

    if args.source in ("agenttrove", "both"):
        print("Streaming open-thoughts/AgentTrove ...")
        ds = load_dataset("open-thoughts/AgentTrove", split="train", streaming=True)
        for row in ds:
            steps, success = extract_agenttrove_session(row)
            if args.min_steps <= len(steps) <= args.max_steps:
                all_sessions.append({"steps": steps, "success": success, "source": "agenttrove"})
                counts["agenttrove"] += 1
            if args.max_sessions and len(all_sessions) >= args.max_sessions:
                break

    print(f"Collected {len(all_sessions)} sessions: {counts}")
    n_success = sum(1 for s in all_sessions if s["success"])
    print(f"  success={n_success} failure={len(all_sessions)-n_success}")
    n_steps = sum(len(s["steps"]) for s in all_sessions)
    print(f"  total steps={n_steps}  avg steps/session={n_steps/max(len(all_sessions),1):.1f}")

    # Save raw sessions (text) — APPEND to a combined corpus
    out = os.path.join(OUT_DIR, "sessions.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        for s in all_sessions:
            f.write(json.dumps(s) + "\n")
    print(f"Appended {len(all_sessions)} sessions to {out}")


if __name__ == "__main__":
    main()
