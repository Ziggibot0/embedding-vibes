"""Build the large training corpus for exp9 attempt #3.

Streams from multiple HF datasets, extracts step texts in the canonical format
(same logic as exp6 data_prep / exp8 extract), and writes one big JSONL.
No labels needed — MLM + BT are self-supervised. The 3,000 labeled Exgentic
sessions from exp8 stay as the eval set; this is training fuel only.

Sources:
  - Exgentic/agent-llm-traces-v2  (OTel spans, has harness metadata)
  - open-thoughts/AgentTrove       (conversations format, 1.7M trajectories)

Target: 50k+ sessions (enough for a 6M model to learn language competence).
"""
import os, sys, json, time, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("corpus")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "exp9_aspace_encoder", "data", "train_corpus.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

MAX_SESSIONS = 60000  # cap; AgentTrove is huge, we don't need all 1.7M
MIN_STEPS, MAX_STEPS = 3, 40
STEP_CHAR_CAP = 20000  # nomic ctx guard (also used by exp7/exp8)


def extract_exgentic(row):
    """Same logic as exp8 — OTel spans -> step texts."""
    steps = []
    for sp in row.get("spans", []):
        attrs = sp.get("attributes") or {}
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
    # dedupe consecutive
    deduped = []
    for s in steps:
        if not deduped or deduped[-1] != s:
            deduped.append(s)
    return deduped


def extract_agenttrove(row):
    """Same logic as exp6 data_prep — conversations -> step texts."""
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
    # dedupe consecutive
    deduped = []
    for s in steps:
        if not deduped or deduped[-1] != s:
            deduped.append(s)
    return deduped


def main():
    from datasets import load_dataset
    n_total = 0
    t0 = time.time()

    with open(OUT, "w", encoding="utf-8") as f:
        # ---- Exgentic: take ALL of it (not just 3k cap from exp8)
        log.info("streaming Exgentic/agent-llm-traces-v2 ...")
        try:
            ds = load_dataset("Exgentic/agent-llm-traces-v2", split="train", streaming=True)
            n_exg = 0
            for row in ds:
                steps = extract_exgentic(row)
                if MIN_STEPS <= len(steps) <= MAX_STEPS:
                    rec = {"steps": [s[:STEP_CHAR_CAP] for s in steps],
                           "source": "exgentic",
                           "harness": row.get("harness", "")}
                    f.write(json.dumps(rec) + "\n")
                    n_exg += 1; n_total += 1
                    if n_exg % 1000 == 0:
                        log.info(f"  exgentic: {n_exg} ({time.time()-t0:.0f}s)")
                if n_total >= MAX_SESSIONS:
                    break
            log.info(f"exgentic done: {n_exg} sessions")
        except Exception as e:
            log.warning(f"exgentic failed: {e}")

        if n_total >= MAX_SESSIONS:
            log.info(f"cap {MAX_SESSIONS} reached")
            return

        # ---- AgentTrove: the big one (1.7M available, take what we need)
        log.info("streaming open-thoughts/AgentTrove ...")
        try:
            ds = load_dataset("open-thoughts/AgentTrove", split="train", streaming=True)
            n_at = 0
            for row in ds:
                steps = extract_agenttrove(row)
                if MIN_STEPS <= len(steps) <= MAX_STEPS:
                    rec = {"steps": [s[:STEP_CHAR_CAP] for s in steps],
                           "source": "agenttrove",
                           "harness": "agenttrove"}
                    f.write(json.dumps(rec) + "\n")
                    n_at += 1; n_total += 1
                    if n_at % 2000 == 0:
                        log.info(f"  agenttrove: {n_at} (total {n_total}, {time.time()-t0:.0f}s)")
                if n_total >= MAX_SESSIONS:
                    break
            log.info(f"agenttrove done: {n_at} sessions")
        except Exception as e:
            log.warning(f"agenttrove failed: {e}")

    log.info(f"CORPUS COMPLETE: {n_total} sessions in {time.time()-t0:.0f}s -> {OUT}")
    # quick stats
    from collections import Counter
    src = Counter()
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src[r["source"]] += 1
    log.info(f"sources: {dict(src)}")


if __name__ == "__main__":
    main()