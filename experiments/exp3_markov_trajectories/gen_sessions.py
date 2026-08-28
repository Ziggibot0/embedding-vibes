"""Generate multi-step reasoning (fallacy vs valid) and embed each step.

Pipeline (run in order):
    python gen_sessions.py        # generate + embed -> results/*.npy, *.json
    python build_mc.py            # PCA + k-means centroids + transition matrices
    python predict.py             # Markov LLR classifier vs static probe baseline
    python visualize.py           # entropy / KL / stationary / heatmap figures

72 fallacy sessions (8 per type) + 80 valid sessions, 6 reasoning steps each.
Sequential generation with retry; async-concurrent embedding.
"""
import asyncio
import json
import os
import time
from urllib.request import Request, urlopen

import numpy as np
from tqdm import tqdm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/chat"
CHAT_MODEL = "qwen3.5:4b"
N_STEPS = 6

FALLACY_TYPES = [
    "circular reasoning", "ad hominem", "ad populum", "false causality",
    "false dilemma", "equivocation", "appeal to emotion",
    "faulty generalization", "straw man",
]

BASE_TOPICS = [
    "Universal basic income", "Carbon taxation", "School voucher programs",
    "Mandatory vaccination", "Nuclear energy expansion", "Remote work policies",
    "Social media regulation", "Healthcare privatization",
    "AI in criminal sentencing", "Four-day work week",
]
# 8 distinct topics per fallacy type
FALLACY_TOPICS = [BASE_TOPICS[i % len(BASE_TOPICS)] for i in range(8 * len(FALLACY_TYPES))]

FALLACY_PROMPT = (
    "You are analyzing a policy proposal step by step.\n"
    "However, you MUST incorporate a {fallacy_type} in your reasoning.\n\n"
    "Topic: {topic}\n\n"
    "Respond with exactly {n_steps} numbered reasoning steps. "
    "Each step should be 1-3 sentences.\n"
    "Step 1: Initial observation about the topic.\n"
    "Step 2: Building on step 1.\n"
    "Step 3: This step MUST contain a {fallacy_type}.\n"
    "Step 4: Continuing the argument.\n"
    "Step 5: Drawing a conclusion.\n"
    "Step 6: Final assessment.\n\n"
    'Format each step on its own line starting with "Step N:"'
)

VALID_PROMPT = (
    "You are analyzing a policy proposal statement by statement, step by step.\n\n"
    "Topic: {topic}\n\n"
    "With exactly {n_steps} numbered steps, break down the topic into reasoned steps.\n"
    "Each step should be 1-3 sentences, citing the specific claims you anchor on.\n"
    "Do not make a logical leap without justification.\n"
    "Step 1: State the topic and the specific facts you would check.\n"
    "Step 2: Name 1-2 concrete data points you would verify, with where to find them.\n"
    "Step 3: Point out 1 specific assumption that is being hidden in the prompt.\n"
    "Step 4: Contrast at least two plausible causes, citing evidence for each.\n"
    "Step 5: Note 1 specific counterargument and what evidence it hinges on.\n"
    "Step 6: Give a conditional conclusion with its explicit precondition.\n\n"
    'Format each step on its own line starting with "Step N:"'
)


async def _chat(prompt, temperature=0.7):
    payload = json.dumps({
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    req = Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def parse_steps(text, n_steps):
    if not text:
        return []
    lines = text.split("\n")
    steps = []
    for i in range(1, n_steps + 1):
        prefix = "Step %d:" % i
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(prefix):
                content = stripped[len(prefix):].strip()
                if content:
                    steps.append(content)
                break
    if len(steps) < 3:
        steps = [l.strip() for l in lines if l.strip()][:n_steps]
    return steps


async def _gen_one(fallacy_type, topic):
    fields = {"topic": topic, "n_steps": N_STEPS,
              "fallacy_type": fallacy_type or "a specific logical fallacy"}
    template = FALLACY_PROMPT if fallacy_type else VALID_PROMPT
    prompt = template.format(**fields)
    for _ in range(3):
        try:
            content = (await _chat(prompt))["message"]["content"] or ""
            steps = parse_steps(content, N_STEPS)
        except Exception:
            steps = []
        if len(steps) >= 3:
            return steps
        prompt = template.format(**fields)
    return []


async def _embed_batch(texts, model, concurrency, default_dim):
    payload = json.dumps({
        "model": model,
        "prompt": texts,
        "options": {"num_embed_mult": 0},
    }).encode("utf-8")
    req = Request(OLLAMA_URL.replace("/chat", "/embeddings"),
                  data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    seq = data.get("sequence_embeddings") or data.get("embeddings")
    if isinstance(seq[0], list):  # packed multi-token rows
        return np.mean(np.array(seq, dtype=np.float64), axis=1)
    return np.array(seq, dtype=np.float32)


async def _batch_split(texts, model, concurrency, default_dim, chunk_limit):
    results = []
    n = len(texts)
    for start in range(0, n, chunk_limit):
        chunk = texts[start:start + chunk_limit]
        arr = await _embed_batch(chunk, model, concurrency, default_dim)
        results.append(arr)
    return np.concatenate(results, axis=0)


async def generate_embed():
    t0 = time.time()
    n_fallacy_targets = 8 * len(FALLACY_TYPES)

    print("[1/4] Generating %d fallacy sessions (concurrent, timeout 60s each)..."
          % n_fallacy_targets, flush=True)
    tasks = [
        asyncio.create_task(
            asyncio.wait_for(_gen_one(FALLACY_TYPES[i % len(FALLACY_TYPES)],
                                      FALLACY_TOPICS[i]), timeout=90)
        )
        for i in range(n_fallacy_targets)
    ]
    await asyncio.sleep(0)

    print("[1/4] Generating fallacy sessions via asyncio.gather...", flush=True)
    raw_fallacy = await asyncio.gather(*[
        asyncio.wait_for(_gen_one(FALLACY_TYPES[i % len(FALLACY_TYPES)],
                                  FALLACY_TOPICS[i]), timeout=90)
        for i in range(n_fallacy_targets)
    ])
    fallacy_sessions = [
        {"steps": steps, "label": "fallacy",
         "fallacy_type": FALLACY_TYPES[i % len(FALLACY_TYPES)],
         "topic": FALLACY_TOPICS[i]}
        for i, steps in enumerate(raw_fallacy) if len(steps) >= 3
    ]
    print("   fallacy sessions collected: %d" % len(fallacy_sessions), flush=True)

    print("[2/4] Generating 80 valid sessions...", flush=True)
    raw_valid = await asyncio.gather(*[
        asyncio.wait_for(_gen_one(None, BASE_TOPICS[i % len(BASE_TOPICS)]), timeout=90)
        for i in range(80)
    ])
    valid_sessions = [
        {"steps": steps, "label": "valid",
         "fallacy_type": None,
         "topic": BASE_TOPICS[i % len(BASE_TOPICS)]}
        for i, steps in enumerate(raw_valid) if len(steps) >= 3
    ]
    print("   valid sessions collected: %d" % len(valid_sessions), flush=True)

    all_sessions = fallacy_sessions + valid_sessions
    n_steps_total = sum(len(s["steps"]) for s in all_sessions)
    print("   total: %d sessions, %d steps" % (len(all_sessions), n_steps_total),
          flush=True)

    # Save sessions + labels
    with open(os.path.join(RESULTS_DIR, "sessions.json"),
              "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, indent=2, ensure_ascii=False)
    labels = [{
        "label": s["label"], "fallacy_type": s.get("fallacy_type"),
        "topic": s["topic"], "n_steps": len(s["steps"])}
        for s in all_sessions]
    with open(os.path.join(RESULTS_DIR, "session_labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    # Build flat step list + meta
    all_texts, meta = [], []
    for si, s in enumerate(all_sessions):
        for ti, step in enumerate(s["steps"]):
            all_texts.append(step)
            meta.append({"session_idx": si, "step_idx": ti})
    print("    embedding %d steps" % len(all_texts), flush=True)
    with open(os.path.join(RESULTS_DIR, "trajectory_meta.json"), "w") as f:
        json.dump(meta, f)

    print("[4/4] Embedding with qwen3-embedding...", flush=True)
    arr = await _batch_split(all_texts[:2400], "qwen3-embedding:latest", 16, 4096, 2400)
    out = os.path.join(RESULTS_DIR, "trajectory_embeddings_qwen3_embedding.npy")
    np.save(out, arr)
    print("   saved: %s shape=%s" % (out, str(arr.shape)), flush=True)

    print("   Embedding with nomic-embed-text...", flush=True)
    arr = await _batch_split(all_texts[:2400], "nomic-embed-text:latest", 16, 768, 2400)
    out = os.path.join(RESULTS_DIR, "trajectory_embeddings_nomic_embed_text.npy")
    np.save(out, arr)
    print("   saved: %s shape=%s" % (out, str(arr.shape)), flush=True)

    print("DONE in %.0fs" % (time.time() - t0), flush=True)


import asyncio  # noqa: E402


if __name__ == "__main__":
    asyncio.run(generate_embed())
