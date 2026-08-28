"""Study A generator + embedder — scaled-up controlled study.

Bigger + controlled version of gen_sessions.py. Ground truth comes from the
forced fallacy-injection prompt (known labels, no judge needed), which is the
fast/stable path. This produces the transition-matrix estimator that we gate
with leave-one-out (predict.py has an LOO mode).

Topic separation: fallacy and valid never share a topic family, so the chain
can't overfit to topic. Study B (judge_gen_and_eval.py) is the independent OOD
gate that defeats prompt-style overfitting.

Run order:
    python gen_bigger.py        # generate 120 fallacy + 120 valid, embed
    python build_mc.py
    python predict.py --loo     # leave-one-out Markov classifier (anti-overfit)
"""
import json, os, asyncio, time
from urllib.request import Request, urlopen
from contextlib import asynccontextmanager
import numpy as np
from tqdm import tqdm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OLLAMA_URL = "http://localhost:11434/api"
CHAT_MODEL = "qwen3.5:4b"
N_STEPS = 6

# 10 fallacy types, 8 each -> 80 fallacy + 80 valid = 160 sessions
# Valid topics are seeded from a disjoint topic pool so fallacy/valid never
# share a topic family (prevents topic as the accidental signal).
FALLACY_TYPES = [
    "circular reasoning", "ad hominem", "ad populum", "false causality",
    "false dilemma", "equivocation", "appeal to emotion",
    "faulty generalization", "straw man", "slippery slope",
]
FALLACY_TOPICS = [
    "Universal basic income", "Carbon taxation", "School voucher programs",
    "Mandatory vaccination", "Nuclear energy expansion", "Remote work policies",
    "Social media regulation", "Healthcare privatization",
    "AI in criminal sentencing", "Four-day work week", "Rent control",
    "Cryptocurrency regulation", "Tuition-free college", "Gun control",
    "Immigration reform", "Animal testing bans", "Healthcare privatization",
]
VALID_TOPIC_TOPICS = [
    "Reforming the pension system", "Standardizing the fiscal approach to UBI",
    "Principled governance in the era of remote labor", "Rational policy toward",
    "A balanced perspective on", "Balancing competing interests in",
    "A measured view of", "The trade-offs of", "Evidence-based",
    "A skeptical standpoint on", "Reasoned analysis of",
]
FALLACY_PROMPT = (
    "You are analyzing a policy topic one step at a time.\n"
    "You MUST incorporate a {fallacy_type} somewhere in your reasoning.\n\n"
    "Topic: {topic}\n\n"
    "Respond with exactly {n_steps} numbered steps. 1-3 sentences each.\n"
    "'Step 1': setup. 'Step 2': build on step 1. 'Step 3': {fallacy_type} here.\n"
    "'Step 4': continue. 'Step 5': a conclusion. 'Step 6': final verdict.\n\n"
    'Format each step on its own line starting with "Step N:"'
)

VALID_PROMPT = (
    "You are analyzing a policy topic one step at a time, rigorously.\n\n"
    "Topic: {topic}\n\n"
    "With exactly {n_steps} numbered steps, break the topic in reasoned steps,\n"
    "1-3 sentences each. Justify every claim, name 1 hidden assumption, flag weak evidence.\n"
    "'Step 1': what's claimed. 'Step 2': 1-2 facts to check. 'Step 3': an assumption to question.\n"
    "'Step 4': two causes, each with evidence. 'Step 5': one counterargument.\n"
    "'Step 6': a conditional conclusion stating its precondition.\n\n"
    'Format each step on its own line starting with "Step N:"'
)


async def _chat(prompt, temperature=0.6, timeout=120):
    payload = json.dumps({"model": CHAT_MODEL, "stream": False,
                          "options": {"temperature": temperature},
                          "messages": [{"role": "user", "content": prompt}]}).encode()
    req = Request(f"{OLLAMA_URL}/chat", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def parse_steps(text, n_steps):
    if not text:
        return []
    steps, lines = [], text.split("\n")
    for i in range(1, n_steps + 1):
        p = f"Step {i}:"
        for line in lines:
            s = line.strip()
            if s.startswith(p):
                steps.append(s[len(p):].strip())
                break
    if len(steps) < 3:
        steps = [line.strip() for line in lines if line.strip()][:n_steps]
    return steps


@asynccontextmanager
async def sem_limit(n):
    sem = asyncio.Semaphore(n)
    async with sem:
        yield sem


async def _gen_one(fallacy_type, topic):
    tmpl = FALLACY_PROMPT if fallacy_type else VALID_PROMPT
    fields = {"topic": topic, "n_steps": N_STEPS,
              "fallacy_type": fallacy_type or "a precise logical error"}
    for _ in range(4):
        async with sem_limit(20):
            try:
                content = (await _chat(tmpl.format(**fields)))["message"]["content"] or ""
                steps = parse_steps(content, N_STEPS)
                if len(steps) >= 3:
                    return steps, content
            except Exception:
                await asyncio.sleep(3)
    return [], ""


def dump(sessions, all_texts):
    """Write sessions + labels + meta, then embed both encoders."""
    out = os.path.join(RESULTS_DIR, "studyA_sessions.json")
    json.dump(sessions, open(out, "w"), indent=2, ensure_ascii=False)
    labels, meta_flat = [], []
    for i, s in enumerate(sessions):
        labels.append({"label": s["label"], "fallacy_type": s.get("fallacy_type"),
                       "topic": s["topic"], "n_steps": len(s["steps"])})
        for _ in range(len(s["steps"])):
            meta_flat.append({"si": i})
    json.dump(labels, open(os.path.join(RESULTS_DIR, "studyA_session_labels.json"), "w"))
    # meta: {session_idx, step_idx} flat, ordered like the flatten of all steps
    meta = []
    for j, s in enumerate(sessions):
        for k in range(len(s["steps"])):
            meta.append({"session_idx": j, "step_idx": k,
                         "raw": s["raw"] if _DUMP_RAW else None})
    json.dump(meta, open(os.path.join(RESULTS_DIR, "studyA_meta.json"), "w"))

    for enc_label, enc, dim in [("nomic_embed_text", "nomic-embed-text:latest", 768),
                                ("qwen3_embedding", "qwen3-embedding:latest", 4096)]:
        print(f"[{enc_label}] embedding {len(all_texts)} steps...", flush=True)
        arrays = []
        for i in range(0, len(all_texts), 256):
            arr = embed_inline(all_texts[i:i+256], enc)
            arrays.append(arr)
        full = np.concatenate(arrays, axis=0)
        np.save(os.path.join(RESULTS_DIR, f"studyA_embeddings_{enc_label}.npy"), full)
        json.dump(meta, open(os.path.join(RESULTS_DIR, f"studyA_meta_{enc_label}.json"), "w"))
        print(f"  -> {full.shape}", flush=True)
    return out


def embed_inline(text, model):
    req = Request(f"{OLLAMA_URL}/embed",
                  data=json.dumps({"model": model, "input": text}).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=240) as resp:
        d = json.loads(resp.read())
    return np.asarray(d["embeddings"], dtype=np.float32)


async def build_sessions(n_per=8, concurrency=20):
    sessions, all_texts = [], []
    print("[1] generating fallacy sessions...", flush=True)
    fallacy_tasks = []
    ft_pool = FALLACY_TYPES * n_per
    topics_pool = FALLACY_TOPICS * (n_per//len(FALLACY_TYPES) + 1)
    for i, ft in enumerate(ft_pool[: len(FALLACY_TYPES) * n_per]):
        fallacy_tasks.append([ft, topics_pool[i % len(topics_pool)]])

    tasks = [asyncio.create_task(_gen_one(x[0], x[1])) for x in fallacy_tasks]
    for completed in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        steps, full = completed.result()
        si = len(sessions)
        ft, topic = fallacy_tasks[tasks.index(completed)]
        if len(steps) >= 3:
            sessions.append({"raw": full, "steps": steps, "label": "fallacy",
                             "fallacy_type": ft, "topic": topic})
            all_texts.extend(steps)

    print("[2] generating valid sessions...", flush=True)
    valid_tasks = []
    for i in range(n_per):
        valid_tasks.append([None, VALID_TOPIC_TOPICS[i % len(VALID_TOPIC_TOPICS)]]*2)
    vtasks = [asyncio.create_task(_gen_one(x[0], x[1])) for x in valid_tasks]
    for completed in tqdm(asyncio.as_completed(vtasks), total=len(vtasks)):
        steps, full = completed.result()
        si = len(sessions)
        ft, topic = valid_tasks[vtasks.index(completed)]
        if len(steps) >= 3:
            sessions.append({"raw": full, "steps": steps, "label": "valid",
                             "fallacy_type": None, "topic": topic})
            all_texts.extend(steps)

    return sessions, all_texts


async def main():
    t0 = time.time()
    sessions, all_texts = await build_sessions()
    out = dump(sessions, all_texts)
    print(f"\nDONE in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
