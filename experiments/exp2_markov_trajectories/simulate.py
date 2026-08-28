"""
Generate multi-step agentic reasoning sessions (fallacy + valid) via Ollama,
then embed each reasoning step with both encoders.

Each session is a sequence of reasoning steps produced by the LLM,
where we control whether the reasoning contains a logical fallacy.
"""
import os, json, time, csv, random
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api"
GENERATE_MODEL = "qwen3-embedding"  # We'll use a chat model for generation; this is just embed
# Actually we need a chat model for generation. Let's use what's available.
# The user has qwen2.5-72b, qwen3-coder, lfm2, qwen3.5:4b, gemma3:4b
# Use qwen3.5:4b for fast generation (small model, but reasoning-capable)

CHAT_MODEL = "qwen3.5:4b"
EMBED_MODELS = [
    ("nomic_embed_text", "nomic-embed-text:latest", 16),
    ("qwen3_embedding", "qwen3-embedding:latest", 4),
]

N_SESSIONS_PER_CLASS = 50  # 50 fallacy, 50 valid = 100 total per fallacy type
N_STEPS = 6  # reasoning steps per session
FALLACY_TYPES = [
    "circular reasoning",
    "ad hominem",
    "ad populum",
    "false causality",
    "false dilemma",
    "equivocation",
    "appeal to emotion",
    "faulty generalization",
    "straw man",
]

# --- Prompt templates ---

FALLACY_PROMPT = """You are analyzing a policy proposal step by step.
However, you MUST incorporate a {fallacy_type} in your reasoning by step 3.

Topic: {topic}

Respond with exactly {n_steps} numbered reasoning steps. Each step should be 1-3 sentences.
Step 1: Initial observation about the topic.
Step 2: Building on step 1.
Step 3: This step MUST contain a {fallacy_type}.
Step 4: Continuing the argument.
Step 5: Drawing a conclusion.
Step 6: Final assessment.

Format each step on its own line starting with "Step N:" """

VALID_PROMPT = """You are analyzing a policy proposal step by step.
Use rigorous, logically sound reasoning throughout.

Topic: {topic}

Respond with exactly {n_steps} numbered reasoning steps. Each step should be 1-3 sentences.
Step 1: Initial observation about the topic.
Step 2: Building on step 1 with evidence.
Step 3: Applying a logical principle.
Step 4: Considering counterarguments.
Step 5: Drawing a conclusion.
Step 6: Final assessment.

Format each step on its own line starting with "Step N:" """

TOPICS = [
    "Universal basic income",
    "Carbon taxation",
    "School voucher programs",
    "Mandatory vaccination",
    "Nuclear energy expansion",
    "Remote work policies",
    "Social media regulation",
    "Healthcare privatization",
    "AI in criminal sentencing",
    "Four-day work week",
    "Rent control in cities",
    "Genetic engineering in agriculture",
    "Space colonization funding",
    "Cryptocurrency regulation",
    "Free college tuition",
    "Gun control legislation",
    "Immigration reform",
    "Animal testing bans",
    "Universal healthcare",
    "Autonomous vehicle regulation",
]

def call_ollama_chat(prompt, model=CHAT_MODEL, timeout=120):
    """Call Ollama chat API for generation."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 512},
            },
            timeout=timeout,
        )
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"  Chat error: {e}")
        return ""


def call_ollama_embed(text, model, timeout=60):
    """Call Ollama embed API."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/embeddings",
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        return resp.json().get("embedding", [])
    except Exception as e:
        print(f"  Embed error: {e}")
        return []


def parse_steps(text, n_steps=6):
    """Parse numbered steps from LLM output."""
    steps = []
    for i in range(1, n_steps + 1):
        prefix = f"Step {i}:"
        for line in text.split("\n"):
            if line.strip().startswith(prefix):
                content = line.strip()[len(prefix):].strip()
                if content:
                    steps.append(content)
                break
    # Fallback: if parsing fails, split by newlines
    if len(steps) < n_steps:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        steps = lines[:n_steps]
    return steps


def generate_session(fallacy_type=None, topic=None):
    """Generate one reasoning session (list of step texts)."""
    if topic is None:
        topic = random.choice(TOPICS)

    if fallacy_type:
        prompt = FALLACY_PROMPT.format(
            fallacy_type=fallacy_type,
            topic=topic,
            n_steps=N_STEPS,
        )
    else:
        prompt = VALID_PROMPT.format(topic=topic, n_steps=N_STEPS)

    text = call_ollama_chat(prompt)
    steps = parse_steps(text, N_STEPS)

    if len(steps) < 3:
        # Retry once
        text = call_ollama_chat(prompt)
        steps = parse_steps(text, N_STEPS)

    return steps, topic


def main():
    sessions = []  # list of {steps, label, fallacy_type, topic}

    print("=== Generating fallacy sessions ===")
    for ft in FALLACY_TYPES:
        for i in range(N_SESSIONS_PER_CLASS):
            topic = TOPICS[(i * 3 + hash(ft)) % len(TOPICS)]
            steps, topic = generate_session(fallacy_type=ft, topic=topic)
            sessions.append({
                "steps": steps,
                "label": "fallacy",
                "fallacy_type": ft,
                "topic": topic,
            })
            if (i + 1) % 10 == 0:
                print(f"  [{ft}] {i+1}/{N_SESSIONS_PER_CLASS}")

    print(f"\n=== Generating valid sessions ===")
    n_valid = N_SESSIONS_PER_CLASS * len(FALLACY_TYPES)  # match fallacy count
    for i in range(n_valid):
        topic = TOPICS[i % len(TOPICS)]
        steps, topic = generate_session(fallacy_type=None, topic=topic)
        sessions.append({
            "steps": steps,
            "label": "valid",
            "fallacy_type": None,
            "topic": topic,
        })
        if (i + 1) % 50 == 0:
            print(f"  [valid] {i+1}/{n_valid}")

    # Save raw sessions
    with open(os.path.join(RESULTS_DIR, "sessions.json"), "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)
    print(f"\nGenerated {len(sessions)} sessions, saved to results/sessions.json")

    # Filter out sessions with too few steps
    sessions = [s for s in sessions if len(s["steps"]) >= 3]
    print(f"Sessions with >= 3 steps: {len(sessions)}")

    # Embed all steps with both encoders
    for enc_name, model_name, concurrency in EMBED_MODELS:
        cache_path = os.path.join(RESULTS_DIR, f"trajectory_embeddings_{enc_name}.npy")
        meta_path = os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")

        # Collect all step texts
        all_texts = []
        meta = []  # which session, which step
        for si, session in enumerate(sessions):
            for ti, step in enumerate(session["steps"]):
                all_texts.append(step)
                meta.append({"session_idx": si, "step_idx": ti})

        if os.path.exists(cache_path):
            existing = np.load(cache_path)
            if existing.shape[0] == len(all_texts):
                print(f"[{enc_name}] Cached: {existing.shape}")
                continue

        print(f"\n[{enc_name}] Embedding {len(all_texts)} steps with {concurrency} workers...")
        embeddings = [None] * len(all_texts)
        t0 = time.time()
        done = 0

        tasks = [(i, text, model_name) for i, text in enumerate(all_texts)]

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(lambda t: (t[0], call_ollama_embed(t[1], t[2])), t): t[0] for t in tasks}
            for future in as_completed(futures):
                idx, emb = future.result()
                embeddings[idx] = emb
                done += 1
                if done % 200 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (len(all_texts) - done) / rate
                    print(f"  {done}/{len(all_texts)} ({rate:.1f}/s, ETA {eta:.0f}s)")

        # Fill None with zeros
        embeddings = [e if e else [0.0] * 768 for e in embeddings]
        emb_array = np.array(embeddings, dtype=np.float32)
        np.save(cache_path, emb_array)
        print(f"  [{enc_name}] Done: {emb_array.shape}")

        with open(meta_path, "w") as f:
            json.dump(meta, f)

    # Save session metadata (labels)
    session_labels = []
    for s in sessions:
        session_labels.append({
            "label": s["label"],
            "fallacy_type": s.get("fallacy_type"),
            "topic": s["topic"],
            "n_steps": len(s["steps"]),
        })
    with open(os.path.join(RESULTS_DIR, "session_labels.json"), "w") as f:
        json.dump(session_labels, f, indent=2)

    print("\nAll done. Run build_mc.py next.")


if __name__ == "__main__":
    main()