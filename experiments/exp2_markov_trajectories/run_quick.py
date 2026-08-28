"""
Quick test runner: generate a small batch of sessions and run the full pipeline.
Uses sequential generation to avoid overloading Ollama.
"""
import os, sys, json, time, random, csv, numpy as np
import requests

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api"
CHAT_MODEL = "qwen3.5:4b"

N_PER_TYPE = 5  # small for quick test
N_STEPS = 6

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

TOPICS = [
    "Universal basic income", "Carbon taxation", "School voucher programs",
    "Mandatory vaccination", "Nuclear energy expansion", "Remote work policies",
    "Social media regulation", "Healthcare privatization", "AI in criminal sentencing",
    "Four-day work week", "Rent control in cities", "Genetic engineering in agriculture",
    "Space colonization funding", "Cryptocurrency regulation", "Free college tuition",
    "Gun control legislation", "Immigration reform", "Animal testing bans",
    "Universal healthcare", "Autonomous vehicle regulation",
]

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


def call_chat(prompt, model=CHAT_MODEL):
    resp = requests.post(f"{OLLAMA_URL}/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.7},
    }, timeout=180)
    return resp.json()["message"]["content"]


def call_embed(text, model):
    resp = requests.post(f"{OLLAMA_URL}/embeddings", json={
        "model": model, "prompt": text,
    }, timeout=60)
    return resp.json().get("embedding", [])


def parse_steps(text, n_steps=6):
    steps = []
    for i in range(1, n_steps + 1):
        prefix = f"Step {i}:"
        for line in text.split("\n"):
            if line.strip().startswith(prefix):
                content = line.strip()[len(prefix):].strip()
                if content:
                    steps.append(content)
                break
    if len(steps) < 3:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        steps = lines[:n_steps]
    return steps


def main():
    sessions = []
    all_step_texts = []
    step_meta = []  # (session_idx, step_idx)

    # Generate fallacy sessions
    print("=== Generating fallacy sessions ===")
    for ft in FALLACY_TYPES:
        for i in range(N_PER_TYPE):
            topic = TOPICS[(i * 3 + hash(ft)) % len(TOPICS)]
            prompt = FALLACY_PROMPT.format(fallacy_type=ft, topic=topic, n_steps=N_STEPS)
            try:
                text = call_chat(prompt)
                steps = parse_steps(text, N_STEPS)
            except Exception as e:
                print(f"  Error: {e}")
                steps = []

            if len(steps) < 3:
                print(f"  [{ft}] session {i}: only {len(steps)} steps, retrying...")
                try:
                    text = call_chat(prompt)
                    steps = parse_steps(text, N_STEPS)
                except:
                    steps = []

            si = len(sessions)
            sessions.append({
                "steps": steps, "label": "fallacy",
                "fallacy_type": ft, "topic": topic,
            })
            for ti, step in enumerate(steps):
                all_step_texts.append(step)
                step_meta.append((si, ti))

            if (i + 1) % 5 == 0:
                print(f"  [{ft}] {i+1}/{N_PER_TYPE}")
        print(f"  [{ft}] done")

    # Generate valid sessions
    n_valid = N_PER_TYPE * len(FALLACY_TYPES)
    print(f"\n=== Generating {n_valid} valid sessions ===")
    for i in range(n_valid):
        topic = TOPICS[i % len(TOPICS)]
        prompt = VALID_PROMPT.format(topic=topic, n_steps=N_STEPS)
        try:
            text = call_chat(prompt)
            steps = parse_steps(text, N_STEPS)
        except Exception as e:
            print(f"  Error: {e}")
            steps = []

        si = len(sessions)
        sessions.append({
            "steps": steps, "label": "valid",
            "fallacy_type": None, "topic": topic,
        })
        for ti, step in enumerate(steps):
            all_step_texts.append(step)
            step_meta.append((si, ti))

        if (i + 1) % 10 == 0:
            print(f"  [valid] {i+1}/{n_valid}")

    # Save sessions
    with open(os.path.join(RESULTS_DIR, "sessions.json"), "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)

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

    print(f"\nTotal sessions: {len(sessions)}")
    print(f"Total step texts: {len(all_step_texts)}")

    # Embed all steps
    for enc_name, model_name, concurrency in [
        ("nomic_embed_text", "nomic-embed-text:latest", 16),
        ("qwen3_embedding", "qwen3-embedding:latest", 4),
    ]:
        cache_path = os.path.join(RESULTS_DIR, f"trajectory_embeddings_{enc_name}.npy")
        meta_path = os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")

        print(f"\n[{enc_name}] Embedding {len(all_step_texts)} steps...")
        t0 = time.time()
        embeddings = []

        for idx, text in enumerate(all_step_texts):
            emb = call_embed(text, model_name)
            embeddings.append(emb if emb else [0.0])
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(all_step_texts) - idx - 1) / rate
                print(f"  {idx+1}/{len(all_step_texts)} ({rate:.1f}/s, ETA {eta:.0f}s)")

        # Detect dimension
        dim = 768
        for e in embeddings:
            if e and len(e) > 0:
                dim = len(e)
                break
        embeddings = [e if e and len(e) > 0 else [0.0] * dim for e in embeddings]
        emb_array = np.array(embeddings, dtype=np.float32)
        np.save(cache_path, emb_array)

        meta_data = [{"session_idx": si, "step_idx": ti} for si, ti in step_meta]
        with open(meta_path, "w") as f:
            json.dump(meta_data, f)

        print(f"  [{enc_name}] Done: {emb_array.shape} in {time.time()-t0:.1f}s")

    print("\nGeneration complete. Run build_mc.py next.")


if __name__ == "__main__":
    main()