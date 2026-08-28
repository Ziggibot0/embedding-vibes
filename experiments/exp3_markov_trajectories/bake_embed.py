"""Batch-embed already-generated sessions.

Reads results/sessions.json, extracts every step text, embeds it in bulk via
Ollama's /embeddings (multi-token batch) for both encoders, and writes the two
trajectory_embeddings_*.npy caches plus the trajectory_meta.json index the
Markov pipeline expects.

Faster than run_quick.py, which embeds one-at-a-time.
"""
import json, os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")

SESSIONS = json.load(open(os.path.join(RESULTS, "sessions.json")))

# flat list of step texts, index -> (session_idx, step_idx)
texts, meta = [], []
for si, s in enumerate(SESSIONS):
    for ti, step in enumerate(s["steps"]):
        texts.append(step)
        meta.append({"session_idx": si, "step_idx": ti,
                     "label": s["label"],
                     "fallacy_type": s.get("fallacy_type")})
print(f"steps to embed: {len(texts)}")

OLLAMA = "http://localhost:11434/api"


def embed_batch(texts, model):
    end = OLLAMA.replace("/api", "/api/embed")  # array endpoint for 0.32
    payload = json.dumps({"model": model, "input": texts}).encode()
    import urllib.request
    req = urllib.request.Request(end, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    arr = np.array(data.get("embeddings"), dtype=np.float32)
    return arr


for enc, model in [
    ("nomic_embed_text", "nomic-embed-text:latest"),
    ("qwen3_embedding", "qwen3-embedding:latest"),
]:
    print(f"[{enc}] emb {len(texts)} steps via {model}...")
    chunk = 200
    arrays = [embed_batch(texts[i:i+chunk], model)
              for i in range(0, len(texts), chunk)]
    arr = np.concatenate(arrays, axis=0)
    out = os.path.join(RESULTS, f"trajectory_embeddings_{enc}.npy")
    np.save(out, arr)
    # compact meta matching the build pipeline's format
    meta_out = [{"session_idx": m["session_idx"], "step_idx": m["step_idx"],
                 "label": m["label"], "fallacy_type": m["fallacy_type"]}
                for m in meta]
    # build_mc.py reads the per-encoder suffixed name
    json.dump(meta_out,
              open(os.path.join(RESULTS, f"trajectory_meta_{enc}.json"), "w"))
    print(f"  -> {out}: {arr.shape}")

print("DONE")
