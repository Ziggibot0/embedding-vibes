# Fast embedding with concurrent requests
import csv, json, time, os, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "logical-fallacy-repo", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/embeddings"

with open(os.path.join(DATA_DIR, "edu_all.csv"), encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r.get("updated_label") and r.get("updated_label") != "miscellaneous"]

texts = [r["source_article"] for r in rows]
labels = [r["updated_label"] for r in rows]
np.save(os.path.join(RESULTS_DIR, "labels.npy"), np.array(labels))
print(f"Loaded {len(texts)} examples, {len(set(labels))} types", flush=True)

def embed_one(args):
    idx, text, model = args
    try:
        resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": text}, timeout=60)
        emb = resp.json().get("embedding", [])
        return idx, emb
    except Exception as e:
        print(f"  ERROR at {idx}: {e}", flush=True)
        return idx, []

ENCODERS = [
    ("nomic-embed-text", "nomic-embed-text:latest", 16),  # 16 concurrent for small model
    ("qwen3-embedding", "qwen3-embedding:latest", 4),    # 4 concurrent for 8B model
]

for enc_name, model_name, concurrency in ENCODERS:
    cache_path = os.path.join(RESULTS_DIR, f"embeddings_{enc_name.replace('-', '_')}.npy")

    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts) and not np.all(emb == 0):
            print(f"[{enc_name}] Cached: {emb.shape}", flush=True)
            continue

    print(f"\n[{enc_name}] Embedding {len(texts)} with {concurrency} workers...", flush=True)
    embeddings = [None] * len(texts)
    t0 = time.time()
    done = 0

    tasks = [(i, text, model_name) for i, text in enumerate(texts)]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(embed_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx, emb = future.result()
            embeddings[idx] = emb
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(texts) - done) / rate
                print(f"  {done}/{len(texts)} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)
                # Save progress
                valid = [e for e in embeddings if e is not None]
                if len(valid) == done:
                    np.save(cache_path, np.array(valid[:done], dtype=np.float32))

    # Fill any None with zeros
    embeddings = [e if e is not None else [0.0] for e in embeddings]
    emb_array = np.array(embeddings, dtype=np.float32)
    np.save(cache_path, emb_array)
    print(f"  [{enc_name}] Done: {emb_array.shape} in {time.time()-t0:.1f}s", flush=True)

print("\nAll embeddings complete.", flush=True)