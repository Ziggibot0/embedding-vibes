"""Exp10.1 — re-embed exp8's 3000 sessions with nomic-embed-text:v1.5.

Re-embed each step with v1.5 (matryoshka), layer-norm, truncate to 64 dims
(exp10's fix), mean-pool steps to a session vector (exp7b: centroid is the
signal carrier). Cache per-session 64-dim vectors + labels + harness/benchmark.

Ollama ignores `dimensionality`; truncation is client-side (layer-norm, keep
first 64 dims). Task prefix "classification: " applied consistently.
"""
import os, json, time, argparse
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
EXP8 = os.path.join(HERE, "..", "exp8_harness_disjoint", "results")
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

MODEL = "nomic-embed-text:v1.5"
OLLAMA = "http://localhost:11434/api/embed"
PREFIX = "classification: "
DIM = 64
BATCH = 32
STEP_CAP = 20000  # v1.5 context guard (matches exp7/exp8/exp9)


def embed_batch(texts):
    resp = requests.post(OLLAMA, json={"model": MODEL, "input": texts}, timeout=180)
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=0, help="0 = all 3000")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(EXP8, "sessions_meta.jsonl"), encoding="utf-8")]
    if args.max_sessions:
        rows = rows[: args.max_sessions]
    print(f"Re-embedding {len(rows)} sessions with {MODEL} -> {DIM} dims")

    # Flatten all steps
    all_texts, step_bounds = [], []
    for r in rows:
        start = len(all_texts)
        for st in r["steps"]:
            all_texts.append(PREFIX + st[:STEP_CAP])
        step_bounds.append((start, len(all_texts)))

    print(f"  total steps: {len(all_texts)}")
    embs = []
    t0 = time.time()
    for i in range(0, len(all_texts), BATCH):
        e = embed_batch(all_texts[i : i + BATCH])
        embs.append(e)
        done = min(i + BATCH, len(all_texts))
        if done % 2000 == 0 or done == len(all_texts):
            print(f"  embedded {done}/{len(all_texts)}  ({time.time()-t0:.0f}s)")
    E = np.concatenate(embs, axis=0)  # (N_steps, 768)

    # Matryoshka layer-norm + truncate to DIM
    E = E - E.mean(1, keepdims=True)
    E = E / (E.std(1, keepdims=True) + 1e-6)
    E = E * (DIM ** 0.5)
    E = E[:, :DIM]

    # Mean-pool steps -> session vector
    sessions = []
    for i, (start, end) in enumerate(step_bounds):
        if end == start:
            continue
        vec = E[start:end].mean(0)
        sessions.append({
            "session_idx": i,
            "vec": vec,
            "success": 1 if rows[i]["success"] else 0,
            "harness": rows[i].get("harness", "unknown"),
            "benchmark": rows[i].get("benchmark", "unknown"),
        })

    V = np.array([s["vec"] for s in sessions])
    np.save(os.path.join(OUT, "sessions_v15_64d.npy"), V)
    with open(os.path.join(OUT, "sessions_v15_64d.json"), "w", encoding="utf-8") as f:
        json.dump([{k: s[k] for k in ("session_idx", "success", "harness", "benchmark")}
                   for s in sessions], f, indent=2)
    print(f"  cached {len(sessions)} session vectors ({V.shape}) to {OUT}/sessions_v15_64d.npy")


if __name__ == "__main__":
    main()
