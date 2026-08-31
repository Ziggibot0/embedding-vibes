"""Exp10 — re-embed exp3 sessions with nomic-embed-text:v1.5 (matryoshka).

The exp3/exp6 caches are nomic v1 (dense, 768-dim, NOT matryoshka). This script
re-embeds the 90 sessions' step texts with v1.5, applies the matryoshka
layer-norm, and caches the FULL 768-dim vectors. Truncation to each D happens
in sweep.py (layer-norm once here, then slice).

v1.5 needs a task prefix. We use "classification: " consistently (the exp3
task is fallacy/valid classification). The prefix choice is a noted confound,
not hidden.

Ollama ignores the `dimensionality` option and returns full 768-dim, so we
truncate client-side per the HF/nomic recipe:
    embeddings = F.layer_norm(embeddings, (768,))
    embeddings[..., :D]
"""
import os, json, time, argparse
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
EXP3 = os.path.join(HERE, "..", "exp3_markov_trajectories", "results")
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

MODEL = "nomic-embed-text:v1.5"
OLLAMA = "http://localhost:11434/api/embed"
PREFIX = "classification: "
EMB_DIM = 768
BATCH = 32


def load_sessions():
    """Return list of (session_idx, [step_texts...], label)."""
    with open(os.path.join(EXP3, "sessions.json"), encoding="utf-8") as f:
        sessions = json.load(f)
    with open(os.path.join(EXP3, "session_labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    out = []
    for si, s in enumerate(sessions):
        out.append((si, s["steps"], labels[si]["label"]))
    return out


def embed_batch(texts):
    """Embed a batch of texts via Ollama; returns (B, 768) float32."""
    resp = requests.post(OLLAMA, json={"model": MODEL, "input": texts}, timeout=120)
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=0, help="0 = all 90")
    args = ap.parse_args()

    sessions = load_sessions()
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]
    print(f"Re-embedding {len(sessions)} sessions with {MODEL} (prefix={PREFIX!r})")

    # Flatten all steps, embed in batches
    all_texts = [PREFIX + st for _, steps, _ in sessions for st in steps]
    print(f"  total steps: {len(all_texts)}")
    embs = []
    t0 = time.time()
    for i in range(0, len(all_texts), BATCH):
        chunk = all_texts[i : i + BATCH]
        e = embed_batch(chunk)
        embs.append(e)
        done = min(i + BATCH, len(all_texts))
        print(f"  embedded {done}/{len(all_texts)}  ({time.time()-t0:.0f}s)")
    E = np.concatenate(embs, axis=0)  # (N_steps, 768)
    print(f"  raw shape: {E.shape}")

    # Matryoshka layer-norm (per the nomic recipe), then cache full 768-dim
    E = E - E.mean(1, keepdims=True)
    E = E / (E.std(1, keepdims=True) + 1e-6)
    E = E * (EMB_DIM ** 0.5)  # re-scale so norm ~ sqrt(D) after truncation

    # Rebuild per-session arrays (T, 768)
    per_session = []
    idx = 0
    for _, steps, label in sessions:
        n = len(steps)
        per_session.append({"session_idx": _, "label": label, "emb": E[idx : idx + n]})
        idx += n

    np.save(os.path.join(OUT, "emb_v15_full.npy"), E)
    with open(os.path.join(OUT, "sessions_v15.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"session_idx": s["session_idx"], "label": s["label"],
              "n_steps": s["emb"].shape[0]} for s in per_session],
            f, indent=2)
    print(f"  cached full 768-dim to {OUT}/emb_v15_full.npy")
    print(f"  cached session meta to {OUT}/sessions_v15.json")


if __name__ == "__main__":
    main()
