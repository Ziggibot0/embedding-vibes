"""
Exp7 — the cheapest decisive test: does temporal/velocity shape separate
SUCCESS from FAILURE on REAL agentic data, better than static position?

This is the pilot's core falsifiable claim, tested on real labeled data
(2000 Exgentic sessions, 818 success / 1182 failure) instead of the 90
synthetic fallacy sessions from exp3/exp5.

No training. Just:
  1. Embed each step with nomic-embed-text (fast)
  2. Static features (centroid + final)  -> probe
  3. Delta features (concatenated velocities, PCA) -> probe
  4. Compare AUC (5-fold CV, same protocol)

If delta AUC > static AUC -> the pilot's core mechanism holds on real data.
If delta AUC ~= static AUC -> temporal shape is redundant on real data.
If delta AUC < static AUC -> temporal shape is worse; rethink.
"""
import os, json, time, requests
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA

DATA = os.path.join(os.path.dirname(__file__), "..", "exp6_joint_jepa", "data", "sessions.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
OLLAMA = "http://localhost:11434"
EMB_MODEL = "nomic-embed-text"


def embed_text(text, retries=3):
    # nomic-embed-text context is ~8192 tokens; truncate long texts
    if len(text) > 20000:
        text = text[:20000]
    for attempt in range(retries):
        try:
            r = requests.post(f"{OLLAMA}/api/embeddings",
                              json={"model": EMB_MODEL, "prompt": text}, timeout=120)
            if r.status_code == 200:
                return r.json()["embedding"]
            # 500 on a specific text — truncate harder and retry
            if "context length" in r.text and len(text) > 5000:
                text = text[:5000]
                continue
            if attempt == retries - 1:
                raise RuntimeError(f"embed failed after {retries}: {r.status_code} {r.text[:200]}")
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(1)
    raise RuntimeError("unreachable")


def load_exgentic_sessions():
    """Load only Exgentic sessions (they have real success/failure labels)."""
    sessions = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["source"] == "exgentic":
                sessions.append(s)
    return sessions


def main():
    sessions = load_exgentic_sessions()
    print(f"Exgentic sessions: {len(sessions)} "
          f"({sum(1 for s in sessions if s['success'])} success, "
          f"{sum(1 for s in sessions if not s['success'])} failure)")

    # Embed all steps (dedupe identical text to save calls)
    all_texts = set()
    for s in sessions:
        for step in s["steps"]:
            all_texts.add(step)
    all_texts = list(all_texts)
    print(f"Unique step texts: {len(all_texts)}")

    # Disk cache so reruns don't re-embed
    cache_path = os.path.join(OUT_DIR, f"emb_cache_{EMB_MODEL}.npy")
    cache = {}
    if os.path.exists(cache_path):
        arr = np.load(cache_path, allow_pickle=True).item()
        cache = {k: v for k, v in arr.items()}
        print(f"Loaded {len(cache)} cached embeddings")

    # Concurrent embedding (nomic is fast; sequential is the bottleneck)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    todo = [t for t in all_texts if t not in cache]
    print(f"To embed: {len(todo)}")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(embed_text, t): t for t in todo}
        for fut in as_completed(futs):
            text = futs[fut]
            try:
                cache[text] = fut.result()
            except Exception as e:
                print(f"  WARN: failed to embed {text[:60]!r}: {e}")
            done += 1
            if done % 500 == 0:
                print(f"  embedded {done}/{len(todo)} ({time.time()-t0:.0f}s)")
    print(f"Embedded {len(cache)}/{len(all_texts)} unique texts in {time.time()-t0:.0f}s")
    np.save(cache_path, np.array(cache, dtype=object), allow_pickle=True)

    # Build per-session step embeddings (defensive: skip malformed cache rows)
    # EMB_LEN from the MODE of value lengths — one empty entry exists (empty text key)
    from collections import Counter as _Counter
    _lens = _Counter(len(v) for v in cache.values() if isinstance(v, (list, tuple)))
    EMB_LEN = _lens.most_common(1)[0][0] if _lens else 768
    sess_embs = []
    y = []
    for s in sessions:
        embs = []
        for step in s["steps"]:
            if step in cache:
                v = cache[step]
                if isinstance(v, (list, tuple)) and len(v) == EMB_LEN:
                    embs.append(v)
        if len(embs) < 2:
            continue
        sess_embs.append(np.array(embs, dtype=np.float32))
        y.append(1 if s["success"] else 0)
    y = np.array(y)
    print(f"Valid sessions: {len(sess_embs)} ({y.sum()} success)")

    # ---- Static features (centroid + final) ----
    X_static = []
    for embs in sess_embs:
        centroid = embs.mean(0)
        final = embs[-1]
        X_static.append(np.concatenate([centroid, final]))
    X_static = np.array(X_static)

    # ---- Delta features (concatenated velocities) ----
    # Sessions have VARIABLE length -> pad delta sequences to the max length.
    max_deltas = max(embs.shape[0] - 1 for embs in sess_embs)
    D = sess_embs[0].shape[1]
    X_delta_raw = np.array([
        np.concatenate([np.diff(embs, axis=0).reshape(-1),
                        np.zeros(((max_deltas - (embs.shape[0] - 1)) * D,))])
        for embs in sess_embs])
    pca = PCA(n_components=min(50, X_delta_raw.shape[1], X_delta_raw.shape[0]))
    X_delta = pca.fit_transform(X_delta_raw)

    def cv_auc(X, name):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        clf = LogisticRegression(max_iter=3000, C=1.0)
        aucs = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
        accs = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy")
        print(f"  {name:<40} AUC={aucs.mean():.3f}±{aucs.std():.3f}  Acc={accs.mean():.3f}")
        return aucs.mean(), accs.mean()

    print(f"\n  Success/failure separation (5-fold CV):")
    # LENGTH-ONLY CONTROL (mandatory: prior work showed length leakage AUC 0.66)
    lens = np.array([[embs.shape[0]] for embs in sess_embs])
    len_auc, len_acc = cv_auc(lens, "Length-only control (n steps)")
    static_auc, static_acc = cv_auc(X_static, "Static (centroid+final)")
    delta_auc, delta_acc = cv_auc(X_delta, "Delta (velocities, PCA50)")

    # Combined
    X_comb = np.concatenate([X_static, X_delta], axis=1)
    comb_auc, comb_acc = cv_auc(X_comb, "Static + Delta")

    print(f"\n  SUMMARY:")
    print(f"  {'Method':<40} {'AUC':<10} {'Acc':<10}")
    print(f"  {'Length-only':<40} {len_auc:<10.3f} {len_acc:<10.3f}")
    print(f"  {'Static':<40} {static_auc:<10.3f} {static_acc:<10.3f}")
    print(f"  {'Delta':<40} {delta_auc:<10.3f} {delta_acc:<10.3f}")
    print(f"  {'Static+Delta':<40} {comb_auc:<10.3f} {comb_acc:<10.3f}")

    print(f"\n  VERDICT:")
    if delta_auc > max(static_auc, len_auc) + 0.05:
        v = "DELTA BEATS STATIC (and length control) on real data — pilot has legs"
    elif delta_auc > max(static_auc, len_auc) - 0.05:
        v = "DELTA ~= STATIC/LENGTH on real data — temporal shape is redundant"
    else:
        v = "DELTA WORSE on real data — rethink"
    print(f"  static={static_auc:.3f} delta={delta_auc:.3f} length={len_auc:.3f} -> {v}")

    out = os.path.join(OUT_DIR, "exp7_results.json")
    with open(out, "w") as f:
        json.dump({"static_auc": float(static_auc), "static_acc": float(static_acc),
                   "delta_auc": float(delta_auc), "delta_acc": float(delta_acc),
                   "length_auc": float(len_auc), "length_acc": float(len_acc),
                   "combined_auc": float(comb_auc), "combined_acc": float(comb_acc),
                   "n_sessions": len(sess_embs), "n_success": int(y.sum()),
                   "n_failure": int((y == 0).sum())}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
