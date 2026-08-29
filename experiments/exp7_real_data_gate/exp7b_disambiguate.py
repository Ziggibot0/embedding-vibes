"""
Exp7b — disambiguate exp7's negative result.

Exp7: static 0.901 vs delta 0.655 on 2000 real sessions. Two explanations:
  (a) zero-padding diluted the delta signal (length confound inside deltas)
  (b) static geometry genuinely carries more outcome signal than velocities

Tests (all cache-only, no re-embedding):
  1. mean-velocity + last-velocity (FIXED-length delta features, no padding)
  2. length-stratified rerun (one length bucket, no padding possible)
  3. prefix curves on real data (early warning value of STATIC features)
  4. static delta decomposition: which static component carries the signal?
"""
import os, json
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA

DATA = os.path.join(os.path.dirname(__file__), "..", "exp6_joint_jepa", "data", "sessions.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
SEED = 42
EMB_MODEL = "nomic-embed-text"


def load():
    arr = np.load(os.path.join(OUT_DIR, f"emb_cache_{EMB_MODEL}.npy"), allow_pickle=True).item()
    _lens = Counter(len(v) for v in arr.values() if isinstance(v, (list, tuple)))
    EMB_LEN = _lens.most_common(1)[0][0]
    sessions = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["source"] == "exgentic":
                sessions.append(s)
    sess_embs, y = [], []
    for s in sessions:
        embs = [arr[st] for st in s["steps"]
                if st in arr and isinstance(arr[st], list) and len(arr[st]) == EMB_LEN]
        if len(embs) >= 2:
            sess_embs.append(np.array(embs, dtype=np.float32))
            y.append(1 if s["success"] else 0)
    return sess_embs, np.array(y)


def cv_auc(X, y, name):
    X = np.asarray(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    aucs = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
    return float(aucs.mean()), float(aucs.std())


def main():
    sess_embs, y = load()
    print(f"Sessions: {len(sess_embs)} ({y.sum()} success)")
    lengths = np.array([e.shape[0] for e in sess_embs])
    print(f"Session lengths: min={lengths.min()} max={lengths.max()} "
          f"mean={lengths.mean():.1f} median={np.median(lengths):.0f}")
    print(f"length histogram (deciles): {np.percentile(lengths, [10,25,50,75,90])}")
    results = {}

    # ---- 1. Fixed-length velocity features (NO padding) ----
    print("\n--- 1. Fixed-length velocity features (no padding) ---")
    mean_v = np.array([np.diff(e, axis=0).mean(0) for e in sess_embs])
    last_v = np.array([np.diff(e, axis=0)[-1] for e in sess_embs])
    speed = np.array([[np.linalg.norm(np.diff(e, axis=0), axis=1).mean(),
                       np.linalg.norm(np.diff(e, axis=0), axis=1).max(),
                       np.linalg.norm(e[-1] - e[0])] for e in sess_embs])
    mv_auc, _ = cv_auc(mean_v, y, "mean_velocity")
    lv_auc, _ = cv_auc(last_v, y, "last_velocity")
    sp_auc, _ = cv_auc(speed, y, "speed_stats")
    print(f"  mean-velocity (768d, fixed)   AUC={mv_auc:.3f}")
    print(f"  last-velocity (768d, fixed)   AUC={lv_auc:.3f}")
    print(f"  speed stats (3d)              AUC={sp_auc:.3f}")
    results["mean_velocity_auc"] = mv_auc
    results["last_velocity_auc"] = lv_auc
    results["speed_stats_auc"] = sp_auc

    # ---- 2. Length-stratified (one bucket, no padding possible) ----
    print("\n--- 2. Length-stratified rerun (18-step sessions only) ---")
    bucket = 18
    idx = np.where(lengths == bucket)[0]
    if len(idx) > 60:
        sub = [sess_embs[i] for i in idx]
        ys = y[idx]
        Xs = np.array([np.concatenate([e.mean(0), e[-1]]) for e in sub])
        Xd = np.array([np.diff(e, axis=0).reshape(-1) for e in sub])
        p = PCA(n_components=min(50, Xd.shape[1], Xd.shape[0]))
        Xd = p.fit_transform(Xd)
        s_auc, _ = cv_auc(Xs, ys, "stratified_static")
        d_auc, _ = cv_auc(Xd, ys, "stratified_delta")
        print(f"  n={len(idx)} (all exactly {bucket} steps)")
        print(f"  static AUC={s_auc:.3f}   delta AUC={d_auc:.3f}")
        results["stratified_static_auc"] = s_auc
        results["stratified_delta_auc"] = d_auc
        results["stratified_n"] = int(len(idx))
    else:
        print(f"  only {len(idx)} sessions of exactly {bucket} steps — skipping")
        results["stratified_n"] = int(len(idx))

    # ---- 3. Prefix curves on real data (static features) ----
    print("\n--- 3. Prefix curves (static features, real data) ---")
    max_len = int(lengths.max())
    prefix_aucs = {}
    for n in [3, 5, 8, 12, 16, 25, 50]:
        if n > max_len:
            continue
        X = []
        for e in sess_embs:
            pre = e[:n]
            if pre.shape[0] < 2:
                pre = np.vstack([pre, pre])
            X.append(np.concatenate([pre.mean(0), pre[-1]]))
        auc, _ = cv_auc(np.array(X), y, "prefix")
        prefix_aucs[n] = auc
        print(f"  first {n:>3} steps: AUC={auc:.3f}")
    results["prefix_static_aucs"] = prefix_aucs

    # ---- 4. Static component decomposition ----
    print("\n--- 4. Static decomposition ---")
    cent = np.array([e.mean(0) for e in sess_embs])
    fin = np.array([e[-1] for e in sess_embs])
    c_auc, _ = cv_auc(cent, y, "centroid")
    f_auc, _ = cv_auc(fin, y, "final")
    print(f"  centroid only  AUC={c_auc:.3f}")
    print(f"  final only     AUC={f_auc:.3f}")
    results["centroid_auc"] = c_auc
    results["final_auc"] = f_auc

    out = os.path.join(OUT_DIR, "exp7b_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()