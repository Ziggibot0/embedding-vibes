"""
Exp5b — two nearly-free design curves on the existing exp3 embeddings.
No training, no new embedding — pure analysis of data already on disk.

Curve A — Dimensionality: where is the real elbow?
  The user's 64-dim hypothesis, tested: task AUC (delta representation)
  as a function of PCA dimension on the raw nomic/qwen3 deltas.
  (exp6 showed the learned 64-dim encoder loses signal; this finds what
  dimension a learned/pca space would need to preserve it.)

Curve B — Early detection: how many steps do we need?
  The pilot's early-warning value: AUC as a function of how many leading
  steps of the trajectory are visible (1, 2, ..., 5 of 6). If AUC is
  already high at step 2-3, the pilot has real early-abort value.

Both use the same 5-fold CV protocol as exp5.
"""
import os, json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "exp3_markov_trajectories", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)
ENCODERS = ["nomic_embed_text", "qwen3_embedding"]
SEED = 42


def load_sessions(enc_name):
    X = np.load(os.path.join(RESULTS_DIR, f"trajectory_embeddings_{enc_name}.npy"))
    with open(os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")) as f:
        meta = json.load(f)
    with open(os.path.join(RESULTS_DIR, "session_labels.json")) as f:
        labels = json.load(f)
    sess = defaultdict(list)
    for mi, m in enumerate(meta):
        sess[m["session_idx"]].append(X[mi])
    out = {}
    for si, embs in sess.items():
        if si >= len(labels):
            continue
        arr = np.array(embs)
        if arr.shape[0] < 2 or np.all(arr == 0):
            continue
        out[si] = arr
    return out, labels


def cv_auc(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    aucs = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
    return float(aucs.mean()), float(aucs.std())


def delta_features(embs_list, keys, y, dim=None):
    """Concatenated deltas, optionally PCA-reduced to `dim`."""
    X = np.array([np.diff(embs_list[si], axis=0).reshape(-1) for si in keys])
    if dim is not None:
        pca = PCA(n_components=min(dim, X.shape[1], X.shape[0]))
        X = pca.fit_transform(X)
    return X


def main():
    results = {"curve_a_dimension": {}, "curve_b_early_detection": {}}

    for enc in ENCODERS:
        sessions, labels = load_sessions(enc)
        keys = sorted(sessions)
        embs_list = {si: sessions[si] for si in keys}
        y = np.array([1 if labels[si]["label"] == "fallacy" else 0 for si in keys])
        print(f"\n=== {enc} ===")

        # ---- Curve A: dimensionality elbow ----
        curve_a = {}
        for dim in [8, 16, 32, 50, 64, 96, 128, 256, 512, None]:
            X = delta_features(embs_list, keys, y, dim=dim)
            auc, std = cv_auc(X, y)
            label = str(dim) if dim else "full"
            curve_a[label] = {"auc": auc, "std": std}
            print(f"  dim={label:>5}  AUC={auc:.3f}±{std:.3f}")
        results["curve_a_dimension"][enc] = curve_a

        # ---- Curve B: early detection (prefix length) ----
        curve_b = {}
        max_steps = min(embs.shape[0] for embs in embs_list.values())
        for n_vis in range(2, max_steps + 1):
            X = []
            for si in keys:
                embs = embs_list[si][:n_vis]            # only first n_vis steps
                X.append(np.diff(embs, axis=0).reshape(-1))
            X = np.array(X)
            # small prefix -> few dims; PCA not needed, use directly
            auc, std = cv_auc(X, y)
            curve_b[str(n_vis)] = {"auc": auc, "std": std}
            print(f"  visible_steps={n_vis}  AUC={auc:.3f}±{std:.3f}")
        results["curve_b_early_detection"][enc] = curve_b

    out = os.path.join(OUT_DIR, "exp5b_curves.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    # Verdicts
    print("\n=== VERDICTS ===")
    for enc in ENCODERS:
        a = results["curve_a_dimension"][enc]
        b = results["curve_b_early_detection"][enc]
        # smallest dim within 0.02 of full-dim AUC
        full = a["full"]["auc"]
        elbow = None
        for d in [8, 16, 32, 50, 64, 96, 128, 256, 512]:
            if a[str(d)]["auc"] >= full - 0.02:
                elbow = d
                break
        print(f"{enc}: dimension elbow = {elbow} (full AUC {full:.3f}); "
              f"64-dim AUC = {a['64']['auc']:.3f}")
        # earliest prefix within 0.03 of full-trajectory AUC
        full6 = b[str(max(b, key=int))]["auc"]
        early = None
        for n in sorted(b, key=int):
            if b[n]["auc"] >= full6 - 0.03:
                early = int(n)
                break
        print(f"{enc}: early detection — AUC at {early} visible steps = "
              f"{b[str(early)]['auc']:.3f} (full-trajectory {full6:.3f})")


if __name__ == "__main__":
    main()