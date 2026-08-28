"""
Experiment 5: Differential Shape Features vs Static Probe (the pilot gate)

Question: do trajectory SHAPE features (velocity, curvature, loop-closure)
separate fallacy from valid reasoning BETTER than the static endpoint probe
from exp1/exp3?

This is the gate for the pilot architecture (DESIGN.md). If shape features
don't beat the static baseline on data we already have, no amount of custom
encoding fixes it. If they do, the temporal/velocity framing has legs.

Reuses the 90 exp3 sessions (45 fallacy / 45 valid, 6 steps each, 20 topics)
and their step embeddings for both encoders (nomic-embed-text, qwen3-embedding).

Three feature families, all compared with the SAME 5-fold CV protocol:
  A. Static baseline  (centroid + final + traj_var + length)  [from predict.py]
  B. Hand-crafted shape stats (velocity, curvature, loop-closure, straightness)
  C. Relative-delta representation (PCA'd concatenated deltas)  [pilot's "shapes
     as relative relationships" idea]

Falsification:
  - shape AUC > static AUC  -> trajectory shape adds signal (pilot has legs)
  - shape AUC ~= static AUC -> shape is redundant (pilot needs the learned encoder)
  - shape AUC < static AUC  -> shape features are worse (rethink)

NOTE on stylistic-vs-logical: this data has no paraphrases, so we cannot yet
separate "shape of the reasoning" from "shape of the surface text." That control
is a follow-up. This experiment only asks whether SHAPE carries signal at all.
"""
import os, sys, json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "exp3_markov_trajectories", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

ENCODERS = ["nomic_embed_text", "qwen3_embedding"]
RANDOM_STATE = 42


def load_sessions(enc_name):
    """Return {session_idx: np.array of step embeddings} + labels."""
    X = np.load(os.path.join(RESULTS_DIR, f"trajectory_embeddings_{enc_name}.npy"))
    with open(os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")) as f:
        meta = json.load(f)
    with open(os.path.join(RESULTS_DIR, "session_labels.json")) as f:
        labels = json.load(f)

    sess_embs = defaultdict(list)
    for mi, m in enumerate(meta):
        sess_embs[m["session_idx"]].append(X[mi])

    sessions = {}
    for si, embs in sess_embs.items():
        if si >= len(labels):
            continue
        arr = np.array(embs)
        if arr.shape[0] < 2 or np.all(arr == 0):
            continue
        sessions[si] = arr
    return sessions, labels


def shape_stats(embs):
    """Hand-crafted differential-geometry features of a trajectory.

    embs: (T, D) array of step embeddings.
    Returns a fixed-length feature vector (translation-invariant where noted).
    """
    T, D = embs.shape
    deltas = np.diff(embs, axis=0)          # (T-1, D) velocities
    speeds = np.linalg.norm(deltas, axis=1) # per-step speed

    # velocity stats
    mean_speed = speeds.mean()
    max_speed = speeds.max()
    min_speed = speeds.min()
    std_speed = speeds.std()
    total_path = speeds.sum()               # total path length

    # direction change (angle between consecutive deltas)
    dir_changes = []
    for t in range(len(deltas) - 1):
        a, b = deltas[t], deltas[t + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
            dir_changes.append(np.arccos(cos))
    mean_dir_change = np.mean(dir_changes) if dir_changes else 0.0
    max_dir_change = np.max(dir_changes) if dir_changes else 0.0
    n_reversals = sum(1 for c in dir_changes if c > np.pi / 2)  # big turns

    # loop-closure / path recurrence: how much the path returns near itself
    # (max over pairs of the negative distance between non-adjacent points)
    loop = 0.0
    if T >= 3:
        min_gap = np.inf
        for i in range(T):
            for j in range(i + 2, T):  # skip adjacent (trivially close)
                d = np.linalg.norm(embs[i] - embs[j])
                if d < min_gap:
                    min_gap = d
        loop = min_gap if np.isfinite(min_gap) else 0.0

    # straightness: net displacement / total path (1 = straight, <1 = curvy)
    net = np.linalg.norm(embs[-1] - embs[0])
    straightness = net / total_path if total_path > 0 else 0.0

    # final displacement magnitude (end-to-start)
    final_disp = net

    return np.array([
        mean_speed, max_speed, min_speed, std_speed, total_path,
        mean_dir_change, max_dir_change, n_reversals,
        loop, straightness, final_disp,
    ])


def delta_representation(embs, n_pca=50):
    """Relative-delta representation: concatenated velocities, PCA-reduced.

    This is the pilot's 'shapes as relative relationships' idea — translation
    invariant (deltas), magnitude-preserving (raw vectors, not normalized).
    """
    deltas = np.diff(embs, axis=0)          # (T-1, D)
    flat = deltas.reshape(-1)               # concatenate
    return flat


def run(enc_name):
    print(f"\n{'='*64}")
    print(f"Exp5 — Differential Shape Features: {enc_name}")
    print(f"{'='*64}")

    sessions, labels = load_sessions(enc_name)
    keys = sorted(sessions)
    y = np.array([1 if labels[si]["label"] == "fallacy" else 0 for si in keys])
    n_f, n_v = (y == 1).sum(), (y == 0).sum()
    print(f"  Sessions: {len(keys)} ({n_f} fallacy, {n_v} valid)")

    # ---- A. Static baseline (same as predict.py) ----
    X_static = []
    for si in keys:
        embs = sessions[si]
        centroid = embs.mean(axis=0)
        final = embs[-1]
        if embs.shape[0] >= 3:
            pca = PCA(n_components=1)
            pca.fit(embs)
            traj_var = pca.explained_variance_ratio_[0]
        else:
            traj_var = 0.0
        X_static.append(np.concatenate([centroid, final, [traj_var, embs.shape[0]]]))
    X_static = np.array(X_static)

    # ---- B. Hand-crafted shape stats ----
    X_shape = np.array([shape_stats(sessions[si]) for si in keys])

    # ---- C. Relative-delta representation (PCA'd concatenated deltas) ----
    # All sessions have 6 steps -> 5 deltas. Concatenate, then PCA to 50d.
    X_delta_raw = np.array([delta_representation(sessions[si]) for si in keys])
    pca = PCA(n_components=min(50, X_delta_raw.shape[1], X_delta_raw.shape[0]))
    X_delta = pca.fit_transform(X_delta_raw)

    def cv_auc(X, y, name):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        clf = LogisticRegression(max_iter=3000, C=1.0)
        aucs = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
        accs = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy")
        print(f"  {name:<38} AUC={aucs.mean():.3f}±{aucs.std():.3f}  Acc={accs.mean():.3f}±{accs.std():.3f}")
        return aucs.mean(), accs.mean()

    print(f"\n  Feature families (5-fold CV, same protocol):")
    static_auc, static_acc = cv_auc(X_static, y, "A. Static (centroid+final)")
    shape_auc, shape_acc = cv_auc(X_shape, y, "B. Shape stats (velocity/curv/loop)")
    delta_auc, delta_acc = cv_auc(X_delta, y, "C. Relative-delta (PCA50)")

    # ---- Combined: shape + static ----
    X_comb = np.concatenate([X_shape, X_static], axis=1)
    comb_auc, comb_acc = cv_auc(X_comb, y, "B+C. Shape + Static")

    print(f"\n  SUMMARY ({enc_name}):")
    print(f"  {'Method':<38} {'AUC':<10} {'Acc':<10}")
    print(f"  {'A. Static baseline':<38} {static_auc:<10.3f} {static_acc:<10.3f}")
    print(f"  {'B. Shape stats':<38} {shape_auc:<10.3f} {shape_acc:<10.3f}")
    print(f"  {'C. Relative-delta':<38} {delta_auc:<10.3f} {delta_acc:<10.3f}")
    print(f"  {'B+C. Shape+Static':<38} {comb_auc:<10.3f} {comb_acc:<10.3f}")

    return {
        "static_auc": float(static_auc), "static_acc": float(static_acc),
        "shape_auc": float(shape_auc), "shape_acc": float(shape_acc),
        "delta_auc": float(delta_auc), "delta_acc": float(delta_acc),
        "combined_auc": float(comb_auc), "combined_acc": float(comb_acc),
        "n_sessions": len(keys), "n_fallacy": int(n_f), "n_valid": int(n_v),
    }


def main():
    all_results = {}
    for enc in ENCODERS:
        try:
            all_results[enc] = run(enc)
        except FileNotFoundError as e:
            print(f"\n  [{enc}] Files not found: {e}")
            print(f"  Run exp3 gen_bigger.py / build_mc.py first.")

    out = os.path.join(OUT_DIR, "exp5_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    # Verdict
    print(f"\n  VERDICT:")
    for enc in ENCODERS:
        if enc not in all_results:
            continue
        r = all_results[enc]
        sa, sh, de = r["static_auc"], r["shape_auc"], r["delta_auc"]
        # The pilot's core idea is the RELATIVE-DELTA representation (C),
        # not the hand-crafted shape stats (B). Compare both.
        if de > sa + 0.05:
            v = "DELTA BEATS STATIC — pilot's core mechanism has legs"
        elif de > sa - 0.05:
            v = "DELTA ~= STATIC — redundant, need learned encoder"
        else:
            v = "DELTA WORSE — rethink"
        print(f"  {enc}: static={sa:.3f} shape_stats={sh:.3f} delta={de:.3f} -> {v}")


if __name__ == "__main__":
    main()
