"""
Markov chain classifier: predict fallacy vs valid from trajectory log-likelihood.

Also compares against static linear probe baseline (from exp1) to test
whether trajectory adds signal beyond single-embedding classification.

Key metric: can transition log-likelihood under T_fallacy vs T_valid
classify sessions better than a static probe on the final step embedding?
"""
import os, sys, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, f1_score
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
K_CLUSTERS = 200

ENCODERS = ["nomic_embed_text", "qwen3_embedding"]


def load_session_trajectories(enc_name):
    """Load states and organize into session trajectories with labels."""
    states = np.load(os.path.join(RESULTS_DIR, f"states_{enc_name}.npy"))
    meta_path = os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")
    labels_path = os.path.join(RESULTS_DIR, "session_labels.json")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(labels_path) as f:
        session_labels = json.load(f)

    # Group states by session
    session_trajs = {}
    for mi, m in enumerate(meta):
        si = m["session_idx"]
        if si not in session_trajs:
            session_trajs[si] = []
        if states[mi] >= 0:
            session_trajs[si].append(int(states[mi]))

    trajectories = []
    labels = []
    fallacy_types = []

    for si, traj in session_trajs.items():
        if si >= len(session_labels):
            continue
        if len(traj) < 2:
            continue
        trajectories.append(np.array(traj))
        labels.append(session_labels[si]["label"])
        fallacy_types.append(session_labels[si].get("fallacy_type"))

    return trajectories, labels, fallacy_types


def markov_log_likelihood(trajectory, T, log=True):
    """Compute log-likelihood of a trajectory under transition matrix T.
    
    LL = sum_t log T[s_t, s_{t+1}]
    """
    ll = 0.0
    for t in range(len(trajectory) - 1):
        i, j = trajectory[t], trajectory[t + 1]
        if i < T.shape[0] and j < T.shape[1]:
            p = T[i, j]
            if p > 0:
                ll += np.log(p) if log else p
            else:
                ll += -20.0 if log else 0.0  # log(1e-20) ≈ -46
    return ll


def markov_predict(trajectory, T_fallacy, T_valid):
    """Classify trajectory by comparing log-likelihoods.
    
    Returns: "fallacy" if LL_fallacy > LL_valid, else "valid"
    Also returns: log-likelihood ratio (positive = fallacy)
    """
    ll_f = markov_log_likelihood(trajectory, T_fallacy)
    ll_v = markov_log_likelihood(trajectory, T_valid)
    llr = ll_f - ll_v  # log-likelihood ratio
    pred = "fallacy" if llr > 0 else "valid"
    return pred, llr


def static_probe_features(enc_name, trajectories, session_labels):
    """Extract static features from trajectory embeddings for baseline comparison.
    
    Features per session:
    - mean embedding (centroid of all steps)
    - final step embedding
    - trajectory variance
    - trajectory length
    """
    emb_path = os.path.join(RESULTS_DIR, f"trajectory_embeddings_{enc_name}.npy")
    meta_path = os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")

    X = np.load(emb_path)
    with open(meta_path) as f:
        meta = json.load(f)

    # Group embeddings by session
    session_embs = {}
    for mi, m in enumerate(meta):
        si = m["session_idx"]
        if si not in session_embs:
            session_embs[si] = []
        session_embs[si].append(X[mi])

    features = []
    valid_indices = []

    for si in sorted(session_embs.keys()):
        if si >= len(session_labels):
            continue
        embs = np.array(session_embs[si])
        if embs.shape[0] < 2 or np.all(embs == 0):
            continue

        # Centroid
        centroid = embs.mean(axis=0)
        # Final step
        final = embs[-1]
        # Trajectory variance (explained by first PC)
        if embs.shape[0] >= 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca.fit(embs)
            traj_var = pca.explained_variance_ratio_[0]
        else:
            traj_var = 0.0

        feat = np.concatenate([centroid, final, [traj_var, embs.shape[0]]])
        features.append(feat)
        valid_indices.append(si)

    return np.array(features), valid_indices


def run_markov_cv(enc_name):
    """Run cross-validated Markov chain classifier."""
    print(f"\n{'='*60}")
    print(f"Markov Classifier: {enc_name}")
    print(f"{'='*60}")

    # Load transition matrices
    T_fallacy = np.load(os.path.join(RESULTS_DIR, f"T_fallacy_{enc_name}.npy"))
    T_valid = np.load(os.path.join(RESULTS_DIR, f"T_valid_{enc_name}.npy"))

    # Load trajectories
    trajectories, labels, fallacy_types = load_session_trajectories(enc_name)

    y_binary = np.array([1 if l == "fallacy" else 0 for l in labels])
    llrs = np.array([markov_predict(t, T_fallacy, T_valid)[1] for t in trajectories])

    n_fallacy = sum(y_binary == 1)
    n_valid = sum(y_binary == 0)
    print(f"  Sessions: {len(trajectories)} ({n_fallacy} fallacy, {n_valid} valid)")

    # --- Markov classifier performance (no CV needed — model is fixed) ---
    preds = np.array([1 if llr > 0 else 0 for llr in llrs])
    acc = accuracy_score(y_binary, preds)
    f1 = f1_score(y_binary, preds)

    # AUC from continuous LLR scores
    if len(set(y_binary)) > 1:
        auc = roc_auc_score(y_binary, llrs)
    else:
        auc = float('nan')

    print(f"\n  Markov Log-Likelihood Classifier:")
    print(f"    Accuracy: {acc:.3f}")
    print(f"    F1:       {f1:.3f}")
    print(f"    AUC:      {auc:.3f}")

    # --- Per-fallacy-type analysis ---
    print(f"\n  Per-fallacy-type LLR (positive = classified as fallacy):")
    ft_llrs = {}
    for i, ft in enumerate(fallacy_types):
        if ft and y_binary[i] == 1:
            if ft not in ft_llrs:
                ft_llrs[ft] = []
            ft_llrs[ft].append(llrs[i])

    # Also get valid LLRs
    valid_llrs = [llrs[i] for i in range(len(labels)) if labels[i] == "valid"]
    print(f"    valid: mean LLR = {np.mean(valid_llrs):.3f} ± {np.std(valid_llrs):.3f}")

    for ft in sorted(ft_llrs.keys()):
        vals = ft_llrs[ft]
        pct_fallacy = sum(1 for v in vals if v > 0) / len(vals) * 100
        print(f"    {ft}: mean LLR = {np.mean(vals):.3f} ± {np.std(vals):.3f}, classified fallacy: {pct_fallacy:.0f}%")

    # --- Static probe baseline ---
    print(f"\n  Static Linear Probe Baseline:")
    try:
        session_labels_list = json.load(open(os.path.join(RESULTS_DIR, "session_labels.json")))
        X_static, valid_idx = static_probe_features(enc_name, trajectories, session_labels_list)
        y_static = np.array([1 if session_labels_list[si]["label"] == "fallacy" else 0 for si in valid_idx])

        # Filter out zero rows
        nonzero = np.any(X_static != 0, axis=1)
        X_static = X_static[nonzero]
        y_static = y_static[nonzero]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_static)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=2000, C=1.0)

        cv_acc = cross_val_score(clf, X_scaled, y_static, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(clf, X_scaled, y_static, cv=cv, scoring='f1')
        cv_auc = cross_val_score(clf, X_scaled, y_static, cv=cv, scoring='roc_auc')

        print(f"    Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
        print(f"    F1:       {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
        print(f"    AUC:      {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    except Exception as e:
        print(f"    Error: {e}")
        cv_acc = np.array([0])
        cv_auc = np.array([0])

    # --- Combined classifier: Markov + Static ---
    print(f"\n  Combined (Markov LLR + Static features):")
    try:
        # Align trajectories with static features
        combined_features = []
        combined_labels = []
        for i, (traj, label) in enumerate(zip(trajectories, labels)):
            if i < len(llrs):
                # Use LLR + trajectory shape features
                feat = np.array([
                    llrs[i],
                    len(traj),
                    np.std([markov_log_likelihood(traj[:k+1], T_fallacy) -
                            markov_log_likelihood(traj[:k+1], T_valid)
                            for k in range(1, len(traj))]) if len(traj) > 2 else 0.0,
                ])
                combined_features.append(feat)
                combined_labels.append(1 if label == "fallacy" else 0)

        X_comb = np.array(combined_features)
        y_comb = np.array(combined_labels)

        scaler = StandardScaler()
        X_comb_scaled = scaler.fit_transform(X_comb)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=2000, C=1.0)

        cv_comb_acc = cross_val_score(clf, X_comb_scaled, y_comb, cv=cv, scoring='accuracy')
        cv_comb_auc = cross_val_score(clf, X_comb_scaled, y_comb, cv=cv, scoring='roc_auc')

        print(f"    Accuracy: {cv_comb_acc.mean():.3f} ± {cv_comb_acc.std():.3f}")
        print(f"    AUC:      {cv_comb_auc.mean():.3f} ± {cv_comb_auc.std():.3f}")
    except Exception as e:
        print(f"    Error: {e}")
        cv_comb_acc = np.array([0])
        cv_comb_auc = np.array([0])

    # --- Summary comparison ---
    print(f"\n  SUMMARY:")
    print(f"  {'Method':<30} {'Accuracy':<15} {'AUC':<15}")
    print(f"  {'Markov (transition LLR)':<30} {acc:<15.3f} {auc:<15.3f}")
    print(f"  {'Static (centroid+final)':<30} {cv_acc.mean():<15.3f} {cv_auc.mean():<15.3f}")
    print(f"  {'Combined':<30} {cv_comb_acc.mean():<15.3f} {cv_comb_auc.mean():<15.3f}")

    return {
        "markov_accuracy": float(acc),
        "markov_f1": float(f1),
        "markov_auc": float(auc),
        "static_accuracy": float(cv_acc.mean()),
        "static_auc": float(cv_auc.mean()),
        "combined_accuracy": float(cv_comb_acc.mean()),
        "combined_auc": float(cv_comb_auc.mean()),
        "per_fallacy_llr": {ft: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                                "pct_classified_fallacy": float(sum(1 for x in v if x > 0) / len(v) * 100)}
                           for ft, v in ft_llrs.items()},
    }


def run_markov_loo(enc_name):
    """Leave-one-out Markov classifier.

    The canonical anti-overfit gate for Markov trajectory classification:
    build each trajectory's (T_fallacy, T_valid) using the OTHER trajectories,
    then score the held-out one. This removes in-sample leakage (the original
    study scored each fallacy session against a T_fallacy it helped construct,
    which trivially yields AUC=1.0).
    """
    states = np.load(os.path.join(RESULTS_DIR, f"states_{enc_name}.npy"))
    meta_path = os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")
    with open(meta_path) as f:
        meta = json.load(f)
    labels_path = os.path.join(RESULTS_DIR, "studyA_session_labels.json"
                              if os.path.exists(os.path.join(RESULTS_DIR, "studyA_session_labels.json"))
                              else os.path.join(RESULTS_DIR, "session_labels.json"))
    with open(labels_path) as f:
        session_labels = json.load(f)

    # States per session (skip un-observed states)
    sessions = {}
    for mi, m in enumerate(meta):
        si = m["session_idx"]
        if states[mi] >= 0:
            sessions.setdefault(si, []).append(int(states[mi]))

    sessions = {si: tr for si, tr in sessions.items() if si < len(session_labels) and len(tr) >= 3}
    sess_keys = sorted(sessions)
    y = np.array([1 if session_labels[si]["label"] == "fallacy" else 0 for si in sess_keys], dtype=int)

    def build_T(trajs, k):
        alpha = 1e-3
        T = np.full((k, k), alpha, dtype=float)
        for tr in trajs:
            for t in range(len(tr) - 1):
                T[tr[t], tr[t+1]] += 1
        rs = T.sum(1, keepdims=True)
        return T / rs

    N = len(sess_keys)
    K = 200

    llrs = np.zeros(N)
    for idx, si in enumerate(sess_keys):
        tr = sessions[si]
        # Leave-one-out per class: drop THIS session from its own bag.
        bag_f = [np.array(v) for s2, v in sessions.items()
                 if y[list(sess_keys).index(s2)] == 1 and s2 != si]
        bag_v = [np.array(v) for s2, v in sessions.items()
                 if y[list(sess_keys).index(s2)] == 0]
        T_f = build_T(bag_f, K) if bag_f else np.full((K, K), 1e-6, dtype=float)
        T_v = build_T(bag_v, K) if bag_v else np.full((K, K), 1e-6, dtype=float)
        # Compare likelihoods of THIS held-out traj against both bag matrices.
        def LL(traj, T):
            out = 0.0
            for t in range(len(traj) - 1):
                p = T[traj[t], traj[t+1]]
                out += (np.log(p) if p > 0 else -20.0)
            return out
        llrs[idx] = LL(tr, T_f) - LL(tr, T_v)

    auc = roc_auc_score(y, llrs) if len(set(y)) > 1 else float('nan')
    preds = (llrs > 0).astype(int)
    acc = accuracy_score(y, preds)
    print(f"  LOO Markov: AUC={auc:.3f}  Acc={acc:.3f}  "
          f"fallacy-mean-LLR={np.mean(llrs[y==1]):.2f} valid-mean-LLR={np.mean(llrs[y==0]):.2f}")
    return {"loo_auc": float(auc), "loo_accuracy": float(acc),
            "loo_fallacy_llr_mean": float(np.mean(llrs[y==1])),
            "loo_valid_llr_mean": float(np.mean(llrs[y==0]))}


def main():
    all_results = {}
    loo_flag = "--loo" in sys.argv

    for enc_name in ENCODERS:
        try:
            if loo_flag:
                all_results[enc_name] = run_markov_loo(enc_name)
            else:
                all_results[enc_name] = run_markov_cv(enc_name)
        except FileNotFoundError as e:
            print(f"\n  [{enc_name}] Files not found: {e}")
            print(f"  Run gen_bigger.py / build_mc.py first.")

    out = "predict_results.json" if not loo_flag else "predict_results_loo.json"
    with open(os.path.join(RESULTS_DIR, out), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/{ out}")
    if loo_flag:
        print("Leave-one-out AUC is the anti-overfit gate: a true generalization number.")


if __name__ == "__main__":
    main()