"""
Build Markov chain transition matrices from trajectory embeddings.

Pipeline:
1. Load trajectory embeddings (from simulate.py)
2. PCA reduce to 50 dimensions
3. K-means cluster into K discrete states
4. Build transition matrices T_fallacy and T_valid
5. Compute KL divergences, stationary distributions, entropies
6. Save everything for predict.py and visualize.py
"""
import os, json, numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from scipy.special import kl_div
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# --- Config ---
PCA_DIMS = 50
K_CLUSTERS = 200  # number of Markov states
MIN_TRANSITIONS = 2  # minimum transitions to trust a T[i,j] estimate
LAPLACE_ALPHA = 1e-3  # smoothing for unseen transitions

ENCODERS = ["nomic_embed_text", "qwen3_embedding"]


def load_data(enc_name):
    """Load embeddings and session metadata."""
    emb_path = os.path.join(RESULTS_DIR, f"trajectory_embeddings_{enc_name}.npy")
    meta_path = os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")
    labels_path = os.path.join(RESULTS_DIR, "session_labels.json")

    X = np.load(emb_path)
    with open(meta_path) as f:
        meta = json.load(f)
    with open(labels_path) as f:
        session_labels = json.load(f)

    return X, meta, session_labels


def discretize(X, meta, n_pca=PCA_DIMS, k=K_CLUSTERS):
    """PCA reduce + k-means cluster → discrete state indices."""
    # Remove zero rows
    nonzero = np.any(X != 0, axis=1)
    X_clean = X[nonzero]

    print(f"  PCA: {X_clean.shape[1]}d -> {n_pca}d")
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_clean)
    print(f"  Explained variance: {np.cumsum(pca.explained_variance_ratio_)[-1]:.3f}")

    print(f"  K-means: {X_pca.shape[0]} points -> {k} clusters")
    km = MiniBatchKMeans(n_clusters=k, batch_size=1024, random_state=42, n_init=10)
    states = km.fit_predict(X_pca)

    # Map back to original indices (zero rows get state -1)
    full_states = np.full(X.shape[0], -1, dtype=int)
    full_states[nonzero] = states

    return X_pca, pca, km, full_states


def build_transition_matrix(trajectories, k=K_CLUSTERS, alpha=LAPLACE_ALPHA):
    """Build row-stochastic transition matrix from list of state sequences.
    
    trajectories: list of arrays, each is a sequence of state indices
    Returns: T (k x k) row-stochastic matrix
    """
    T = np.full((k, k), alpha)  # Laplace smoothing
    
    for traj in trajectories:
        for t in range(len(traj) - 1):
            i, j = traj[t], traj[t + 1]
            if i >= 0 and j >= 0:  # skip zero-embedding steps
                T[i, j] += 1

    # Row-normalize
    row_sums = T.sum(axis=1, keepdims=True)
    T = T / row_sums

    return T


def compute_stationary(T):
    """Compute stationary distribution via power iteration."""
    pi = np.ones(T.shape[0]) / T.shape[0]
    for _ in range(1000):
        pi_new = pi @ T
        if np.allclose(pi, pi_new, atol=1e-10):
            break
        pi = pi_new
    # Renormalize
    pi = pi / pi.sum()
    return pi


def compute_entropy(T):
    """Per-state transition entropy H(i) = -sum_j T[i,j] log T[i,j]."""
    H = np.zeros(T.shape[0])
    for i in range(T.shape[0]):
        row = T[i]
        row = row[row > 0]  # filter zeros for log
        H[i] = -np.sum(row * np.log(row))
    return H


def kl_divergence(T_p, T_q):
    """KL divergence per state: KL(T_p[i] || T_q[i]) for each state i.
    Returns per-state KL and mean KL."""
    kl_per_state = np.zeros(T_p.shape[0])
    for i in range(T_p.shape[0]):
        p = T_p[i]
        q = T_q[i]
        # Only compute where both p and q are positive
        mask = (p > 0) & (q > 0)
        if mask.any():
            kl_per_state[i] = np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))
    return kl_per_state


def build_all():
    """Build Markov chains for both encoders."""
    all_results = {}

    for enc_name in ENCODERS:
        print(f"\n{'='*60}")
        print(f"Encoder: {enc_name}")
        print(f"{'='*60}")

        X, meta, session_labels = load_data(enc_name)
        print(f"  Embeddings: {X.shape}")

        # Discretize
        X_pca, pca, km, states = discretize(X, meta)

        # Build trajectory sequences per session
        # Group by session_idx
        session_trajs = {}  # session_idx -> list of state indices
        for mi, m in enumerate(meta):
            si = m["session_idx"]
            if si not in session_trajs:
                session_trajs[si] = []
            if states[mi] >= 0:
                session_trajs[si].append(states[mi])

        # Separate by label
        fallacy_trajs = []
        valid_trajs = []
        fallacy_type_trajs = {}  # fallacy_type -> list of trajectories

        for si, traj in session_trajs.items():
            if si >= len(session_labels):
                continue
            label = session_labels[si]["label"]
            ft = session_labels[si].get("fallacy_type")

            traj_arr = np.array(traj)
            if len(traj_arr) < 2:
                continue  # need at least 2 states for a transition

            if label == "fallacy":
                fallacy_trajs.append(traj_arr)
                if ft:
                    if ft not in fallacy_type_trajs:
                        fallacy_type_trajs[ft] = []
                    fallacy_type_trajs[ft].append(traj_arr)
            else:
                valid_trajs.append(traj_arr)

        print(f"  Fallacy trajectories: {len(fallacy_trajs)}")
        print(f"  Valid trajectories: {len(valid_trajs)}")
        for ft, trajs in sorted(fallacy_type_trajs.items()):
            print(f"    {ft}: {len(trajs)}")

        # Build transition matrices
        print(f"\n  Building transition matrices...")
        T_fallacy = build_transition_matrix(fallacy_trajs, K_CLUSTERS)
        T_valid = build_transition_matrix(valid_trajs, K_CLUSTERS)

        # Per-fallacy-type transition matrices
        T_by_type = {}
        for ft, trajs in fallacy_type_trajs.items():
            if len(trajs) >= 3:
                T_by_type[ft] = build_transition_matrix(trajs, K_CLUSTERS)

        # Stationary distributions
        print(f"  Computing stationary distributions...")
        pi_fallacy = compute_stationary(T_fallacy)
        pi_valid = compute_stationary(T_valid)

        # Entropy
        H_fallacy = compute_entropy(T_fallacy)
        H_valid = compute_entropy(T_valid)

        # KL divergence (fallacy vs valid transitions)
        kl_fv = kl_divergence(T_fallacy, T_valid)
        kl_vf = kl_divergence(T_valid, T_fallacy)

        # Top discriminative states (highest KL)
        top_kl_states = np.argsort(kl_fv)[-20:][::-1]

        # Summary statistics
        mean_H_fallacy = np.mean(H_fallacy[H_fallacy > 0])
        mean_H_valid = np.mean(H_valid[H_valid > 0])
        mean_kl = np.mean(kl_fv[kl_fv > 0])

        print(f"\n  Mean transition entropy:")
        print(f"    Fallacy: {mean_H_fallacy:.4f}")
        print(f"    Valid:   {mean_H_valid:.4f}")
        print(f"    Ratio:   {mean_H_fallacy / mean_H_valid:.4f}")
        print(f"  Mean KL(fallacy || valid): {mean_kl:.6f}")
        print(f"  Top discriminative states: {top_kl_states[:10]}")

        # Save everything
        results = {
            "encoder": enc_name,
            "n_fallacy_trajs": len(fallacy_trajs),
            "n_valid_trajs": len(valid_trajs),
            "n_fallacy_types": len(T_by_type),
            "mean_H_fallacy": float(mean_H_fallacy),
            "mean_H_valid": float(mean_H_valid),
            "H_ratio": float(mean_H_fallacy / mean_H_valid) if mean_H_valid > 0 else 0,
            "mean_kl_fv": float(mean_kl),
        }

        # Save matrices
        np.save(os.path.join(RESULTS_DIR, f"T_fallacy_{enc_name}.npy"), T_fallacy)
        np.save(os.path.join(RESULTS_DIR, f"T_valid_{enc_name}.npy"), T_valid)
        np.save(os.path.join(RESULTS_DIR, f"pi_fallacy_{enc_name}.npy"), pi_fallacy)
        np.save(os.path.join(RESULTS_DIR, f"pi_valid_{enc_name}.npy"), pi_valid)
        np.save(os.path.join(RESULTS_DIR, f"H_fallacy_{enc_name}.npy"), H_fallacy)
        np.save(os.path.join(RESULTS_DIR, f"H_valid_{enc_name}.npy"), H_valid)
        np.save(os.path.join(RESULTS_DIR, f"kl_fv_{enc_name}.npy"), kl_fv)
        np.save(os.path.join(RESULTS_DIR, f"states_{enc_name}.npy"), states)
        np.save(os.path.join(RESULTS_DIR, f"pca_{enc_name}.npy"), X_pca)

        for ft, T_ft in T_by_type.items():
            ft_safe = ft.replace(" ", "_")
            np.save(os.path.join(RESULTS_DIR, f"T_{ft_safe}_{enc_name}.npy"), T_ft)

        all_results[enc_name] = results

    # Cross-encoder comparison
    if len(all_results) >= 2:
        encs = list(all_results.keys())
        print(f"\n{'='*60}")
        print("CROSS-ENCODER COMPARISON")
        print(f"{'='*60}")
        for metric in ["mean_H_fallacy", "mean_H_valid", "H_ratio", "mean_kl_fv"]:
            vals = [all_results[e].get(metric, "N/A") for e in encs]
            print(f"  {metric:<20} {encs[0]:<15} {encs[1]:<15}")

    # Save summary
    with open(os.path.join(RESULTS_DIR, "markov_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/markov_results.json")
    print("Run predict.py next.")


if __name__ == "__main__":
    build_all()