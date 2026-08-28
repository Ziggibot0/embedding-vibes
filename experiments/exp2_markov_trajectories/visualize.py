"""
Visualize Markov chain results: transition graphs, stationary distributions,
entropy heatmaps, and trajectory plots.

Outputs:
- figures/transition_entropy_{enc}.png — per-state entropy comparison
- figures/stationary_dist_{enc}.png — stationary distribution comparison
- figures/kl_divergence_{enc}.png — per-state KL divergence
- figures/trajectory_pca_{enc}.png — example trajectories in PCA space
- figures/fallacy_vs_valid_entropy_{enc}.png — entropy distributions
"""
import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

ENCODERS = ["nomic_embed_text", "qwen3_embedding"]


def plot_entropy_comparison(enc_name):
    """Per-state transition entropy: fallacy vs valid."""
    H_f = np.load(os.path.join(RESULTS_DIR, f"H_fallacy_{enc_name}.npy"))
    H_v = np.load(os.path.join(RESULTS_DIR, f"H_valid_{enc_name}.npy"))

    # Only plot states that have non-zero entropy
    mask = (H_f > 0) & (H_v > 0)
    H_f_active = H_f[mask]
    H_v_active = H_v[mask]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Scatter
    ax = axes[0]
    ax.scatter(H_v_active, H_f_active, alpha=0.3, s=10)
    lims = [0, max(H_v_active.max(), H_f_active.max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Valid entropy H(i)')
    ax.set_ylabel('Fallacy entropy H(i)')
    ax.set_title('Per-state transition entropy')
    ax.legend()

    # Histograms
    ax = axes[1]
    ax.hist(H_v_active, bins=40, alpha=0.6, label='Valid', density=True)
    ax.hist(H_f_active, bins=40, alpha=0.6, label='Fallacy', density=True)
    ax.set_xlabel('Transition entropy')
    ax.set_ylabel('Density')
    ax.set_title('Entropy distributions')
    ax.legend()

    # Difference
    ax = axes[2]
    diff = H_f_active - H_v_active
    ax.hist(diff, bins=40, alpha=0.7, color='purple')
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('H(fallacy) - H(valid)')
    ax.set_ylabel('Count')
    ax.set_title(f'Mean diff: {diff.mean():.4f}')

    plt.suptitle(f'Transition Entropy: {enc_name}', fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"entropy_comparison_{enc_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_kl_divergence(enc_name):
    """Per-state KL divergence between fallacy and valid transition distributions."""
    kl_fv = np.load(os.path.join(RESULTS_DIR, f"kl_fv_{enc_name}.npy"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of KL values
    ax = axes[0]
    active = kl_fv[kl_fv > 0]
    ax.hist(active, bins=50, alpha=0.7, color='teal')
    ax.set_xlabel('KL(T_fallacy[i] || T_valid[i])')
    ax.set_ylabel('Count')
    ax.set_title(f'Per-state KL divergence (mean={active.mean():.4f})')

    # Top-20 discriminative states
    ax = axes[1]
    top_k = 20
    top_states = np.argsort(kl_fv)[-top_k:][::-1]
    top_vals = kl_fv[top_states]
    ax.barh(range(top_k), top_vals, color='teal')
    ax.set_xlabel('KL divergence')
    ax.set_ylabel('State index')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([str(s) for s in top_states])
    ax.set_title(f'Top {top_k} discriminative states')

    plt.suptitle(f'KL Divergence: {enc_name}', fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"kl_divergence_{enc_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_stationary_comparison(enc_name):
    """Compare stationary distributions of fallacy vs valid Markov chains."""
    pi_f = np.load(os.path.join(RESULTS_DIR, f"pi_fallacy_{enc_name}.npy"))
    pi_v = np.load(os.path.join(RESULTS_DIR, f"pi_valid_{enc_name}.npy"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Scatter
    ax = axes[0]
    active = (pi_f > 1e-6) | (pi_v > 1e-6)
    ax.scatter(pi_v[active], pi_f[active], alpha=0.3, s=10)
    ax.set_xlabel('pi_valid(i)')
    ax.set_ylabel('pi_fallacy(i)')
    ax.set_title('Stationary distributions')

    # Top states where they differ most
    ax = axes[1]
    diff = pi_f - pi_v
    top_diff = np.argsort(np.abs(diff))[-20:][::-1]
    colors = ['red' if diff[s] > 0 else 'blue' for s in top_diff]
    ax.barh(range(20), diff[top_diff], color=colors)
    ax.set_xlabel('pi_fallacy - pi_valid')
    ax.set_ylabel('State')
    ax.set_yticks(range(20))
    ax.set_yticklabels([str(s) for s in top_diff])
    ax.set_title('Biggest stationary diffs (red=fallacy, blue=valid)')

    # KL between stationary distributions
    ax = axes[2]
    mask = (pi_f > 1e-10) & (pi_v > 1e-10)
    kl_pi = np.sum(pi_f[mask] * (np.log(pi_f[mask]) - np.log(pi_v[mask])))
    js_pi = 0.5 * np.sum(pi_f[mask] * (np.log(pi_f[mask]) - np.log(0.5*(pi_f[mask]+pi_v[mask])))) + \
            0.5 * np.sum(pi_v[mask] * (np.log(pi_v[mask]) - np.log(0.5*(pi_f[mask]+pi_v[mask]))))
    
    ax.text(0.5, 0.7, f'KL(pi_fallacy || pi_valid) = {kl_pi:.6f}', 
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.5, f'JS divergence = {js_pi:.6f}', 
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.set_title('Distribution divergence')
    ax.axis('off')

    plt.suptitle(f'Stationary Distributions: {enc_name}', fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"stationary_{enc_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_trajectories_pca(enc_name, max_sessions=30):
    """Plot example trajectories in PCA space, colored by label."""
    X_pca = np.load(os.path.join(RESULTS_DIR, f"pca_{enc_name}.npy"))
    states = np.load(os.path.join(RESULTS_DIR, f"states_{enc_name}.npy"))
    meta_path = os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")
    labels_path = os.path.join(RESULTS_DIR, "session_labels.json")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(labels_path) as f:
        session_labels = json.load(f)

    # Group by session
    session_states = {}
    for mi, m in enumerate(meta):
        si = m["session_idx"]
        if si not in session_states:
            session_states[si] = []
        session_states[si].append(mi)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectories
    n_fallacy = 0
    n_valid = 0

    for si, indices in sorted(session_states.items()):
        if si >= len(session_labels):
            continue
        label = session_labels[si]["label"]

        if label == "fallacy" and n_fallacy >= max_sessions:
            continue
        if label == "valid" and n_valid >= max_sessions:
            continue

        # Get PCA coordinates for this session's steps
        pts = X_pca[indices]
        if len(pts) < 2:
            continue

        color = 'red' if label == "fallacy" else 'blue'
        alpha = 0.4

        # Plot trajectory as line with dots
        ax.plot(pts[:, 0], pts[:, 1], '-o', color=color, alpha=alpha, 
                markersize=3, linewidth=0.8)
        
        # Arrow at end
        if len(pts) >= 2:
            dx = pts[-1, 0] - pts[-2, 0]
            dy = pts[-1, 1] - pts[-2, 1]
            ax.annotate('', xy=(pts[-1, 0], pts[-1, 1]),
                       xytext=(pts[-2, 0], pts[-2, 1]),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=1))

        if label == "fallacy":
            n_fallacy += 1
        else:
            n_valid += 1

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Trajectories in PCA Space ({enc_name})')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', marker='o', linestyle='-', label='Fallacy', alpha=0.6),
        Line2D([0], [0], color='blue', marker='o', linestyle='-', label='Valid', alpha=0.6),
    ]
    ax.legend(handles=legend_elements)

    path = os.path.join(FIGURES_DIR, f"trajectories_pca_{enc_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_transition_heatmap(enc_name, top_k=30):
    """Heatmap of transition probabilities for top-K most active states."""
    T_f = np.load(os.path.join(RESULTS_DIR, f"T_fallacy_{enc_name}.npy"))
    T_v = np.load(os.path.join(RESULTS_DIR, f"T_valid_{enc_name}.npy"))

    # Find most active states (highest total transition mass)
    activity = T_f.sum(axis=1) + T_v.sum(axis=1)
    top_states = np.argsort(activity)[-top_k:][::-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Fallacy transition matrix (top states)
    ax = axes[0]
    T_sub_f = T_f[np.ix_(top_states, top_states)]
    im = ax.imshow(T_sub_f, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('To state')
    ax.set_ylabel('From state')
    ax.set_title('Fallacy transitions')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Valid transition matrix (top states)
    ax = axes[1]
    T_sub_v = T_v[np.ix_(top_states, top_states)]
    im = ax.imshow(T_sub_v, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('To state')
    ax.set_ylabel('From state')
    ax.set_title('Valid transitions')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Difference
    ax = axes[2]
    diff = T_sub_f - T_sub_v
    im = ax.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    ax.set_xlabel('To state')
    ax.set_ylabel('From state')
    ax.set_title('Difference (fallacy - valid)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'Transition Heatmaps (top {top_k} states): {enc_name}', fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"transition_heatmap_{enc_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    for enc_name in ENCODERS:
        print(f"\nVisualizing: {enc_name}")
        try:
            plot_entropy_comparison(enc_name)
            plot_kl_divergence(enc_name)
            plot_stationary_comparison(enc_name)
            plot_trajectories_pca(enc_name)
            plot_transition_heatmap(enc_name)
        except FileNotFoundError as e:
            print(f"  Missing files: {e}")
            print(f"  Run build_mc.py first.")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()