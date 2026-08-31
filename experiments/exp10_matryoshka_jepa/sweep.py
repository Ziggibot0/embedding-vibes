"""Exp10 — the matryoshka dimension sweep (the russian-doll test).

For each matryoshka dim D in {64, 128, 256, 512, 768}:
  1. Truncate the cached v1.5 embeddings to D (already layer-normed).
  2. Relative-delta representation (np.diff over steps) -> PCA-50.
  3. 5-fold CV logistic AUC for fallacy/valid (exp6 protocol).
  4. Train the JEPA predictor on the D-dim deltas; report L1 vs mean baseline.

Gates (from DESIGN.md):
  G1: 64-dim truncated AUC > exp6 learned 64-dim (0.862 nomic / 0.864 qwen3)
  G2: 64-dim truncated AUC >= 0.90 (raw nomic deltas were 0.960)
  G3: AUC drop from 768 -> 64 is < 0.05 (elbow flat)
  G4: JEPA predictor beats mean-of-training by >= 50%
"""
import os, json, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# Reuse exp6's JEPA models so the predictor comparison is apples-to-apples
sys.path.insert(0, os.path.join(HERE, "..", "exp6_joint_jepa"))
from train import Projector, Predictor, barlow_twins_loss  # noqa: E402

DIMS = [64, 128, 256, 512, 768]
SEED = 42
HORIZONS = [1, 2, 3]
EPOCHS = 200
torch.manual_seed(SEED)
np.random.seed(SEED)


def load():
    E = np.load(os.path.join(DATA, "emb_v15_full.npy"))  # (N_steps, 768)
    with open(os.path.join(DATA, "sessions_v15.json")) as f:
        meta = json.load(f)
    # rebuild per-session arrays
    sessions, labels = {}, {}
    idx = 0
    for m in meta:
        n = m["n_steps"]
        sessions[m["session_idx"]] = E[idx : idx + n]
        labels[m["session_idx"]] = m["label"]
        idx += n
    return sessions, labels


def cv_auc(X, y):
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    aucs = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
    return float(aucs.mean()), float(aucs.std())


def delta_rep(sessions, D):
    """Truncate to D, then relative-delta (np.diff) -> PCA-50."""
    keys = sorted(sessions)
    X = np.array([np.diff(sessions[k][:, :D], axis=0).reshape(-1) for k in keys])
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]))
    return pca.fit_transform(X), keys


def jepa_prediction(sessions, D, keys):
    """Train JEPA predictor on D-dim deltas; return (pred_l1, mean_l1)."""
    # Build (e_t, e_{t+k}) triples from the D-dim step embeddings
    triples = []
    for k in keys:
        e = sessions[k][:, :D]
        T = e.shape[0]
        for t in range(T):
            for h in HORIZONS:
                if t + h < T:
                    triples.append((e[t], e[t + h], h))
    if len(triples) < 20:
        return None, None
    z_t = torch.tensor(np.array([tr[0] for tr in triples]), dtype=torch.float32)
    z_tk = torch.tensor(np.array([tr[1] for tr in triples]), dtype=torch.float32)
    k_idx = torch.tensor([tr[2] for tr in triples], dtype=torch.long)

    # exp10: matryoshka truncation IS the projection — no learned Projector.
    # The predictor operates directly on the truncated D-dim vectors.
    pred = Predictor(D)
    opt = torch.optim.AdamW(pred.parameters(), lr=1e-3, weight_decay=1e-5)
    n = len(triples)
    bs = 64
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            zt, ztk, kk = z_t[idx], z_tk[idx], k_idx[idx]
            e_t = zt                      # truncated D-dim (no projection)
            e_target = ztk                # truncated D-dim target
            preds = pred(e_t)
            loss_pred = 0.0
            for j, h in enumerate(HORIZONS):
                mask = (kk == h)
                if mask.any():
                    loss_pred += torch.nn.functional.l1_loss(preds[h][mask], e_target[mask])
            all_pred = torch.cat([preds[h] for h in HORIZONS], dim=0)
            all_tgt = torch.cat([e_target for _ in HORIZONS], dim=0)
            loss = loss_pred + 0.5 * barlow_twins_loss(all_pred, all_tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Measure L1 vs mean-of-training baseline
    with torch.no_grad():
        all_e = np.concatenate([sessions[k][:, :D] for k in keys], axis=0)
        mean_e = all_e.mean(0)
        pred_errs, mean_errs = [], []
        for k in keys:
            e = sessions[k][:, :D]
            T = e.shape[0]
            for t in range(T):
                for h in HORIZONS:
                    if t + h < T:
                        e_t = torch.tensor(e[t], dtype=torch.float32).unsqueeze(0)
                        e_tk = e[t + h]
                        e_pred = pred(e_t)[h].numpy()[0]
                        pred_errs.append(np.abs(e_pred - e_tk).mean())
                        mean_errs.append(np.abs(mean_e - e_tk).mean())
    return float(np.mean(pred_errs)), float(np.mean(mean_errs))


def main():
    sessions, labels = load()
    keys = sorted(sessions)
    y = np.array([1 if labels[k] == "fallacy" else 0 for k in keys])
    print(f"Sessions: {len(keys)}  fallacy={y.sum()} valid={(~y.astype(bool)).sum()}")

    results = {}
    for D in DIMS:
        X, _ = delta_rep(sessions, D)
        auc, auc_std = cv_auc(X, y)
        pred_l1, mean_l1 = jepa_prediction(sessions, D, keys)
        pct = (pred_l1 / mean_l1 - 1) * 100 if (pred_l1 and mean_l1) else None
        results[D] = {
            "auc": auc, "auc_std": auc_std,
            "pred_l1": pred_l1, "mean_l1": mean_l1,
            "pred_beats_mean_pct": pct,
        }
        print(f"  D={D:4d}  AUC={auc:.3f}±{auc_std:.3f}  "
              f"JEPA L1={pred_l1:.4f} vs mean={mean_l1:.4f} "
              f"({pct:+.1f}% if pct is not None else 'n/a')")

    # Gates
    g1 = results[64]["auc"] > 0.862  # exp6 learned 64-dim (nomic)
    g2 = results[64]["auc"] >= 0.90
    g3 = (results[768]["auc"] - results[64]["auc"]) < 0.05
    g4 = (results[64]["pred_beats_mean_pct"] is not None
          and results[64]["pred_beats_mean_pct"] <= -50.0)
    verdict = {
        "G1_truncation_beats_learned_crush": g1,
        "G2_64dim_recovers_toward_raw": g2,
        "G3_elbow_flat_to_64": g3,
        "G4_jepa_beats_mean_50pct": g4,
    }
    print("\n  GATES:")
    for k, v in verdict.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")

    out = os.path.join(OUT, "exp10_results.json")
    with open(out, "w") as f:
        json.dump({"results": results, "gates": verdict}, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
