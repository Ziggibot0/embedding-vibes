"""Exp10.1 — Outcome crystallization: Barlow Twins on same-outcome session pairs.

Train a 64-dim alignment so two sessions with the SAME outcome (both success
or both failure), however different their paths, map to the same point.

Leakage discipline (pre-registered): pairs are formed ONLY within the training
split. In LOHO, the BT encoder trains on 4 harnesses' sessions (pairs from
those 4 only) and is evaluated on the 5th harness's sessions, which never
contributed a pair. If the aligned space transfers, it is real crystallization.

Eval:
  - LOHO (leave-one-harness-out): BT on 4 harnesses, linear-probe outcome on 5th.
  - In-format 5-fold.
  - Compare BT-aligned vs raw matryoshka 64-dim (does alignment add value?).

Gates (DESIGN.md):
  G1: LOHO mean AUC > 0.60 (tags-only bar from exp7a)
  G2: BT-aligned LOHO AUC > raw matryoshka 64-dim LOHO AUC
  G3: per-dim variance of aligned 64-dim >= 0.2 (no collapse)
  G4: LOHO (train 4 harnesses, test 5th) — harness-invariant outcome crystallization
"""
import os, json, sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

DIM = 64
SEED = 42
EPOCHS = 300
LR = 1e-3
torch.manual_seed(SEED)
np.random.seed(SEED)


def load():
    V = np.load(os.path.join(DATA, "sessions_v15_64d.npy"))
    with open(os.path.join(DATA, "sessions_v15_64d.json")) as f:
        meta = json.load(f)
    return V, meta


class Aligner(nn.Module):
    """64 -> 64 Barlow-Twins alignment. Matryoshka truncation is the base rep;
    this aligns so same-outcome sessions cluster (no lossy projection)."""
    def __init__(self, d=DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d),
        )

    def forward(self, z):
        return self.net(z)


def barlow_twins(z1, z2, lambd=0.005):
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = (z1.T @ z2) / B
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].flatten().pow_(2).sum()
    return on_diag + lambd * off_diag


def make_pairs(V, y, idx):
    """Same-outcome pairs from sessions in idx (training split only)."""
    pos = [i for i in idx if y[i] == 1]
    neg = [i for i in idx if y[i] == 0]
    rng = np.random.RandomState(SEED)
    pairs = []
    for i in idx:
        pool = pos if y[i] == 1 else neg
        if len(pool) < 2:
            continue
        j = rng.choice([p for p in pool if p != i])
        pairs.append((i, j))
    return pairs


def train_aligner(V, y, train_idx):
    pairs = make_pairs(V, y, train_idx)
    if len(pairs) < 20:
        return None
    model = Aligner(DIM)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    z = torch.tensor(V, dtype=torch.float32)
    p1 = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    p2 = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    n = len(pairs)
    bs = 64
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idxb = perm[i : i + bs]
            a, b = p1[idxb], p2[idxb]
            za, zb = model(z[a]), model(z[b])
            loss = barlow_twins(za, zb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def cv_auc(X, y):
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    aucs = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
    return float(aucs.mean()), float(aucs.std())


def main():
    V, meta = load()
    y = np.array([m["success"] for m in meta])
    harnesses = sorted(set(m["harness"] for m in meta))
    print(f"Sessions: {len(V)}  success={y.sum()} failure={(~y.astype(bool)).sum()}")
    print(f"Harnesses: {harnesses}")

    # ---- Raw matryoshka 64-dim baseline (no alignment) ----
    raw_auc, raw_std = cv_auc(V, y)
    print(f"\n  RAW matryoshka 64-dim in-format AUC: {raw_auc:.3f}±{raw_std:.3f}")

    # ---- LOHO: train BT on 4 harnesses, test on 5th ----
    loho_aligned, loho_raw = [], []
    for held in harnesses:
        test_idx = np.array([i for i, m in enumerate(meta) if m["harness"] == held])
        train_idx = np.array([i for i, m in enumerate(meta) if m["harness"] != held])
        if len(test_idx) < 20 or len(set(y[train_idx])) < 2:
            continue
        model = train_aligner(V, y, train_idx)
        if model is None:
            continue
        with torch.no_grad():
            Va = model(torch.tensor(V, dtype=torch.float32)).numpy()
        # aligned probe on held-out harness
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(StandardScaler().fit_transform(Va[train_idx]), y[train_idx])
        a_auc = float(roc_auc_score(
            y[test_idx], clf.predict_proba(StandardScaler().fit_transform(Va[test_idx]))[:, 1]))
        # raw probe on held-out harness
        clf2 = LogisticRegression(max_iter=3000, C=1.0)
        clf2.fit(StandardScaler().fit_transform(V[train_idx]), y[train_idx])
        r_auc = float(roc_auc_score(
            y[test_idx], clf2.predict_proba(StandardScaler().fit_transform(V[test_idx]))[:, 1]))
        loho_aligned.append(a_auc)
        loho_raw.append(r_auc)
        print(f"  LOHO held={held:24s} n_test={len(test_idx):4d} "
              f"aligned={a_auc:.3f} raw={r_auc:.3f}")

    mean_aligned = float(np.mean(loho_aligned)) if loho_aligned else None
    mean_raw = float(np.mean(loho_raw)) if loho_raw else None
    print(f"\n  LOHO mean: aligned={mean_aligned:.3f} raw={mean_raw:.3f}")

    # ---- Collapse check on aligned space (train on all, check variance) ----
    model_all = train_aligner(V, y, np.arange(len(V)))
    with torch.no_grad():
        Va_all = model_all(torch.tensor(V, dtype=torch.float32)).numpy()
    min_var = float(Va_all.var(0).min())
    print(f"  Aligned per-dim min variance: {min_var:.3f}")

    # ---- Gates ----
    g1 = mean_aligned is not None and mean_aligned > 0.60
    g2 = mean_aligned is not None and mean_raw is not None and mean_aligned > mean_raw
    g3 = min_var >= 0.2
    g4 = mean_aligned is not None and mean_aligned > 0.60  # LOHO is the harness-invariance test
    verdict = {
        "G1_outcome_crystallizes_gt_0.60": g1,
        "G2_bt_beats_raw": g2,
        "G3_no_collapse": g3,
        "G4_harness_invariant": g4,
    }
    print("\n  GATES:")
    for k, v in verdict.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")

    out = os.path.join(OUT, "exp10_1_results.json")
    with open(out, "w") as f:
        json.dump({
            "raw_informat_auc": raw_auc,
            "loho_aligned": loho_aligned, "loho_raw": loho_raw,
            "loho_aligned_mean": mean_aligned, "loho_raw_mean": mean_raw,
            "aligned_min_var": min_var, "gates": verdict,
        }, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
