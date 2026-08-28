"""
Exp6 — Joint Encoder + JEPA Predictor (the pilot's learned models)

Builds and trains the two learned components of the pilot architecture
(DESIGN.md) TOGETHER, in one loop, so they coordinate:

  Encoder (Barlow Twins, 64-dim):  raw step embedding -> compact e_t
  Predictor (JEPA, multi-horizon): e_t -> predicted e_{t+k}  (k=1,2,3)

Joint training is the point: the encoder learns to produce shapes the
predictor can actually work with, and Barlow Twins + stop-gradient/EMA
target prevent collapse.

Input: the 90 exp3 sessions' step embeddings (both encoders).
Output: trained projector + predictor, plus evaluation.

Evaluation (the honest questions):
  1. Collapse check: does the projected 64-dim space have non-trivial variance?
  2. Delta separation: do the LEARNED 64-dim deltas separate fallacy/valid
     better than the RAW-embedding deltas from exp5 (nomic 0.970 / qwen3 1.000)?
     This is the key test — does learning a compact shape space help or hurt?
  3. Prediction test: does the JEPA predictor forecast e_{t+k} better than a
     mean-of-training baseline?
"""
import os, json, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "exp3_markov_trajectories", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

ENCODERS = ["nomic_embed_text", "qwen3_embedding"]
EMB_DIM = {"nomic_embed_text": 768, "qwen3_embedding": 4096}
PROJ_DIM = 64
HORIZONS = [1, 2, 3]
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading (reuse exp3 sessions)
# ---------------------------------------------------------------------------
def load_sessions(enc_name):
    """Return {session_idx: np.array (T, D) of step embeddings} + labels."""
    X = np.load(os.path.join(RESULTS_DIR, f"trajectory_embeddings_{enc_name}.npy"))
    with open(os.path.join(RESULTS_DIR, f"trajectory_meta_{enc_name}.json")) as f:
        meta = json.load(f)
    with open(os.path.join(RESULTS_DIR, "session_labels.json")) as f:
        labels = json.load(f)
    from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Projector(nn.Module):
    """Raw step embedding -> compact 64-dim e_t (Barlow Twins regularized)."""
    def __init__(self, in_dim, out_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, z):
        return self.net(z)


class Predictor(nn.Module):
    """JEPA forward model: e_t -> predicted e_{t+k} for k in HORIZONS.

    Shared trunk + one head per horizon. Multi-horizon forces the encoder
    to preserve both immediate transitions and longer-horizon shape.
    """
    def __init__(self, in_dim=PROJ_DIM, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({
            str(k): nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, in_dim))
            for k in HORIZONS
        })

    def forward(self, e_t):
        h = self.trunk(e_t)
        return {k: self.heads[str(k)](h) for k in HORIZONS}


def barlow_twins_loss(z1, z2, lambd=0.005):
    """Barlow Twins: cross-correlation matrix -> identity.

    z1, z2: (B, D) projected embeddings (predicted vs target).
    Diagonal -> 1 (invariance), off-diagonal -> 0 (redundancy reduction).
    """
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = (z1.T @ z2) / B                      # (D, D) cross-correlation
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].flatten().pow_(2).sum()
    return on_diag + lambd * off_diag


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_triples(sessions, labels):
    """Flatten all (e_t, e_{t+k}) pairs across sessions for a given encoder.

    Returns list of (z_t, z_{t+k}, k) raw-embedding pairs.
    """
    triples = []
    for si, embs in sessions.items():
        T = embs.shape[0]
        for t in range(T):
            for k in HORIZONS:
                if t + k < T:
                    triples.append((embs[t], embs[t + k], k))
    return triples


def train(enc_name, epochs=200, lr=1e-3, batch_size=64, ema_momentum=0.99):
    print(f"\n{'='*64}")
    print(f"Exp6 — Joint Encoder + JEPA: {enc_name}")
    print(f"{'='*64}")

    sessions, labels = load_sessions(enc_name)
    triples = build_triples(sessions, labels)
    print(f"  Sessions: {len(sessions)}  Triples: {len(triples)}")

    in_dim = EMB_DIM[enc_name]
    proj = Projector(in_dim)
    pred = Predictor()
    proj_ema = Projector(in_dim)          # target encoder (EMA, stop-gradient)
    proj_ema.load_state_dict(proj.state_dict())
    for p in proj_ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(list(proj.parameters()) + list(pred.parameters()),
                            lr=lr, weight_decay=1e-5)

    # Precompute tensors
    z_t = torch.tensor(np.array([tr[0] for tr in triples]), dtype=torch.float32)
    z_tk = torch.tensor(np.array([tr[1] for tr in triples]), dtype=torch.float32)
    k_idx = torch.tensor([tr[2] for tr in triples], dtype=torch.long)
    n = len(triples)

    loss_hist = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batch = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            zt, ztk, kk = z_t[idx], z_tk[idx], k_idx[idx]

            e_t = proj(zt)                       # online encoder
            with torch.no_grad():
                e_target = proj_ema(ztk)         # target encoder (EMA, stop-grad)

            preds = pred(e_t)                    # {k: predicted e_{t+k}}
            # prediction loss for the actual horizon of each sample
            loss_pred = 0.0
            for j, k in enumerate(HORIZONS):
                mask = (kk == k)
                if mask.any():
                    loss_pred += F.l1_loss(preds[k][mask], e_target[mask])
            # Barlow Twins on predicted vs target (all horizons pooled)
            all_pred = torch.cat([preds[k] for k in HORIZONS], dim=0)
            all_tgt = torch.cat([e_target for _ in HORIZONS], dim=0)
            loss_bt = barlow_twins_loss(all_pred, all_tgt)

            loss = loss_pred + 0.5 * loss_bt
            opt.zero_grad()
            loss.backward()
            opt.step()

            # EMA update of target encoder
            with torch.no_grad():
                for p, p_ema in zip(proj.parameters(), proj_ema.parameters()):
                    p_ema.data.mul_(ema_momentum).add_(p.data, alpha=1 - ema_momentum)

            total_loss += loss.item()
            n_batch += 1

        loss_hist.append(total_loss / max(n_batch, 1))
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}  loss={loss_hist[-1]:.4f}")

    # Collapse check: variance of projected embeddings
    with torch.no_grad():
        all_z = torch.tensor(np.array([embs for embs in sessions.values()]).reshape(-1, in_dim),
                             dtype=torch.float32)
        all_e = proj(all_z)
        var = all_e.var(0).mean().item()
        std = all_e.std(0).mean().item()
    print(f"  Collapse check: projected var={var:.4f} std={std:.4f} "
          f"({'COLLAPSED' if var < 1e-3 else 'OK'})")

    torch.save({"proj": proj.state_dict(), "pred": pred.state_dict(),
                "proj_ema": proj_ema.state_dict()},
               os.path.join(OUT_DIR, f"model_{enc_name}.pt"))
    return proj, pred, proj_ema, loss_hist


if __name__ == "__main__":
    for enc in ENCODERS:
        try:
            train(enc)
        except FileNotFoundError as e:
            print(f"\n  [{enc}] Files not found: {e}")
