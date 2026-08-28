"""
Exp6 — train the FROM-SCRATCH text encoder + JEPA predictor jointly.

The encoder takes TEXT (token ids), not embeddings. It is trained from
scratch with three objectives in one loss:

  L = L_mlm + λ_bt * L_barlow + β * L_jepa

  L_mlm    : masked language modeling — gives the encoder language grounding
             (this is what makes it a text encoder at all)
  L_barlow : Barlow Twins on the step embedding — redundancy reduction,
             "pack info per dim" (the user's 64-dim idea)
  L_jepa   : multi-horizon prediction e_{t+k} = f(e_t) — shapes the space for
             reasoning (the pilot's forward model)

Joint training: the encoder learns to produce step embeddings the JEPA
predictor can actually forecast, and Barlow Twins + stop-gradient/EMA target
prevent collapse.

Data: sessions.jsonl (multi-harness agentic trajectories, text steps).
"""
import os, json, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

DATA = os.path.join(os.path.dirname(__file__), "data", "sessions.jsonl")
TOK_PATH = os.path.join(os.path.dirname(__file__), "data", "tokenizer.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class DMLSafeLayerNorm(nn.Module):
    """LayerNorm implemented with basic ops (DML doesn't support torch.layer_norm).

    Uses mean/var over the last dim, which torch-directml does support.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


# ---------------------------------------------------------------------------
# Model: small transformer encoder (from scratch)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block using DML-safe LayerNorm."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = DMLSafeLayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.ln2 = DMLSafeLayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        # pre-norm attention
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + attn_out
        # pre-norm FFN
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class TextEncoder(nn.Module):
    """Small transformer encoder: token ids -> step embedding e_t (PROJ_DIM).

    This IS the embedding model — it consumes raw text and produces a
    task-trained embedding. Not downstream of any frozen embedding model.

    NOTE: nn.Embedding and torch.layer_norm are not supported on
    torch-directml (DML). We keep the token embedding on CPU (a cheap gather)
    and use DMLSafeLayerNorm for the transformer layers.
    """
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4,
                 d_ff=512, proj_dim=64, max_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.tok_emb = nn.Embedding(vocab_size, d_model)  # stays on CPU
        self.pos_emb = nn.Embedding(max_len, d_model)     # stays on CPU
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = DMLSafeLayerNorm(d_model)
        # MLM head (language grounding)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        # Projection to compact step embedding
        self.proj = nn.Sequential(
            nn.Linear(d_model, 128), DMLSafeLayerNorm(128), nn.GELU(),
            nn.Linear(128, proj_dim), DMLSafeLayerNorm(proj_dim),
        )

    def _embed(self, tok_ids):
        """Token lookup on CPU, then move dense vectors to compute device."""
        cpu = tok_ids.cpu()
        x = self.tok_emb(cpu) + self.pos_emb(
            torch.arange(cpu.shape[1], device=cpu.device))
        return x.to(tok_ids.device)

    def forward(self, tok_ids, mask=None):
        B, T = tok_ids.shape
        x = self._embed(tok_ids)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=(mask == 0) if mask is not None else None)
        x = self.norm(x)
        # step embedding = mean-pool over tokens (masked)
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        e_t = self.proj(x)
        return e_t, x  # e_t (B, proj_dim), pooled hidden (B, d_model)

    def mlm_logits(self, tok_ids, mask=None):
        """Full forward for MLM: returns logits over vocab at each position."""
        B, T = tok_ids.shape
        x = self._embed(tok_ids)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=(mask == 0) if mask is not None else None)
        x = self.norm(x)
        return self.mlm_head(x)


class JEPAPredictor(nn.Module):
    """Multi-horizon forward model: e_t -> predicted e_{t+k}."""
    def __init__(self, in_dim=64, hidden=128, horizons=(1, 2, 3)):
        super().__init__()
        self.horizons = horizons
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), DMLSafeLayerNorm(hidden), nn.GELU())
        self.heads = nn.ModuleDict({
            str(k): nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, in_dim))
            for k in horizons
        })

    def forward(self, e_t):
        h = self.trunk(e_t)
        return {k: self.heads[str(k)](h) for k in self.horizons}


def barlow_twins_loss(z1, z2, lambd=0.005):
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = (z1.T @ z2) / B
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].flatten().pow_(2).sum()
    return on_diag + lambd * off_diag


# ---------------------------------------------------------------------------
# Data loading + masking
# ---------------------------------------------------------------------------
def load_sessions(path=DATA):
    sessions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line))
    return sessions


def mask_tokens(tok_ids, mask_token_id, pad_token_id, p=0.15):
    """Standard BERT-style masking. Returns (masked_ids, labels)."""
    labels = tok_ids.clone()
    prob = torch.rand(tok_ids.shape, device=tok_ids.device)
    mask = (prob < p) & (tok_ids != pad_token_id)
    labels[~mask] = -100
    tok_ids = tok_ids.clone()
    tok_ids[mask] = mask_token_id
    return tok_ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lambda-bt", type=float, default=0.5)
    ap.add_argument("--beta-jepa", type=float, default=0.3)
    ap.add_argument("--data", type=str, default=DATA)
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu or dml (AMD iGPU via torch-directml)")
    args = ap.parse_args()

    # Device selection
    if args.device == "dml":
        import torch_directml
        device = torch_directml.device()
        print(f"Using DML device: {torch_directml.device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    tok = Tokenizer.from_file(TOK_PATH)
    vocab = tok.get_vocab()
    pad_id = vocab["[PAD]"]
    mask_id = vocab["[MASK]"]
    vocab_size = len(vocab)

    sessions = load_sessions(args.data)
    print(f"Loaded {len(sessions)} sessions")

    # Tokenize all steps
    tokenized = []  # list of (session_idx, step_idx, token_ids)
    for si, s in enumerate(sessions):
        for ti, step in enumerate(s["steps"]):
            ids = tok.encode(step).ids[: args.max_len]
            if len(ids) >= 2:
                tokenized.append((si, ti, ids))
    print(f"Tokenized {len(tokenized)} steps")

    # Build session step-embedding index for JEPA pairs
    # session_steps[si] = list of (step_idx, token_ids)
    from collections import defaultdict
    session_steps = defaultdict(list)
    for si, ti, ids in tokenized:
        session_steps[si].append((ti, ids))

    # PRECOMPUTE JEPA pairs: (source_step_idx, target_step_idx, k)
    # This avoids the quadratic per-batch scan.
    jepa_pairs = []  # (si, ti_src, ti_tgt, k)
    for si, items in session_steps.items():
        items.sort()
        step_ids = [ti for ti, _ in items]
        for ti, _ in items:
            for k in (1, 2, 3):
                if ti + k in step_ids:
                    jepa_pairs.append((si, ti, ti + k, k))
    print(f"JEPA pairs: {len(jepa_pairs)}")

    # Global step index: (si, ti) -> position in `tokenized` list
    step_global = {}
    for gi, (si, ti, _) in enumerate(tokenized):
        step_global[(si, ti)] = gi
    # Vectorized pair arrays: for each pair, the global index of src and tgt
    pair_src = np.array([step_global[(si, ti_src)] for si, ti_src, _, _ in jepa_pairs])
    pair_tgt = np.array([step_global[(si, ti_tgt)] for si, _, ti_tgt, _ in jepa_pairs])
    pair_k = np.array([k for _, _, _, k in jepa_pairs])

    enc = TextEncoder(vocab_size, proj_dim=args.proj_dim, max_len=args.max_len).to(device)
    pred = JEPAPredictor(in_dim=args.proj_dim).to(device)
    enc_ema = TextEncoder(vocab_size, proj_dim=args.proj_dim, max_len=args.max_len).to(device)
    enc_ema.load_state_dict(enc.state_dict())
    for p in enc_ema.parameters():
        p.requires_grad_(False)
    # Keep token/pos embeddings on CPU (DML doesn't support nn.Embedding)
    if args.device == "dml":
        for m in (enc, enc_ema):
            m.tok_emb = m.tok_emb.cpu()
            m.pos_emb = m.pos_emb.cpu()

    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=args.lr, weight_decay=1e-5)
    # On DML, embeddings live on CPU -> separate optimizer for them
    if args.device == "dml":
        dml_params = [p for p in list(enc.parameters()) + list(pred.parameters())
                      if p.device.type != "cpu"]
        cpu_params = [p for p in list(enc.parameters()) + list(pred.parameters())
                      if p.device.type == "cpu"]
        opt = torch.optim.AdamW(dml_params, lr=args.lr, weight_decay=1e-5)
        opt_cpu = torch.optim.AdamW(cpu_params, lr=args.lr, weight_decay=1e-5)
    else:
        opt_cpu = None

    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in pred.parameters())
    print(f"Trainable params: {n_params/1e6:.2f}M")

    def pad_batch(seqs):
        maxlen = max(len(s) for s in seqs)
        out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    for epoch in range(args.epochs):
        np.random.shuffle(tokenized)
        total = 0.0
        n_batch = 0
        # MLM batches over individual steps
        for i in range(0, len(tokenized), args.batch_size):
            batch = tokenized[i:i + args.batch_size]
            seqs = [b[2] for b in batch]
            tok_ids = pad_batch(seqs).to(device)
            mask = (tok_ids != pad_id).float()
            masked, labels = mask_tokens(tok_ids, mask_id, pad_id)
            if args.device == "dml":
                assert tok_ids.device.type == "privateuseone", f"tok_ids {tok_ids.device}"
                assert mask.device.type == "privateuseone", f"mask {mask.device}"
                assert masked.device.type == "privateuseone", f"masked {masked.device}"
                assert labels.device.type == "privateuseone", f"labels {labels.device}"
                assert enc.blocks[0].ln1.weight.device.type == "privateuseone", \
                    f"ln1 {enc.blocks[0].ln1.weight.device}"
                assert enc_ema.blocks[0].ln1.weight.device.type == "privateuseone", \
                    f"ema ln1 {enc_ema.blocks[0].ln1.weight.device}"

            # MLM loss
            logits = enc.mlm_logits(masked, mask)
            loss_mlm = F.cross_entropy(logits.reshape(-1, vocab_size),
                                       labels.reshape(-1), ignore_index=-100)

            # Step embeddings for JEPA (online + EMA target)
            e_t, _ = enc(tok_ids, mask)
            with torch.no_grad():
                e_target, _ = enc_ema(tok_ids, mask)

            # Barlow Twins on step embeddings
            loss_bt = barlow_twins_loss(e_t, e_target)

            # JEPA: vectorized over pairs whose src is in this batch.
            # Map global step index -> local row in this batch.
            batch_global = np.array([step_global[(b[0], b[1])] for b in batch])
            local_of = {int(g): j for j, g in enumerate(batch_global)}
            # pairs where src is in this batch
            in_batch = np.isin(pair_src, batch_global)
            if in_batch.any():
                src_g = pair_src[in_batch]
                tgt_g = pair_tgt[in_batch]
                ks = pair_k[in_batch]
                # keep pairs whose tgt is ALSO in this batch
                tgt_in = np.isin(tgt_g, batch_global)
                src_g = src_g[tgt_in]
                tgt_g = tgt_g[tgt_in]
                ks = ks[tgt_in]
                if len(src_g) > 0:
                    src_local = np.array([local_of[int(g)] for g in src_g])
                    tgt_local = np.array([local_of[int(g)] for g in tgt_g])
                    e_src = e_t[src_local]
                    e_tgt = e_target[tgt_local]
                    # predict per-horizon
                    loss_jepa = 0.0
                    for k in pred.horizons:
                        m = ks == k
                        if m.any():
                            e_pred = pred(e_src[m])[k]
                            loss_jepa += F.l1_loss(e_pred, e_tgt[m])
                    loss_jepa = loss_jepa / len(pred.horizons)
                else:
                    loss_jepa = torch.tensor(0.0, device=device)
            else:
                loss_jepa = torch.tensor(0.0, device=device)

            loss = loss_mlm + args.lambda_bt * loss_bt + args.beta_jepa * loss_jepa
            opt.zero_grad()
            if opt_cpu is not None:
                opt_cpu.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            if opt_cpu is not None:
                opt_cpu.step()

            # EMA update
            with torch.no_grad():
                for p, p_ema in zip(enc.parameters(), enc_ema.parameters()):
                    p_ema.data.mul_(0.99).add_(p.data, alpha=0.01)
            if args.device == "dml":
                bad = [n for n, p in enc_ema.named_parameters()
                       if p.device.type != "privateuseone" and "tok_emb" not in n and "pos_emb" not in n]
                if bad:
                    raise RuntimeError(f"EMA moved params to CPU: {bad[:5]}")

            total += loss.item()
            n_batch += 1

        print(f"epoch {epoch+1:3d}  loss={total/max(n_batch,1):.4f}")

    # Collapse check
    with torch.no_grad():
        all_e = []
        for si, items in session_steps.items():
            for ti, ids in items[:1]:
                t = torch.tensor([ids], dtype=torch.long, device=device)
                m = (t != pad_id).float()
                e, _ = enc(t, m)
                all_e.append(e)
        all_e = torch.cat(all_e)
        var = all_e.var(0).mean().item()
    print(f"Collapse check: projected var={var:.4f} ({'OK' if var > 1e-3 else 'COLLAPSED'})")

    torch.save({"enc": enc.state_dict(), "pred": pred.state_dict(),
                "enc_ema": enc_ema.state_dict(), "args": vars(args)},
               os.path.join(OUT_DIR, f"model_proj{args.proj_dim}.pt"))
    print(f"Saved model to {OUT_DIR}/model_proj{args.proj_dim}.pt")


if __name__ == "__main__":
    main()
