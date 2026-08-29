"""Exp9 — train the A-space encoder from scratch (no frozen embedding model).

Losses: MLM (language) + Barlow Twins (format-crystallization pairs) +
prefix consistency (session-level fill-in-the-blank) [+ Run B: outcome head].

Two-run discipline (DESIGN.md): Run A = no outcome supervision; Run B = with.
Checkpoints: results/ckpt_run{A,B}.pt. Logs: results/train_run{A,B}.log.
"""
import os, sys, json, time, random, argparse, logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
EXP8 = os.path.join(HERE, "..", "exp8_harness_disjoint")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

from renderer import RENDERERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("exp9")

SEED = 42
PROJ_DIM = 8           # agentic-space dims (user spec / NOMENCLATURE A-space)
MAX_LEN = 256          # tokens per step
D_MODEL, N_LAYERS, N_HEADS = 256, 4, 4
PREFIX_MAX_STEPS = 16  # prefix-loss cost cap (documented in DESIGN)
BT_LAMBD = 0.005       # Barlow off-diagonal weight

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)


def load_sessions():
    path = os.path.join(EXP8, "results", "sessions_meta.jsonl")
    sessions = [json.loads(l) for l in open(path, encoding="utf-8")]
    log.info(f"loaded {len(sessions)} sessions from exp8 extract")
    return sessions


def get_tokenizer(sessions):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    out = os.path.join(RESULTS, "tokenizer.json")
    if os.path.exists(out):
        tok = Tokenizer.from_file(out)
        log.info(f"tokenizer cache hit ({tok.get_vocab_size()} vocab)")
        return tok
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[MASK]"],
        min_frequency=2)
    tok.train_from_iterator((st for s in sessions for st in s["steps"]), trainer)
    tok.save(out)
    log.info(f"trained BPE tokenizer -> {out} ({tok.get_vocab_size()} vocab)")
    return tok


class Block(nn.Module):
    def __init__(self, d, h, p=0.1, ff=None):
        super().__init__()
        ff = ff if ff is not None else 4 * d
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(ff, d), nn.Dropout(p))

    def forward(self, x, pad_mask=None):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         key_padding_mask=pad_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class ASpaceEncoder(nn.Module):
    """Token ids -> PROJ_DIM agentic-space vector. Own tokenizer, no pretrained."""

    def __init__(self, vocab, d=D_MODEL, layers=N_LAYERS, heads=N_HEADS, proj=PROJ_DIM):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.proj = nn.Linear(d, proj)
        self.mlm_head = nn.Linear(d, vocab)

    def forward(self, ids, pad_mask=None):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok(ids) + self.pos(pos)
        for b in self.blocks:
            x = b(x, pad_mask)
        x = self.ln(x)
        z = self.proj(x[:, 0])   # CLS-token state -> 8 dims
        return z, x

    def mlm_logits(self, states):
        return self.mlm_head(states)


def barlow_twins(z1, z2, lambd=BT_LAMBD):
    """Cross-correlation -> identity; same batch, two renderings."""
    N = z1.shape[0]
    z1n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = (z1n.T @ z2n) / N
    on_diag = (1 - c.diagonal()).pow(2).sum()
    d_ = c.shape[0]
    off = c - torch.eye(d_, device=c.device) * c.diagonal()
    off_diag = off.pow(2).sum()
    return on_diag + lambd * off_diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-bt", type=float, default=1.0)
    ap.add_argument("--lambda-prefix", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    log.addHandler(logging.FileHandler(os.path.join(RESULTS, f"train_run{args.run}.log"), mode="w"))

    device = args.device
    if device is None:
        # ROCm on Windows exposes the AMD iGPU as a CUDA device (torch 2.9+rocm)
        if torch.cuda.is_available():
            device = "cuda"
            log.info(f"using cuda/rocm device: {torch.cuda.get_device_name(0)}")
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                log.info("using torch_directml (iGPU)")
            except Exception:
                device = "cpu"
                log.info("using cpu")

    sessions_all = load_sessions()
    tok = get_tokenizer(sessions_all)
    vocab = tok.get_vocab_size()
    PAD_ID, CLS_ID, MASK_ID = (tok.token_to_id(t) for t in ("[PAD]", "[CLS]", "[MASK]"))

    sessions = [s for s in sessions_all
                if s.get("success") is not None and s.get("harness")]

    # encode canonical steps once; keep original texts for BT renderings
    sess = []
    for s in sessions:
        enc = tok.encode_batch([t[:20000] for t in s["steps"]])
        ids = [[CLS_ID] + e.ids[:MAX_LEN - 1] for e in enc]
        if len(ids) >= 2:
            sess.append({"ids": ids, "texts": s["steps"], "harness": s["harness"],
                         "benchmark": s["benchmark"], "y": 1 if s["success"] else 0})
    log.info(f"{len(sess)} usable sessions, "
             f"{len(set(x['harness'] for x in sess))} harnesses")

    model = ASpaceEncoder(vocab).to(device)
    outcome_head = nn.Linear(PROJ_DIM, 1).to(device)  # Run B only
    params = list(model.parameters()) + (list(outcome_head.parameters()) if args.run == "B" else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)

    render_names = list(RENDERERS.keys())
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    def pad_to_batch(seqs):
        L = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def encode_fresh(texts):
        enc = tok.encode_batch(texts)
        return [[CLS_ID] + e.ids[:MAX_LEN - 1] for e in enc]

    for epoch in range(args.epochs):
        model.train()
        for it in range(args.steps_per_epoch):
            b = rng.randint(0, len(sess), size=args.batch)

            # ---- MLM on canonical steps
            mlm_seqs = [sess[i]["ids"][rng.randint(0, len(sess[i]["ids"]))] for i in b]
            X = pad_to_batch(mlm_seqs).to(device)
            labels = X.clone()
            prob = torch.rand(X.shape, device=device)
            special = (X == PAD_ID) | (X == CLS_ID)
            to_mask = (prob < 0.15) & ~special
            labels[~to_mask] = -100
            rnd_tok = torch.randint(4, vocab, X.shape, device=device)
            rand_pick = torch.rand(X.shape, device=device) < 0.1
            X_in = torch.where(to_mask & rand_pick, rnd_tok, X)
            X_in = torch.where(to_mask & ~rand_pick, torch.full_like(X, MASK_ID), X_in)
            z, states = model(X_in, (X == PAD_ID))
            l_mlm = F.cross_entropy(model.mlm_logits(states).view(-1, vocab),
                                    labels.view(-1), ignore_index=-100)

            # ---- Barlow Twins: same step, ALL rendering pairs (attempt #2 fix:
            # one pair per batch left the space directionally collapsed; now
            # every format pair in C(5,2)=10 is pulled together each step, and
            # z is L2-normalized before the loss so direction must carry info)
            chosen = {i: rng.randint(0, len(sess[i]["texts"])) for i in b}
            pair_terms = []
            for r1 in range(len(render_names)):
                for r2 in range(r1 + 1, len(render_names)):
                    steps1 = [RENDERERS[render_names[r1]](sess[i]["texts"][chosen[i]]) for i in b]
                    steps2 = [RENDERERS[render_names[r2]](sess[i]["texts"][chosen[i]]) for i in b]
                    X1 = pad_to_batch(encode_fresh(steps1)).to(device)
                    X2 = pad_to_batch(encode_fresh(steps2)).to(device)
                    z1, _ = model(X1, (X1 == PAD_ID))
                    z2, _ = model(X2, (X2 == PAD_ID))
                    pair_terms.append(barlow_twins(F.normalize(z1, dim=-1),
                                                   F.normalize(z2, dim=-1)))
            l_bt = torch.stack(pair_terms).mean()

            # ---- prefix consistency + (Run B) outcome, ONE batched forward
            # (was 8 separate forwards — the CPU bottleneck)
            flat, bounds = [], {}
            for k, i in enumerate(b[:8]):  # cost cap
                ids_i = sess[i]["ids"][:PREFIX_MAX_STEPS]
                start = len(flat)
                flat.extend(ids_i)
                t = rng.randint(1, len(ids_i))
                bounds[k] = (start, t, len(ids_i), i)
            Xf = pad_to_batch(flat).to(device)
            zf, _ = model(Xf, (Xf == PAD_ID))
            pfx_terms, out_terms = [], []
            for k, (start, t, n, i) in bounds.items():
                zi = zf[start:start + n]
                pfx_terms.append(F.mse_loss(zi[:t].mean(0), zi.detach().mean(0)))
                if args.run == "B":
                    yb = torch.tensor(float(sess[i]["y"]), device=device)
                    out_terms.append(F.binary_cross_entropy_with_logits(
                        outcome_head(zi.mean(0).unsqueeze(0)).squeeze(), yb))
            l_pfx = torch.stack(pfx_terms).mean() if pfx_terms else torch.tensor(0.0, device=device)
            l_out = (torch.stack(out_terms).mean() if out_terms
                     else torch.tensor(0.0, device=device))

            loss = l_mlm + args.lambda_bt * l_bt + args.lambda_prefix * l_pfx + l_out
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            if it % 100 == 0:
                log.info(f"ep{epoch+1} it{it:4d} mlm={float(l_mlm):.3f} "
                         f"bt={float(l_bt):.3f} pfx={float(l_pfx):.3f} "
                         f"out={float(l_out):.3f} ({time.time()-t0:.0f}s)")

        ckpt = os.path.join(RESULTS, f"ckpt_run{args.run}.pt")
        torch.save({"model": model.state_dict(),
                    "outcome_head": outcome_head.state_dict() if args.run == "B" else None,
                    "args": vars(args), "vocab": vocab, "proj_dim": PROJ_DIM}, ckpt)
        log.info(f"epoch {epoch+1} checkpoint -> {ckpt}")

    log.info("training complete")


if __name__ == "__main__":
    main()