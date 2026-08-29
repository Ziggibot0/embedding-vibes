"""Exp9 eval — the harness-disjoint exam for the A-space encoder.

Loads a checkpoint, embeds exp8's 3,000 labeled sessions into 8 dims (mean of
step vectors), fits LogisticRegression per split (same protocol as exp8), and
checks the pre-registered gates G1-G4 (DESIGN.md).
"""
import os, sys, json, argparse, logging
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP8 = os.path.join(HERE, "..", "exp8_harness_disjoint")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

from train import ASpaceEncoder, load_sessions, get_tokenizer, MAX_LEN  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("exp9eval")

SEED = 42
CLS_KEY = "[CLS]"
PAD_KEY = "[PAD]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], required=True)
    ap.add_argument("--device", default="cpu", help="eval is light; cpu fine")
    args = ap.parse_args()

    sessions_all = load_sessions()
    tok = get_tokenizer(sessions_all)
    vocab = tok.get_vocab_size()
    CLS_ID, PAD_ID = tok.token_to_id(CLS_KEY), tok.token_to_id(PAD_KEY)

    ck = torch.load(os.path.join(RESULTS, f"ckpt_run{args.run}.pt"),
                    map_location=args.device, weights_only=False)
    model = ASpaceEncoder(vocab).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    log.info(f"loaded ckpt_run{args.run} (proj_dim={ck['proj_dim']})")

    sessions = [s for s in sessions_all
                if s.get("success") is not None and s.get("harness")]

    # ---- embed every session into 8 dims (batched, no grad)
    Z, y, groups, bench = [], [], [], []
    B = 64
    with torch.no_grad():
        buf_ids, buf_meta = [], []
        def flush():
            if not buf_ids:
                return
            L = max(len(x) for x in buf_ids)
            X = torch.zeros(len(buf_ids), L, dtype=torch.long)
            for r, s in enumerate(buf_ids):
                X[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            pad = (X == PAD_ID)
            z, _ = model(X.to(args.device), pad.to(args.device))
            for r, meta in enumerate(buf_meta):
                Z.append(z[r].cpu().numpy())
                y.append(meta["y"]); groups.append(meta["harness"])
                bench.append(meta["benchmark"])
            buf_ids.clear(); buf_meta.clear()

        for s in sessions:
            enc = tok.encode_batch([t[:20000] for t in s["steps"]])
            ids = [[CLS_ID] + e.ids[:MAX_LEN - 1] for e in enc]
            if len(ids) < 2:
                continue
            # embed in step-chunks to bound memory, then mean -> session vector
            zs = []
            for k in range(0, len(ids), B):
                chunk = ids[k:k + B]
                L = max(len(x) for x in chunk)
                X = torch.zeros(len(chunk), L, dtype=torch.long)
                for r, x in enumerate(chunk):
                    X[r, :len(x)] = torch.tensor(x, dtype=torch.long)
                z, _ = model(X.to(args.device), (X == PAD_ID).to(args.device))
                zs.append(z.cpu())
            zi = torch.cat(zs).mean(0)
            Z.append(zi.numpy())
            y.append(1 if s["success"] else 0)
            groups.append(s["harness"]); bench.append(s["benchmark"])

    Z = np.array(Z); y = np.array(y)
    log.info(f"embedded {len(Z)} sessions -> Z {Z.shape}; "
             f"harnesses: {dict((lambda c: {k: c[k] for k in c})(__import__('collections').Counter(groups)))}")

    # ---- collapse check (G4)
    zvar = Z.var(0)
    log.info(f"per-dim variance: {np.round(zvar, 4)}")

    # ---- same eval protocol as exp8
    def fit_eval(tr, te):
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Z[tr]), y[tr])
        return roc_auc_score(y[te], clf.predict_proba(sc.transform(Z[te]))[:, 1])

    res = {"run": args.run, "n": int(len(Z)), "per_dim_var": zvar.tolist()}

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    aucs = [fit_eval(tr, te) for tr, te in skf.split(Z, y)]
    res["in_format_auc"] = float(np.mean(aucs))
    log.info(f"in_format 5-fold AUC: {res['in_format_auc']:.3f}")

    ga = np.array(groups)
    loho = {}
    for g in sorted(set(ga)):
        te = np.where(ga == g)[0]; tr = np.where(ga != g)[0]
        if len(te) < 150 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            log.info(f"  loho skip {g} (n={len(te)})")
            continue
        loho[g] = fit_eval(tr, te)
        log.info(f"  loho {g}: AUC {loho[g]:.3f} (n={len(te)})")
    res["loho"] = loho
    res["loho_mean"] = float(np.mean(list(loho.values()))) if loho else None

    ba = np.array(bench)
    lobo = {}
    for g in sorted(set(ba)):
        te = np.where(ba == g)[0]; tr = np.where(ba != g)[0]
        if len(te) < 150 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            log.info(f"  lobo skip {g} (n={len(te)})")
            continue
        lobo[g] = fit_eval(tr, te)
        log.info(f"  lobo {g}: AUC {lobo[g]:.3f} (n={len(te)})")
    res["lobo"] = lobo
    res["lobo_mean"] = float(np.mean(list(lobo.values()))) if lobo else None

    # ---- gates (pre-registered in DESIGN.md)
    gates = {}
    gates["G1_signal_exists"] = res["loho_mean"] is not None and res["loho_mean"] >= 0.60
    gates["G4_no_collapse"] = bool((zvar >= 0.2).all())
    # G2/G3 compare against exp8 numbers, loaded if present
    try:
        exp8 = json.load(open(os.path.join(EXP8, "results", "exp8_results.json")))
        def mm(split, feat):
            per = exp8.get(split, {}).get(feat, {})
            vals = [v["auc"] for v in per.values() if isinstance(v, dict) and "auc" in v]
            return float(np.mean(vals)) if vals else None
        tf_drop = (exp8["in_format"]["tfidf"]["mean_auc"] - mm("loho", "tfidf")) if mm("loho", "tfidf") else None
        our_drop = res["in_format_auc"] - res["loho_mean"] if res["loho_mean"] else None
        if tf_drop is not None and our_drop is not None:
            gates["G2_transfers_like_lexical"] = bool(our_drop <= tf_drop + 0.02)
        pca8_loho = mm("loho", "static_pca8")
        if pca8_loho is not None and res["loho_mean"] is not None:
            gates["G3_dream"] = bool(res["loho_mean"] >= pca8_loho - 0.05)
    except FileNotFoundError:
        log.info("exp8 results not present yet — G2/G3 deferred")

    res["gates"] = gates
    out = os.path.join(RESULTS, f"eval_run{args.run}.json")
    json.dump(res, open(out, "w"), indent=2, default=str)
    log.info(f"gates: {json.dumps(gates)}")
    log.info(f"written -> {out}")


if __name__ == "__main__":
    main()