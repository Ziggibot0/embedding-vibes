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
    ap.add_argument("--device", default=None, help="default: cuda/rocm if available")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap sessions (0=all)")
    args = ap.parse_args()

    # EVAL always uses exp8's 3k labeled sessions (the test set),
    # NOT the big training corpus.
    EXP8 = os.path.join(HERE, "..", "exp8_harness_disjoint", "results")
    eval_sessions_path = os.path.join(EXP8, "sessions_meta.jsonl")
    sessions_all = [json.loads(l) for l in open(eval_sessions_path, encoding="utf-8")]
    log.info(f"loaded {len(sessions_all)} eval sessions from exp8")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"eval device: {args.device}")

    # Defensive: if GPU is busy/OOM (e.g. shared with a training process),
    # fall back to CPU rather than crash the chained pipeline.
    if args.device == "cuda":
        try:
            probe = torch.zeros(8, 8, device="cuda") + 1
            del probe
        except Exception as e:
            log.info(f"cuda probe failed ({str(e)[:80]}) — falling back to cpu")
            args.device = "cpu"
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
    if args.limit:
        sessions = sessions[:args.limit]
        log.info(f"DEBUG limit: {len(sessions)} sessions")

    # ---- embed every session into 8 dims (batched, no grad)
    Z, y, groups, bench = [], [], [], []
    B = 128
    with torch.no_grad():
        def embed_step_ids(ids):
            """ids: list of token-id lists -> (len(ids), proj_dim) tensor."""
            out = []
            with torch.no_grad():
                for k in range(0, len(ids), B):
                    chunk = ids[k:k + B]
                    L = max(len(x) for x in chunk)
                    X = torch.zeros(len(chunk), L, dtype=torch.long)
                    for r, x in enumerate(chunk):
                        X[r, :len(x)] = torch.tensor(x, dtype=torch.long)
                    z, _ = model(X.to(args.device), (X == PAD_ID).to(args.device))
                    out.append(z.cpu())
            return torch.cat(out)

        for s in sessions:
            enc = tok.encode_batch([t[:20000] for t in s["steps"]])
            ids = [[CLS_ID] + e.ids[:MAX_LEN - 1] for e in enc]
            if len(ids) < 2:
                continue
            zi = embed_step_ids(ids).mean(0)
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

    # ---- Direct A-space measurements (added 2026-08-29 ~01:35 EDT while
    # training was STILL RUNNING and before any exp9 eval numbers existed).
    # Diagnostics only — NO new pass/fail thresholds; gate-setting deferred to
    # the next pre-registration. These answer "is the space agentic
    # (format-invariant) or merely lexical", which G1 alone cannot. ----
    import torch.nn.functional as F
    from renderer import RENDERERS
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA

    render_list = list(RENDERERS.keys())
    pd_ = ck["proj_dim"]

    def embed_texts(texts):
        """Batched: all texts through the model in one pass set."""
        enc = tok.encode_batch([t[:20000] for t in texts])
        ids = [[CLS_ID] + e.ids[:MAX_LEN - 1] for e in enc]
        return embed_step_ids(ids)

    rng2 = np.random.RandomState(SEED)

    # (a) step-level cross-format consistency — the DIRECT crystallization test:
    # same step rendered in all 5 formats should land on ~the same point.
    pool = rng2.choice(len(sessions), size=min(400, len(sessions)), replace=False)
    sample = []
    for si in pool:
        st = sessions[int(si)]["steps"]
        if len(st) >= 2:
            sample.append(st[rng2.randint(0, len(st))])
    sample = sample[:300]
    # one giant batched pass: 300 steps x 5 renderings = 1500 texts
    all_rend = [RENDERERS[r](t) for t in sample for r in render_list]
    Zr = embed_texts(all_rend).reshape(len(sample), len(render_list), -1)
    step_z = list(Zr)
    if step_z:
        n_s, k = len(step_z), len(render_list)
        A = F.normalize(torch.stack(step_z), dim=2).reshape(-1, pd_)
        S = A @ A.T
        same = torch.zeros(n_s * k, n_s * k, dtype=torch.bool)
        for i in range(n_s):
            same[i * k:(i + 1) * k, i * k:(i + 1) * k] = True
        off_eye = ~torch.eye(S.shape[0], dtype=torch.bool)
        res["cross_format_step"] = {
            "n_steps": n_s, "formats": render_list,
            "within_step_mean_cos": float(S[same & off_eye].mean()),
            "between_step_mean_cos": float(S[~same].mean()),
        }
        res["cross_format_step"]["invariance_margin"] = (
            res["cross_format_step"]["within_step_mean_cos"]
            - res["cross_format_step"]["between_step_mean_cos"])
        log.info(f"cross-format step: within {res['cross_format_step']['within_step_mean_cos']:.3f} "
                 f"vs between {res['cross_format_step']['between_step_mean_cos']:.3f}")

    # (b) session-level cross-format consistency (whole session per format)
    sess_sample = rng2.choice(len(sessions), size=min(120, len(sessions)), replace=False)
    # batch: collect ALL rendered steps of ALL sampled sessions in one list
    flat_texts, sess_bounds = [], []
    for si in sess_sample:
        st = sessions[int(si)]["steps"]
        if len(st) < 3:
            continue
        start = len(flat_texts)
        for r in render_list:
            flat_texts.extend(RENDERERS[r](t) for t in st)
        sess_bounds.append((start, len(flat_texts), len(st)))
    if sess_bounds:
        Zflat = embed_texts(flat_texts)
        sess_vecs = []
        for start, end, n_steps in sess_bounds:
            block = Zflat[start:end].reshape(len(render_list), n_steps, -1)
            sess_vecs.append(block.mean(1))  # (5 formats, proj_dim)
    if sess_vecs:
        n_v, k = len(sess_vecs), len(render_list)
        Bm = F.normalize(torch.stack(sess_vecs), dim=2).reshape(-1, sess_vecs[0].shape[-1])
        S2 = Bm @ Bm.T
        same2 = torch.zeros(n_v * k, n_v * k, dtype=torch.bool)
        for i in range(n_v):
            same2[i * k:(i + 1) * k, i * k:(i + 1) * k] = True
        off_eye2 = ~torch.eye(S2.shape[0], dtype=torch.bool)
        res["cross_format_session"] = {
            "n_sessions": n_v,
            "within_session_mean_cos": float(S2[same2 & off_eye2].mean()),
            "between_session_mean_cos": float(S2[~same2].mean()),
        }

    # (c) matched-dimension lexical control: TF-IDF -> PCA-8, same LOHO protocol.
    # If our 8 dims don't beat an equally-compressed bag of words, the space
    # is a lexical hasher, not an agentic space.
    joined = [" ".join(t[:2000] for t in s["steps"]) for s in sessions]
    ga2 = np.array(groups)
    tf8 = {}
    for g in sorted(set(ga2)):
        te = np.where(ga2 == g)[0]; tr = np.where(ga2 != g)[0]
        if len(te) < 150 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        vec = TfidfVectorizer(max_features=2000, min_df=5, sublinear_tf=True)
        Xtr = vec.fit_transform([joined[int(i)] for i in tr]).toarray()
        Xte = vec.transform([joined[int(i)] for i in te]).toarray()
        p = PCA(n_components=8, random_state=SEED).fit(Xtr)
        sc = StandardScaler().fit(p.transform(Xtr))
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(
            sc.transform(p.transform(Xtr)), y[tr])
        tf8[g] = float(roc_auc_score(y[te], clf.predict_proba(
            sc.transform(p.transform(Xte)))[:, 1]))
    res["tfidf_pca8_loho"] = tf8
    if tf8:
        log.info(f"tfidf-pca8 LOHO mean: {np.mean(list(tf8.values())):.3f} "
                 f"(ours: {res.get('loho_mean')})")

    # (d) bootstrap 95% CI on our LOHO AUCs (test-set resampling, 300x)
    boot = {}
    rng3 = np.random.RandomState(SEED)
    for g in sorted(set(ga2)):
        te = np.where(ga2 == g)[0]; tr = np.where(ga2 != g)[0]
        if len(te) < 150 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Z[tr]), y[tr])
        probs = clf.predict_proba(sc.transform(Z[te]))[:, 1]
        yt = y[te]
        aucs = []
        for _ in range(300):
            b = rng3.randint(0, len(yt), len(yt))
            if len(set(yt[b])) < 2:
                continue
            aucs.append(roc_auc_score(yt[b], probs[b]))
        boot[g] = {"auc": float(roc_auc_score(yt, probs)),
                   "ci95": [float(np.percentile(aucs, 2.5)),
                            float(np.percentile(aucs, 97.5))]}
    res["loho_bootstrap"] = boot

    # (e) bootstrap 95% CI on LOBO AUCs (attempt #2 addition — the attempt #1
    # bright spot was unbootstrapped; same test-set resampling method)
    boot_lobo = {}
    rng4 = np.random.RandomState(SEED)
    ba2 = np.array(bench)
    for g in sorted(set(ba2)):
        te = np.where(ba2 == g)[0]; tr = np.where(ba2 != g)[0]
        if len(te) < 150 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Z[tr]), y[tr])
        probs = clf.predict_proba(sc.transform(Z[te]))[:, 1]
        yt = y[te]
        aucs = []
        for _ in range(300):
            bs = rng4.randint(0, len(yt), len(yt))
            if len(set(yt[bs])) < 2:
                continue
            aucs.append(roc_auc_score(yt[bs], probs[bs]))
        boot_lobo[g] = {"auc": float(roc_auc_score(yt, probs)),
                        "ci95": [float(np.percentile(aucs, 2.5)),
                                 float(np.percentile(aucs, 97.5))]}
    res["lobo_bootstrap"] = boot_lobo

    out = os.path.join(RESULTS, f"eval_run{args.run}.json")
    json.dump(res, open(out, "w"), indent=2, default=str)
    log.info(f"gates: {json.dumps(gates)}")
    log.info(f"written -> {out}")


if __name__ == "__main__":
    main()