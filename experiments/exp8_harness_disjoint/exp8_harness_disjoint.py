"""Exp8 — harness-disjoint transfer of frozen session features.

Pipeline: stream -> extract (metadata intact) -> embed (resumable cache)
-> evaluate (in_format CV + leave-one-harness/benchmark-out) -> gates -> report.

All thresholds are pre-registered in DESIGN.md; this file contains no magic
numbers beyond those and standard protocol constants (seed 42, 5 folds,
logreg C=1.0 — identical to exp7 for comparability).
"""
import os, sys, json, time, re, logging
from collections import Counter, defaultdict

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp6_joint_jepa"))

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
SEED = 42
EMB_MODEL = "nomic-embed-text"
OLLAMA = "http://localhost:11434"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
    logging.FileHandler(os.path.join(HERE, "exp8.log"), mode="w"),
    logging.StreamHandler(sys.stdout)])
log = logging.getLogger("exp8")

# ---------------------------------------------------------------- phase 1
def extract_exgentic_session(row):
    """Identical logic to exp6 data_prep.extract_exgentic_session + metadata.

    Kept deliberately line-for-line in step logic (parse gen_ai messages,
    '[tool_call] ' prefix, consecutive dedupe) so exp7's 0.901 is comparable.
    """
    steps = []
    for sp in row.get("spans", []):
        attrs = sp.get("attributes") or {}
        out = attrs.get("gen_ai.output.messages")
        if out:
            try:
                msgs = json.loads(out) if isinstance(out, str) else out
            except Exception:
                msgs = []
            for m in msgs:
                for part in m.get("parts", []):
                    if part.get("type") == "text" and part.get("content"):
                        steps.append(part["content"].strip())
                    elif part.get("type") == "tool_call":
                        steps.append(f"[tool_call] {part.get('name','')}: {part.get('arguments','')}")
        inp = attrs.get("gen_ai.input.messages")
        if inp and not steps:
            try:
                msgs = json.loads(inp) if isinstance(inp, str) else inp
            except Exception:
                msgs = []
            for m in msgs:
                for part in m.get("parts", []):
                    if part.get("type") == "text" and part.get("content"):
                        steps.append(part["content"].strip())
    deduped = []
    for s in steps:
        if not deduped or deduped[-1] != s:
            deduped.append(s)
    meta = {k: row.get(k) for k in
            ("harness", "benchmark", "benchmark_subset", "models", "success",
             "score", "total_tokens", "execution_time", "run_id",
             "session_id", "status")}
    meta["n_steps_source"] = row.get("steps")  # int count; 'steps' name is reserved for the text list
    return deduped, meta


def phase_extract(max_sessions=3000):
    from datasets import load_dataset
    out_path = os.path.join(RESULTS, "sessions_meta.jsonl")
    if os.path.exists(out_path):
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        log.info(f"[extract] cache hit {out_path} ({n} sessions) — skipping stream")
        return out_path
    log.info("[extract] streaming Exgentic/agent-llm-traces-v2 ...")
    ds = load_dataset("Exgentic/agent-llm-traces-v2", split="train", streaming=True)
    n_written = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            if row.get("success") is None or not row.get("harness"):
                continue  # need outcome label AND harness metadata (exp8's purpose)
            steps, meta = extract_exgentic_session(row)
            if not (3 <= len(steps) <= 40):
                continue
            steps = [s[:20000] for s in steps]  # nomic ctx guard (same as exp7 embed_text)
            rec = {"steps": steps, "source": "exgentic", **meta}
            f.write(json.dumps(rec) + "\n")
            n_written += 1
            if n_written % 200 == 0:
                log.info(f"  {n_written} written ({time.time()-t0:.0f}s)")
            if n_written >= max_sessions:
                log.info(f"[extract] cap {max_sessions} reached — documented in DESIGN.md")
                break
    log.info(f"[extract] done: {n_written} sessions in {time.time()-t0:.0f}s -> {out_path}")
    return out_path

# ---------------------------------------------------------------- phase 2
def embed_text(text, retries=3):
    if len(text) > 20000:
        text = text[:20000]
    for attempt in range(retries):
        try:
            r = requests.post(f"{OLLAMA}/api/embeddings",
                              json={"model": EMB_MODEL, "prompt": text}, timeout=120)
            if r.status_code == 200:
                return r.json()["embedding"]
            if "context length" in r.text and len(text) > 5000:
                text = text[:5000]
                continue
            if attempt == retries - 1:
                raise RuntimeError(f"embed failed: {r.status_code} {r.text[:200]}")
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(1)
    raise RuntimeError("unreachable")


def phase_embed(sessions_path):
    cache_path = os.path.join(RESULTS, f"emb_cache_{EMB_MODEL}.npy")
    # seed from exp7's cache: identical extraction logic => most texts already embedded
    seed_path = os.path.join(HERE, "..", "exp7_real_data_gate", "results",
                             f"emb_cache_{EMB_MODEL}.npy")
    texts = set()
    for line in open(sessions_path, encoding="utf-8"):
        s = json.loads(line)
        texts.update(s["steps"])
    texts = list(texts)
    if os.path.exists(cache_path):
        arr = np.load(cache_path, allow_pickle=True).item()
        cache = {k: v for k, v in arr.items()}
        texts = [t for t in texts if t not in cache]
    else:
        cache = {}
        if os.path.exists(seed_path):
            seed = np.load(seed_path, allow_pickle=True).item()
            before = len(texts)
            texts = [t for t in texts if t not in seed]
            cache.update({k: v for k, v in seed.items() if isinstance(v, list)})
            log.info(f"[embed] seeded {len(cache)} vectors from exp7 cache ({before-len(texts)} texts skipped)")
    log.info(f"[embed] unique texts to embed: {len(texts)} (cache has {len(cache)})")
    if texts:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        t0 = time.time(); done = 0; fails = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(embed_text, t): t for t in texts}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    cache[t] = fut.result()
                except Exception as e:
                    fails += 1
                    log.warning(f"  embed fail: {t[:60]!r}: {str(e)[:100]}")
                done += 1
                if done % 500 == 0:
                    rate = done / (time.time() - t0)
                    log.info(f"  {done}/{len(texts)} ({rate:.0f}/s ETA {(len(texts)-done)/rate:.0f}s)")
        log.info(f"[embed] done: +{done}, {fails} failures, cache={len(cache)} in {time.time()-t0:.0f}s")
        np.save(cache_path, np.array(cache, dtype=object), allow_pickle=True)
    return cache_path

# ---------------------------------------------------------------- phase 3
TAG_RE = re.compile(r"\[(?:tool_call\]\s*)?[a-zA-Z_]+\]")


def build_features(*args, **kwargs):
    """DEPRECATED stub: features are built inside evaluation closures to
    enforce fit-on-train-only (no scaler/PCA/tfidf leakage)."""
    raise NotImplementedError


def tag_features(sessions):
    rows = []
    for s in sessions:
        c = Counter()
        for t in s["steps"]:
            c.update(TAG_RE.findall(t))
        tool_rate = np.mean([bool(re.match(r"\[(tool_call|user|assistant|tool)", t)) for t in s["steps"]])
        rows.append((c, len(s["steps"]), tool_rate))
    return rows


def session_join(s):
    return " ".join(t[:2000] for t in s["steps"])


def main():
    t_start = time.time()
    sessions_path = phase_extract()
    sessions = [json.loads(l) for l in open(sessions_path, encoding="utf-8")]
    log.info(f"[main] {len(sessions)} sessions loaded")

    cache = np.load(phase_embed(sessions_path), allow_pickle=True).item()
    lens = Counter(len(v) for v in cache.values() if isinstance(v, (list, tuple)))
    EMB_LEN = lens.most_common(1)[0][0]
    log.info(f"[main] cache size {len(cache)}, EMB_LEN={EMB_LEN}")

    # sessions with >=2 embedded steps and non-null success label
    keep, y_list, groups = [], [], []
    for s in sessions:
        embs = [cache[t] for t in s["steps"]
                if t in cache and isinstance(cache[t], list) and len(cache[t]) == EMB_LEN]
        if len(embs) >= 2 and s.get("success") is not None:
            keep.append({**s, "_embs": np.array(embs, dtype=np.float32)})
            y_list.append(1 if s["success"] else 0)
            groups.append(s.get("harness"))
    y = np.array(y_list)
    log.info(f"[main] usable: {len(keep)} sessions, pos={y.sum()}, harness groups: {Counter(groups)}")

    # -------- feature blocks (embedding-free, per session) --------
    static_full = np.array([np.concatenate([s["_embs"].mean(0), s["_embs"][-1]]) for s in keep])
    meanvel     = np.array([np.diff(s["_embs"], axis=0).mean(0) if len(s["_embs"]) > 1 else np.zeros(EMB_LEN) for s in keep])
    length      = np.array([[len(s["steps"])] for s in keep], dtype=float)
    tagsrows    = tag_features(keep)
    joined      = [session_join(s) for s in keep]

    # tag matrix with train-only vocab is handled in eval loop via closure state
    tag_counts = [r[0] for r in tagsrows]
    tag_rate   = np.array([[r[2], r[1]] for r in tagsrows], dtype=float)

    # -------- shared evaluation machinery --------
    def fit_predict(Xtr_builder, Xte_builder, tr, te):
        """Build fold-fit-only transforms then a logreg; returns (y_prob, y_true)."""
        Xtr, Xte = Xtr_builder(tr), Xte_builder(te)
        sc = StandardScaler(with_mean=not hasattr(Xtr, "toarray")).fit(Xtr)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), y[tr])
        return clf.predict_proba(sc.transform(Xte))[:, 1], y[te]

    def metrics(y_true, y_prob):
        return {
            "auc": float(roc_auc_score(y_true, y_prob)),
            "acc": float(accuracy_score(y_true, y_prob >= 0.5)),
            "brier": float(brier_score_loss(y_true, y_prob)),
        }

    def ece(y_true, y_prob, bins=10):
        # 10-bin expected calibration error (P1 consumes thresholds)
        probs = np.clip(y_prob, 0, 1)
        edges = np.linspace(0, 1, bins + 1)
        e = 0.0
        for i in range(bins):
            m = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= edges[i + 1])
            if m.sum() > 0:
                e += (m.sum() / len(probs)) * abs(probs[m].mean() - y_true[m].mean())
        return float(e)

    def in_format_cv(X_builder_pair, feature_name):
        """5-fold stratified CV; builder_pair = (fit_fn(tr)->Xtr, apply_fn(te)->Xte)."""
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        ms = []
        for tr, te in skf.split(np.zeros(len(y)), y):
            yp, yt = fit_predict(X_builder_pair[0], X_builder_pair[1], tr, te)
            ms.append({**metrics(yt, yp), "ece": ece(yt, yp)})
        log.info(f"[in_format] {feature_name:<18} AUC {np.mean([m['auc'] for m in ms]):.3f}")
        return {"per_fold": ms, "mean_auc": float(np.mean([m['auc'] for m in ms])),
                "mean_ece": float(np.mean([m['ece'] for m in ms])),
                "mean_brier": float(np.mean([m['brier'] for m in ms]))}

    def disjoint_eval(builders, group_values, split_name, min_group=150):
        """Leave-one-group-out; builders = (fit(tr)->Xtr, apply(te)->Xte)."""
        groups_arr = np.array([str(g) for g in group_values])
        out = {}
        skipped = {}
        for g in sorted(set(groups_arr)):
            te = np.where(groups_arr == g)[0]; tr = np.where(groups_arr != g)[0]
            # class balance guard
            if len(te) < min_group:
                skipped[g] = int(len(te))
                continue
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                skipped[g] = f"single-class {len(te)}"
                continue
            yp, yt = fit_predict(builders[0], builders[1], tr, te)
            out[g] = {**metrics(yt, yp), "ece": ece(yt, yp), "n": int(len(te)),
                      "n_pos": int(yt.sum())}
            log.info(f"[{split_name}] hold={g} AUC {out[g]['auc']:.3f} (n={len(te)})")
        return out, skipped

    results = {"n_sessions": len(keep), "n_success": int(y.sum()),
               "harness_dist": dict(Counter(groups)), "ece_bins": 10, "seed": SEED}

    # -------- builders per feature (fit-on-train-only discipline) --------
    def dense(mat):
        return (lambda tr: mat[[*map(int, tr)]], lambda te: mat[[*map(int, te)]])

    tag_vocab_state = {}
    def tag_fit(tr):
        df = Counter()
        for i in tr:
            df.update(tag_counts[int(i)].keys())
        vocab = [t for t, _ in df.most_common(40)]
        tag_vocab_state["v"] = {t: j for j, t in enumerate(vocab)}
        M = np.zeros((len(tr), len(vocab) + 2))
        for r, i in enumerate(tr):
            i = int(i)
            for t, c in tag_counts[i].items():
                if t in tag_vocab_state["v"]:
                    M[r, tag_vocab_state["v"][t]] = c
            M[r, -2] = tag_rate[i, 0]; M[r, -1] = tag_rate[i, 1]
        return M
    def tag_apply(te):
        M = np.zeros((len(te), len(tag_vocab_state["v"]) + 2))
        for r, i in enumerate(te):
            i = int(i)
            for t, c in tag_counts[i].items():
                if t in tag_vocab_state["v"]:
                    M[r, tag_vocab_state["v"][t]] = c
            M[r, -2] = tag_rate[i, 0]; M[r, -1] = tag_rate[i, 1]
        return M

    tfidf_state = {}
    def tfidf_fit(tr):
        v = TfidfVectorizer(max_features=2000, min_df=5, sublinear_tf=True)
        X = v.fit_transform([joined[int(i)] for i in tr])
        tfidf_state["v"] = v
        return X
    def tfidf_apply(te):
        return tfidf_state["v"].transform([joined[int(i)] for i in te])

    pca_state = {}
    def pca8_fit(tr):
        sc = StandardScaler().fit(static_full[[*map(int, tr)]])
        p = PCA(n_components=8, random_state=SEED).fit(sc.transform(static_full[[*map(int, tr)]]))
        pca_state["sc"], pca_state["p"] = sc, p
        return p.transform(sc.transform(static_full[[*map(int, tr)]]))
    def pca8_apply(te):
        return pca_state["p"].transform(pca_state["sc"].transform(static_full[[*map(int, te)]]))

    def sparse_stack(tf_builder, dense_mat):
        """tfidf + dense feature stack; fits tfidf on tr then hstacks scaled dense."""
        def fit(tr):
            Xs = tfidf_fit(tr)
            from scipy.sparse import hstack, csr_matrix
            D = static_full if dense_mat is None else dense_mat
            Dtr = StandardScaler().fit_transform(D[[*map(int, tr)]])
            tfidf_state["ds"] = StandardScaler().fit(D[[*map(int, tr)]])
            return hstack([Xs, csr_matrix(Dtr)]).tocsr()
        def apply(te):
            from scipy.sparse import hstack, csr_matrix
            Xs = tfidf_apply(te)
            D = static_full if dense_mat is None else dense_mat
            return hstack([Xs, csr_matrix(tfidf_state["ds"].transform(D[[*map(int, te)]]))]).tocsr()
        return (fit, apply)

    from scipy.sparse import hstack as sp_hstack, csr_matrix

    def tfidf_meanvel_fit(tr):
        Xs = tfidf_fit(tr)
        Dtr = StandardScaler().fit_transform(meanvel[[*map(int, tr)]])
        tfidf_state["ds"] = StandardScaler().fit(meanvel[[*map(int, tr)]])
        return sp_hstack([Xs, csr_matrix(Dtr)]).tocsr()
    def tfidf_meanvel_apply(te):
        return sp_hstack([tfidf_apply(te), csr_matrix(tfidf_state["ds"].transform(meanvel[[*map(int, te)]]))]).tocsr()

    FEATS = {
        "static_full": dense(static_full),
        "static_pca8": (pca8_fit, pca8_apply),
        "meanvel":     dense(meanvel),
        "tfidf":       (tfidf_fit, tfidf_apply),
        "tags":        (tag_fit, tag_apply),
        "length":      dense(length),
        "tfidf+meanvel": (tfidf_meanvel_fit, tfidf_meanvel_apply),
    }

    bench = [s.get("benchmark") for s in keep]
    models_g = [",".join(s.get("models") or []) for s in keep]

    for fname, (fb, ab) in FEATS.items():
        feature_name = fname  # for logging inside disjoint_eval
        results.setdefault("in_format", {})[fname] = in_format_cv((fb, ab), fname)
        results.setdefault("loho", {})[fname], sk = disjoint_eval((fb, ab), groups, "loho")
        if sk: results["loho_skipped"] = results.get("loho_skipped", {}); results["loho_skipped"][fname] = sk
        results.setdefault("lobo", {})[fname], sk2 = disjoint_eval((fb, ab), bench, "lobo")
        if sk2: results["lobo_skipped"] = results.get("lobo_skipped", {}); results["lobo_skipped"][fname] = sk2

    # models-disjoint only if >=2 model groups with >=150
    mc = Counter(models_g)
    if sum(1 for g, n in mc.items() if n >= 150) >= 2:
        for fname, (fb, ab) in FEATS.items():
            feature_name = fname
            results.setdefault("lomo", {})[fname], _ = disjoint_eval((fb, ab), models_g, "lomo")

    with open(os.path.join(RESULTS, "exp8_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"[main] results written ({time.time()-t_start:.0f}s total)")

    # ---------------- gates + quick-read summary ----------------
    def m(res, split, feat, field="mean_auc"):
        return res.get(split, {}).get(feat, {}).get(field, np.nan)
    def mm(res, split, feat, groups_out_key):
        per = res.get(split, {}).get(feat, {})
        aucs = [v["auc"] for v in per.values() if isinstance(v, dict) and "auc" in v]
        return float(np.mean(aucs)) if aucs else np.nan

    summary_lines = ["# Exp8 quick-read summary (auto-generated)", ""]
    def row(label, *vals):
        summary_lines.append("| " + " | ".join([label] + [f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else "—" for v in vals]) + " |")

    header = ["feature", "in_format", "loho", "lobo"]
    summary_lines.append("| " + " | ".join(header) + " |")
    summary_lines.append("|" + "---|" * len(header))
    for f_ in FEATS:
        row(f_, m(results, "in_format", f_), mm(results, "loho", f_, "x"), mm(results, "lobo", f_, "x"))

    inf = {f_: m(results, "in_format", f_) for f_ in FEATS}
    loho = {f_: mm(results, "loho", f_, "x") for f_ in FEATS}
    drop = {f_: (inf[f_] - loho[f_]) if not (np.isnan(inf[f_]) or np.isnan(loho[f_])) else np.nan for f_ in FEATS}

    gates = {}
    gates["A_monitor_transfers"] = bool(inf.get("static_full", 0) >= 0.75 and loho.get("static_full", 0) >= 0.70)
    gates["B_geometry_beats_lexical_drop"] = bool((drop["tfidf"] - drop["static_full"]) >= 0.03) if not np.isnan(drop["tfidf"]) else None
    gates["C_velocity_null_recheck"] = bool((m(results, "in_format", "tfidf+meanvel") - m(results, "in_format", "tfidf")) <= 0.02)
    gates["D_lowdim_transfers"] = bool((drop["static_pca8"] - drop["static_full"]) <= 0.02) if not np.isnan(drop["static_pca8"]) else None
    summary_lines += ["", "## Drops (in_format − loho mean)", ""]
    for f_ in FEATS:
        summary_lines.append(f"- {f_}: in_format {inf[f_]:.3f} → loho {loho[f_]:.3f} (drop {drop[f_]:.3f})")
    summary_lines += ["", "## Gates (pre-registered)", ""]
    for g_, v_ in gates.items():
        summary_lines.append(f"- **{g_}: {'PASS' if v_ else 'FAIL'}**")

    with open(os.path.join(RESULTS, "exp8_summary.md"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    log.info("[gates] " + json.dumps(gates))
    log.info(f"[main] done in {time.time()-t_start:.0f}s. Read: results/exp8_summary.md")


if __name__ == "__main__":
    main()