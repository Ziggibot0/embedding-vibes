"""
Exp6 evaluation — does the LEARNED 64-dim shape space help?

Key question: the raw-embedding relative-delta representation hit
nomic 0.970 / qwen3 1.000 in exp5. Does projecting through the learned
Barlow-Twins encoder (64-dim) preserve or improve that separation?

Also: prediction test (does the JEPA predictor beat a mean baseline?).
"""
import os, json, numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA

from train import Projector, Predictor, load_sessions, EMB_DIM, PROJ_DIM, HORIZONS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "exp3_markov_trajectories", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
ENCODERS = ["nomic_embed_text", "qwen3_embedding"]
SEED = 42


def cv_auc(X, y, name):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    aucs = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
    accs = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy")
    print(f"  {name:<40} AUC={aucs.mean():.3f}±{aucs.std():.3f}  Acc={accs.mean():.3f}")
    return aucs.mean(), accs.mean()


def evaluate(enc_name):
    print(f"\n{'='*64}")
    print(f"Exp6 eval — Learned 64-dim shape space: {enc_name}")
    print(f"{'='*64}")

    sessions, labels = load_sessions(enc_name)
    keys = sorted(sessions)
    y = np.array([1 if labels[si]["label"] == "fallacy" else 0 for si in keys])

    ckpt = torch.load(os.path.join(OUT_DIR, f"model_{enc_name}.pt"), map_location="cpu")
    proj = Projector(EMB_DIM[enc_name])
    proj.load_state_dict(ckpt["proj"])
    proj.eval()

    # Project each session's steps through the learned encoder
    with torch.no_grad():
        proj_sessions = {}
        for si in keys:
            embs = torch.tensor(sessions[si], dtype=torch.float32)
            proj_sessions[si] = proj(embs).numpy()

    # --- Learned-delta representation (concatenated 64-dim deltas, PCA50) ---
    X_delta_learned = np.array([
        np.diff(proj_sessions[si], axis=0).reshape(-1) for si in keys
    ])
    pca = PCA(n_components=min(50, X_delta_learned.shape[1], X_delta_learned.shape[0]))
    X_delta_learned_pca = pca.fit_transform(X_delta_learned)

    # --- Raw-delta baseline (from exp5, recomputed here for same protocol) ---
    X_delta_raw = np.array([
        np.diff(sessions[si], axis=0).reshape(-1) for si in keys
    ])
    pca_r = PCA(n_components=min(50, X_delta_raw.shape[1], X_delta_raw.shape[0]))
    X_delta_raw_pca = pca_r.fit_transform(X_delta_raw)

    print(f"\n  Delta separation (5-fold CV):")
    raw_auc, raw_acc = cv_auc(X_delta_raw_pca, y, "Raw-embedding deltas (exp5)")
    learned_auc, learned_acc = cv_auc(X_delta_learned_pca, y, "Learned 64-dim deltas")

    # --- Prediction test: does the JEPA predictor beat a mean baseline? ---
    pred = Predictor()
    pred.load_state_dict(ckpt["pred"])
    pred.eval()

    # Build (e_t, e_{t+k}) pairs, measure L1 of predictor vs mean-of-training
    with torch.no_grad():
        all_e = np.concatenate([proj_sessions[si] for si in keys], axis=0)
        mean_e = all_e.mean(0)
        pred_errs, mean_errs = [], []
        for si in keys:
            e = proj_sessions[si]
            T = e.shape[0]
            for t in range(T):
                for k in HORIZONS:
                    if t + k < T:
                        e_t = torch.tensor(e[t], dtype=torch.float32).unsqueeze(0)
                        e_tk = e[t + k]
                        e_pred = pred(e_t)[k].numpy()[0]
                        pred_errs.append(np.abs(e_pred - e_tk).mean())
                        mean_errs.append(np.abs(mean_e - e_tk).mean())
    pred_err = np.mean(pred_errs)
    mean_err = np.mean(mean_errs)
    print(f"\n  Prediction test (L1 to true e_{{t+k}}):")
    print(f"    JEPA predictor: {pred_err:.4f}")
    print(f"    Mean baseline:  {mean_err:.4f}")
    print(f"    {'PREDICTOR BEATS MEAN' if pred_err < mean_err else 'MEAN BEATS PREDICTOR'} "
          f"({(pred_err/mean_err - 1)*100:+.1f}%)")

    return {
        "raw_delta_auc": float(raw_auc), "raw_delta_acc": float(raw_acc),
        "learned_delta_auc": float(learned_auc), "learned_delta_acc": float(learned_acc),
        "pred_l1": float(pred_err), "mean_l1": float(mean_err),
        "pred_beats_mean": bool(pred_err < mean_err),
    }


def main():
    all_results = {}
    for enc in ENCODERS:
        try:
            all_results[enc] = evaluate(enc)
        except FileNotFoundError as e:
            print(f"\n  [{enc}] Files not found: {e}")

    out = os.path.join(OUT_DIR, "exp6_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    print(f"\n  VERDICT:")
    for enc in ENCODERS:
        if enc not in all_results:
            continue
        r = all_results[enc]
        raw, learned = r["raw_delta_auc"], r["learned_delta_auc"]
        if learned >= raw - 0.02:
            v = "LEARNED 64-DIM PRESERVES DELTA SIGNAL — compact space works"
        else:
            v = "LEARNED 64-DIM LOSES SIGNAL — projection too lossy"
        print(f"  {enc}: raw_delta={raw:.3f} learned_delta={learned:.3f} -> {v}")


if __name__ == "__main__":
    main()
