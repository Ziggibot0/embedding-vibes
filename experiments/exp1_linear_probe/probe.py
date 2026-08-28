# Step 2-4: Linear probe experiments + visualization
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load labels
labels = np.load(os.path.join(RESULTS_DIR, "labels.npy"), allow_pickle=True)
print(f"Labels: {len(labels)} examples, {len(set(labels))} types")
print(f"Distribution: {Counter(labels).most_common()}")

ENCODERS = ["nomic_embed_text", "qwen3_embedding"]

results = {}

for enc_name in ENCODERS:
    emb_path = os.path.join(RESULTS_DIR, f"embeddings_{enc_name}.npy")
    if not os.path.exists(emb_path):
        print(f"\n[{enc_name}] No embeddings found, skipping")
        continue

    X = np.load(emb_path)
    y = np.array(labels)

    # Filter out zero-embedding rows (errors)
    nonzero_mask = np.any(X != 0, axis=1)
    X = X[nonzero_mask]
    y = y[nonzero_mask]
    print(f"\n{'='*60}")
    print(f"[{enc_name}] Embeddings: {X.shape}")
    print(f"[{enc_name}] Labels: {len(set(y))} types, {len(y)} examples")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Multiclass classification ===
    print(f"\n--- Multiclass Linear Probe ({enc_name}) ---")

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class='multinomial')

    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_macro')
    cv_acc = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    print(f"  CV Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  CV Macro-F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Chance baseline: {1/len(set(y)):.3f}")
    print(f"  Majority class baseline: {Counter(y).most_common(1)[0][1]/len(y):.3f}")

    # Train/test split for detailed report
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n  Per-class F1 (test set):")
    report = classification_report(y_test, y_pred, output_dict=True)
    for label in sorted(report.keys()):
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"    {label}: F1={report[label]['f1-score']:.3f} (n={report[label]['support']})")
    print(f"  Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"  Weighted F1: {report['weighted avg']['f1-score']:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y)))
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {sorted(set(y))}")
    print(cm)

    # === Per-fallacy one-vs-rest AUC ===
    print(f"\n--- Per-Fallacy Separability (one-vs-rest AUC) ({enc_name}) ---")
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)

    per_type_auc = {}
    for i, fallacy_type in enumerate(lb.classes_):
        if y_bin.shape[1] == 1:
            # Binary case
            y_ovr = y_bin.ravel()
        else:
            y_ovr = y_bin[:, i]

        if y_ovr.sum() < 10 or y_ovr.sum() == len(y_ovr):
            per_type_auc[fallacy_type] = float('nan')
            continue

        clf_bin = LogisticRegression(max_iter=2000, C=1.0)
        cv_auc = cross_val_score(clf_bin, X_scaled, y_ovr, cv=cv, scoring='roc_auc')
        per_type_auc[fallacy_type] = cv_auc.mean()
        print(f"  {fallacy_type}: AUC={cv_auc.mean():.3f} ± {cv_auc.std():.3f} (n={y_ovr.sum()})")

    # Rank by AUC
    print(f"\n  Ranked separability:")
    for ft, auc in sorted(per_type_auc.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else 0):
        print(f"    {ft}: {auc:.3f}")

    # === PCA for visualization ===
    print(f"\n--- PCA ({enc_name}) ---")
    pca = PCA(n_components=min(10, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    print(f"  Explained variance ratio (first 10): {pca.explained_variance_ratio_[:10]}")
    print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_[:10])}")

    # Save PCA projections
    np.save(os.path.join(RESULTS_DIR, f"pca_{enc_name}.npy"), X_pca)

    # Save results
    results[enc_name] = {
        "shape": list(X.shape),
        "cv_accuracy": float(cv_acc.mean()),
        "cv_accuracy_std": float(cv_acc.std()),
        "cv_macro_f1": float(cv_scores.mean()),
        "cv_macro_f1_std": float(cv_scores.std()),
        "chance_baseline": 1/len(set(y)),
        "majority_baseline": Counter(y).most_common(1)[0][1]/len(y),
        "per_type_auc": {k: float(v) if not np.isnan(v) else None for k, v in per_type_auc.items()},
        "pca_explained_variance": pca.explained_variance_ratio_[:10].tolist(),
        "classification_report": report,
    }

# === Cross-encoder comparison ===
print(f"\n{'='*60}")
print("CROSS-ENCODER COMPARISON")
print(f"{'='*60}")

if len(results) >= 2:
    encs = list(results.keys())
    print(f"\n  {'Metric':<25} {encs[0]:<20} {encs[1]:<20}")
    print(f"  {'CV Accuracy':<25} {results[encs[0]]['cv_accuracy']:<20.3f} {results[encs[1]]['cv_accuracy']:<20.3f}")
    print(f"  {'CV Macro-F1':<25} {results[encs[0]]['cv_macro_f1']:<20.3f} {results[encs[1]]['cv_macro_f1']:<20.3f}")
    print(f"  {'Chance baseline':<25} {results[encs[0]]['chance_baseline']:<20.3f} {results[encs[1]]['chance_baseline']:<20.3f}")
    print(f"  {'Majority baseline':<25} {results[encs[0]]['majority_baseline']:<20.3f} {results[encs[1]]['majority_baseline']:<20.3f}")

    # Compare per-type AUC rankings
    print(f"\n  Per-fallacy AUC comparison:")
    print(f"  {'Fallacy type':<25} {encs[0]:<15} {encs[1]:<15} {'Diff':<10}")
    aucs0 = results[encs[0]]['per_type_auc']
    aucs1 = results[encs[1]]['per_type_auc']
    for ft in sorted(aucs0.keys()):
        a0 = aucs0[ft] or 0
        a1 = aucs1.get(ft, None) or 0
        print(f"  {ft:<25} {a0:<15.3f} {a1:<15.3f} {a1-a0:<+10.3f}")

# Save all results
with open(os.path.join(RESULTS_DIR, "probe_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_DIR}/probe_results.json")