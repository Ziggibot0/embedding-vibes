"""
Experiment 2a: Chunking Strategy Ablation
Tests 4 chunking methods to see which yields the most fallacy detection signal.
"""
import csv, json, re, os, time, numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "logical-fallacy-repo", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/embeddings"

# ============================================================
# CHUNKING STRATEGIES
# ============================================================

def chunk_whole(text):
    """Baseline: no chunking, return the whole text as one chunk"""
    return [text]

def chunk_sentences(text):
    """Split on sentence boundaries"""
    # Simple sentence splitter — handles ., !, ? followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# Logical connectives that mark reasoning transitions
CONNECTIVES = [
    'therefore', 'because', 'so', 'thus', 'hence', 'which means',
    'consequently', 'since', 'as a result', 'it follows that',
    'which proves', 'which shows', 'which demonstrates',
    'we can conclude', 'this proves', 'this shows',
    'thus proving', 'thereby', 'wherefore', 'accordingly',
    'for this reason', 'on account of', 'due to',
]

def chunk_connectives(text):
    """Split on logical connectives — each chunk is a logical step"""
    # Build a regex that splits BEFORE connectives
    pattern = r'\s+(?:' + '|'.join(re.escape(c) for c in CONNECTIVES) + r')\b'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    # Also capture the connective itself and prepend it to the following chunk
    # Re-find all connectives to reconstruct
    connective_matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    
    chunks = []
    if connective_matches:
        # Text before first connective
        chunks.append(text[:connective_matches[0].start()].strip())
        # Text between connectives
        for i, match in enumerate(connective_matches):
            start = match.start()
            end = connective_matches[i+1].start() if i+1 < len(connective_matches) else len(text)
            chunks.append(text[start:end].strip())
    else:
        # No connectives found, return whole text
        chunks = [text.strip()]
    
    return [c for c in chunks if c and len(c) > 3]

def chunk_clauses(text):
    """Split on commas, semicolons, and connectives — finer than sentences"""
    # Split on commas, semicolons, and connectives
    pattern = r'[,;]\s*|\s+(?:' + '|'.join(re.escape(c) for c in CONNECTIVES) + r')\b'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]

CHUNKING_STRATEGIES = {
    'whole': chunk_whole,
    'sentence': chunk_sentences,
    'connective': chunk_connectives,
    'clause': chunk_clauses,
}

# ============================================================
# EMBEDDING
# ============================================================

def embed_batch(texts, model="nomic-embed-text:latest", concurrency=16):
    """Embed a list of texts concurrently via Ollama"""
    def embed_one(args):
        idx, text = args
        try:
            resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": text}, timeout=30)
            return idx, resp.json().get("embedding", [])
        except:
            return idx, []
    
    embeddings = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(embed_one, (i, t)): i for i, t in enumerate(texts)}
        for f in as_completed(futures):
            idx, emb = f.result()
            embeddings[idx] = emb
    
    return np.array([e if e else [0.0]*768 for e in embeddings], dtype=np.float32)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading fallacy dataset...")
with open(os.path.join(DATA_DIR, "edu_all.csv"), encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_rows = [r for r in reader if r.get("updated_label") and r.get("updated_label") != "miscellaneous"]

# Sample 200 fallacies for speed (stratified)
import random
random.seed(42)
fallacy_sample = []
for label in set(r["updated_label"] for r in all_rows):
    type_examples = [r for r in all_rows if r["updated_label"] == label]
    n = min(15, len(type_examples))
    fallacy_sample.extend(random.sample(type_examples, n))

# Valid arguments from exp1 falsification test
valid_arguments = [
    "A study of 10,000 students across 50 schools found that 85% perform better with adequate sleep. This suggests most students benefit from sufficient rest.",
    "In a randomized controlled trial of 500 patients, the drug reduced symptoms in 73% of cases compared to 22% in the placebo group, demonstrating effectiveness.",
    "After surveying 2,000 residents from every neighborhood, researchers found consistent support for the park initiative across all demographics.",
    "The increase in temperature correlates with increased ice cream sales because both are caused by summer heat, not because ice cream causes temperature rise.",
    "The controlled experiment isolated the variable: group A received the treatment, group B did not. The significant difference in outcomes indicates the treatment caused the effect.",
    "Removing the contaminated water source eliminated the outbreak, and reintroducing it caused new cases. This confirms the water as the cause.",
    "We can either invest in renewable energy now or face higher costs later, but we could also do nothing and accept the risks.",
    "You have three options: accept the offer, decline it, or negotiate the terms before deciding.",
    "The peer-reviewed study published in Nature has been replicated by three independent labs, making its findings reliable.",
    "The expert witness has relevant credentials, published research in this specific field, and their claims are supported by evidence.",
    "The professor's argument contains a mathematical error in step 3 where they divide by zero, invalidating the conclusion.",
    "The study's methodology has a selection bias because it only surveyed college graduates, not the general population.",
    "Democracies tend to have higher GDP per capita because they protect property rights and encourage innovation, as shown by cross-country regression analysis.",
    "Exercise improves cardiovascular health because it strengthens the heart muscle and improves blood circulation, as demonstrated in longitudinal studies.",
    "The word 'bank' in this context refers specifically to a financial institution, not a river bank, as indicated by the surrounding discussion of interest rates.",
    "While the story is heartbreaking, we must evaluate the policy based on evidence of its outcomes, not on the emotional impact of individual cases.",
    "All mammals are warm-blooded. Whales are mammals. Therefore, whales are warm-blooded.",
    "If it rains, the ground gets wet. It is raining. Therefore, the ground is wet.",
    "Among 1,000 randomly sampled voters, 52% preferred candidate A, with a margin of error of 3%. The race is within the margin of error.",
    "Increased CO2 from industrial emissions traps more infrared radiation in the atmosphere. This additional heat energy raises global temperatures. Temperature records confirm this prediction.",
    "Just as overfishing depletes fish stocks faster than they can recover, overharvesting any renewable resource beyond its regeneration rate leads to depletion.",
    "The person making the extraordinary claim bears the burden of providing extraordinary evidence. Without such evidence, we should withhold belief.",
    "If we allow 10% more development without upgrading infrastructure, the water system will exceed capacity and service interruptions will occur.",
    "The IPCC report represents consensus from 195 countries and thousands of climate scientists, making it the most authoritative source on climate change.",
    "Students who study regularly tend to retain information better because spaced repetition strengthens memory consolidation, as shown in cognitive psychology research.",
    "The bridge collapsed because the steel supports corroded beyond their load-bearing threshold, as determined by forensic engineering analysis.",
    "Vaccines work by training the immune system to recognize pathogens, which is why vaccinated populations have lower infection rates, as shown in epidemiological data.",
    "The company went bankrupt because its expenses exceeded revenue for three consecutive years, depleting its cash reserves entirely.",
    "Regular oil changes extend engine life because fresh oil maintains lubrication properties that prevent metal-on-metal wear, as documented in manufacturer specifications.",
]

print(f"Fallacy sample: {len(fallacy_sample)} examples across {len(set(r['updated_label'] for r in fallacy_sample))} types")
print(f"Valid arguments: {len(valid_arguments)}")

fallacy_texts = [r["source_article"] for r in fallacy_sample]
fallacy_labels = [r["updated_label"] for r in fallacy_sample]
valid_texts = valid_arguments
valid_labels = ["valid"] * len(valid_texts)

all_texts = fallacy_texts + valid_texts
all_type_labels = fallacy_labels + valid_labels
all_binary_labels = [1]*len(fallacy_texts) + [0]*len(valid_texts)

# ============================================================
# RUN CHUNKING STRATEGIES
# ============================================================

results = {}

for strategy_name, chunk_fn in CHUNKING_STRATEGIES.items():
    print(f"\n{'='*60}")
    print(f"CHUNKING STRATEGY: {strategy_name}")
    print(f"{'='*60}")
    
    # Chunk all texts
    all_chunks = []
    chunk_ranges = []  # (start, end) indices into all_chunks for each example
    
    for text in all_texts:
        chunks = chunk_fn(text)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        end = len(all_chunks)
        chunk_ranges.append((start, end))
    
    chunk_counts = [end - start for start, end in chunk_ranges]
    print(f"  Total chunks: {len(all_chunks)} (mean {np.mean(chunk_counts):.1f} per example, "
          f"min {min(chunk_counts)}, max {max(chunk_counts)})")
    
    # Embed all chunks
    print(f"  Embedding {len(all_chunks)} chunks...")
    t0 = time.time()
    chunk_embeddings = embed_batch(all_chunks, concurrency=16)
    embed_time = time.time() - t0
    print(f"  Embedded in {embed_time:.1f}s, shape: {chunk_embeddings.shape}")
    
    # For each example, aggregate chunk embeddings into a single representation
    # Method 1: Mean of chunk embeddings (trajectory aggregate)
    # Method 2: Last chunk embedding (terminal point)
    # Method 3: Max over per-chunk fallacy probability (best single chunk)
    
    mean_embeddings = np.zeros((len(all_texts), 768), dtype=np.float32)
    last_embeddings = np.zeros((len(all_texts), 768), dtype=np.float32)
    max_embeddings = np.zeros((len(all_texts), 768), dtype=np.float32)  # max-norm chunk
    
    for i, (start, end) in enumerate(chunk_ranges):
        chunks_emb = chunk_embeddings[start:end]
        # Filter zero embeddings
        valid_chunks = chunks_emb[np.any(chunks_emb != 0, axis=1)]
        if len(valid_chunks) == 0:
            continue
        mean_embeddings[i] = valid_chunks.mean(axis=0)
        last_embeddings[i] = valid_chunks[-1]
        # Max-norm: pick the chunk with highest L2 norm (most "active" region)
        norms = np.linalg.norm(valid_chunks, axis=1)
        max_embeddings[i] = valid_chunks[np.argmax(norms)]
    
    # ============================================================
    # BINARY FALLACY vs VALID PROBE
    # ============================================================
    
    y_binary = np.array(all_binary_labels)
    
    for agg_name, agg_emb in [("mean", mean_embeddings), ("last", last_embeddings), ("maxnorm", max_embeddings)]:
        # Filter zero embeddings
        mask = np.any(agg_emb != 0, axis=1)
        X = agg_emb[mask]
        y = y_binary[mask]
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        acc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X_s, y, cv=cv, scoring="accuracy")
        auc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X_s, y, cv=cv, scoring="roc_auc")
        
        print(f"  [{strategy_name}/{agg_name}] Binary AUC: {auc.mean():.3f} ± {auc.std():.3f}, "
              f"Acc: {acc.mean():.3f} ± {acc.std():.3f}")
        
        key = f"{strategy_name}_{agg_name}"
        results[key] = {
            "strategy": strategy_name,
            "aggregation": agg_name,
            "binary_auc": float(auc.mean()),
            "binary_auc_std": float(auc.std()),
            "binary_acc": float(acc.mean()),
            "binary_acc_std": float(acc.std()),
            "n_chunks_mean": float(np.mean(chunk_counts)),
            "n_chunks_std": float(np.std(chunk_counts)),
        }
    
    # ============================================================
    # MULTICLASS FALLACY TYPE PROBE (fallacies only)
    # ============================================================
    
    y_type = np.array(all_type_labels)
    fallacy_mask = np.array([l != "valid" for l in all_type_labels])
    
    for agg_name, agg_emb in [("mean", mean_embeddings), ("last", last_embeddings)]:
        # Only fallacies
        mask = np.any(agg_emb != 0, axis=1) & fallacy_mask
        X = agg_emb[mask]
        y = y_type[mask]
        
        if len(set(y)) < 2:
            continue
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        acc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X_s, y, cv=cv, scoring="accuracy")
        f1 = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X_s, y, cv=cv, scoring="f1_macro")
        
        print(f"  [{strategy_name}/{agg_name}] Multiclass Acc: {acc.mean():.3f}, F1: {f1.mean():.3f} "
              f"({len(set(y))} classes, {len(y)} examples)")
        
        results[f"{strategy_name}_{agg_name}_multiclass"] = {
            "strategy": strategy_name,
            "aggregation": agg_name,
            "multiclass_acc": float(acc.mean()),
            "multiclass_f1": float(f1.mean()),
        }

# ============================================================
    # TRAJECTORY SHAPE METRICS
    # ============================================================
    
    print(f"\n  Trajectory shape metrics ({strategy_name}):")
    
    fallacy_path_lengths = []
    valid_path_lengths = []
    fallacy_direction_changes = []
    valid_direction_changes = []
    
    for i, (start, end) in enumerate(chunk_ranges):
        chunks_emb = chunk_embeddings[start:end]
        valid_chunks = chunks_emb[np.any(chunks_emb != 0, axis=1)]
        
        if len(valid_chunks) < 2:
            continue
        
        # Path length: sum of consecutive distances
        diffs = np.diff(valid_chunks, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        path_length = distances.sum()
        
        # Direction changes: count sign changes in consecutive diffs
        if len(diffs) >= 2:
            # Normalize diffs to unit vectors
            norms = np.linalg.norm(diffs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            unit_diffs = diffs / norms
            # Cosine similarity between consecutive directions
            cos_sims = np.sum(unit_diffs[:-1] * unit_diffs[1:], axis=1)
            direction_changes = np.sum(cos_sims < 0)  # count negative cosines
        else:
            direction_changes = 0
        
        if all_binary_labels[i] == 1:
            fallacy_path_lengths.append(path_length)
            fallacy_direction_changes.append(direction_changes)
        else:
            valid_path_lengths.append(path_length)
            valid_direction_changes.append(direction_changes)
    
    if fallacy_path_lengths and valid_path_lengths:
        from scipy.stats import mannwhitneyu
        stat_p, p_p = mannwhitneyu(fallacy_path_lengths, valid_path_lengths, alternative='two-sided')
        stat_d, p_d = mannwhitneyu(fallacy_direction_changes, valid_direction_changes, alternative='two-sided')
        
        print(f"    Path length: fallacy={np.mean(fallacy_path_lengths):.2f}, valid={np.mean(valid_path_lengths):.2f}, p={p_p:.4f}")
        print(f"    Direction changes: fallacy={np.mean(fallacy_direction_changes):.1f}, valid={np.mean(valid_direction_changes):.1f}, p={p_d:.4f}")
        
        results[f"{strategy_name}_shape"] = {
            "strategy": strategy_name,
            "fallacy_path_length_mean": float(np.mean(fallacy_path_lengths)),
            "valid_path_length_mean": float(np.mean(valid_path_lengths)),
            "path_length_p_value": float(p_p),
            "fallacy_direction_changes_mean": float(np.mean(fallacy_direction_changes)),
            "valid_direction_changes_mean": float(np.mean(valid_direction_changes)),
            "direction_changes_p_value": float(p_d),
        }

# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*60}")
print("SUMMARY: Binary fallacy-vs-valid AUC by chunking strategy")
print(f"{'='*60}")
print(f"{'Strategy':<12} {'Agg':<10} {'AUC':<10} {'Acc':<10} {'Chunks/ex':<12}")
print("-" * 60)

for key, r in sorted(results.items(), key=lambda x: -x[1].get('binary_auc', 0)):
    if 'binary_auc' in r:
        print(f"{r['strategy']:<12} {r['aggregation']:<10} {r['binary_auc']:<10.3f} {r['binary_acc']:<10.3f} {r.get('n_chunks_mean', 1):<12.1f}")

print(f"\n{'='*60}")
print("SUMMARY: Multiclass fallacy-type accuracy by chunking strategy")
print(f"{'='*60}")
print(f"{'Strategy':<12} {'Agg':<10} {'Acc':<10} {'F1':<10}")
print("-" * 40)

for key, r in sorted(results.items(), key=lambda x: -x[1].get('multiclass_acc', 0)):
    if 'multiclass_acc' in r:
        print(f"{r['strategy']:<12} {r['aggregation']:<10} {r['multiclass_acc']:<10.3f} {r['multiclass_f1']:<10.3f}")

print(f"\n{'='*60}")
print("SUMMARY: Trajectory shape metrics")
print(f"{'='*60}")

for key, r in sorted(results.items(), key=lambda x: x[1].get('path_length_p_value', 1)):
    if 'path_length_p_value' in r:
        sig_p = "***" if r['path_length_p_value'] < 0.001 else "**" if r['path_length_p_value'] < 0.01 else "*" if r['path_length_p_value'] < 0.05 else "ns"
        sig_d = "***" if r['direction_changes_p_value'] < 0.001 else "**" if r['direction_changes_p_value'] < 0.01 else "*" if r['direction_changes_p_value'] < 0.05 else "ns"
        print(f"{r['strategy']:<12} path_len: fall={r['fallacy_path_length_mean']:.2f} vs valid={r['valid_path_length_mean']:.2f} ({sig_p}), "
              f"dir_chg: fall={r['fallacy_direction_changes_mean']:.1f} vs valid={r['valid_direction_changes_mean']:.1f} ({sig_d})")

# Save results
with open(os.path.join(RESULTS_DIR, "chunking_ablation.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_DIR}/chunking_ablation.json")