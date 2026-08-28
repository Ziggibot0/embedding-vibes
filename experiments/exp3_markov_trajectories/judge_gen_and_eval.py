"""Study B judge-labeler + generator for INDEPENDENT OOD test.

Study A (gen_bigger.py) is a controlled, prompt-injection study. It can only
prove the Markov chain learns the *mechanical signature of our fallacy-injection
prompt*, which is still overfit-to-us. It cannot tell us the chain understands
logic.

Study B exists to defeat that. It generates *genuine* fallacious reasoning (no
"you must commit X fallacy" instructions) and lets an independent judge
(separate model, or same model in off-label mode) tag the reasoning as
fallacy / valid. That removes both the prompt-style leak AND the injection-sign
leak.

Run:
    python gen_bigger.py        # Study A: controlled, in-distribution
    python build_mc.py
    python predict.py           # Study A: does not overfit because of topic split

    python judge_gen_and_eval.py  # Study B: OOD independent labels
AUC on OOD sessions is real generalization. AUC on in-distribution is the
controlled comparison. The gap between them is overfitting-to-us.

If the gap is large, the chain is relying on procedural cues, not logic.
If the gap is small, the chain has learned something more abstract. That
is the number to report as "is it overfit to our prompt style?"
"""
import json, os, asyncio, time
from urllib.request import Request, urlopen
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OLLAMA_URL = "http://localhost:11434/api"


# Independent judge model — distinct from Study A's qwen3.5:4b generator.
JUDGE_MODEL = "deepseek-v4-flash:cloud"
GENERATOR_MODEL = "qwen3.5:4b"


async def _chat(prompt, model, temperature=0.6, timeout=120):
    payload = json.dumps({"model": model, "stream": False,
                          "options": {"temperature": temperature},
                          "messages": [{"role": "user", "content": prompt}]}).encode()
    req = Request(f"{OLLAMA_URL}/chat", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def parse_steps(text, n_steps):
    if not text:
        return []
    steps, lines = [], text.split("\n")
    for i in range(1, n_steps + 1):
        p = f"Step {i}:"
        for line in lines:
            s = line.strip()
            if s.startswith(p):
                steps.append(s[len(p):].strip())
                break
    if len(steps) < 3:
        steps = [line.strip() for line in lines if line.strip()][:n_steps]
    return steps


def judge_prompt(topic):
    return (
        "You are a careful logic judge. Read 6 reasoning steps about a topic.\n"
        "Rate only whether the REASONING STRUCTURE contains a logical fallacy\n"
        "(e.g. non-sequitur, false cause, equivocation, ad hominem, circular). "
        "Answer with exactly two tokens: 'fallacy' or 'valid'. Do not explain.\n\n"
        f"Topic: {topic}\n\nSteps:\n"
    )


async def judge(text, topic):
    p = judge_prompt(topic) + "\n".join(f"{i+1}. {s}" for i, s in enumerate(text))
    for _ in range(3):
        try:
            out = (await _chat(p, JUDGE_MODEL))["message"]["content"].strip().lower()
            return "fallacy" if "fallacy" in out else "valid"
        except Exception:
            await asyncio.sleep(2)
    return "unknown"


async def gen_one(idx, topic, n_steps=6, max_try=3):
    p = (
        f"Explain your position on '{topic}' in exactly {n_steps} numbered steps, "
        "each 1-3 sentences, Step 1: setup, Step 2: build on step 1, "
        "Step 3: a conclusion, Step 4: add another supporting idea, "
        "Step 5: acknowledge a counterargument, "
        "Step 6: final assessment. Output ONLY the numbered steps, one per line 'Step N:'\n"
    )
    for _ in range(max_try):
        try:
            out = (await _chat(p, GENERATOR_MODEL))["message"]["content"] or ""
            steps = parse_steps(out, n_steps)
            if len(steps) >= 3:
                return steps, topic
        except Exception:
            await asyncio.sleep(2)
    return [], topic


async def judge_batch(rows):
    sem = asyncio.Semaphore(8)
    async def j(r):
        async with sem:
            return (await judge(r["steps"], r["topic"]), r)
    return asyncio.gather(*(j(r) for r in rows))


def eval_studyB(all_rows):
    from sklearn.metrics import roc_auc_score, accuracy_score
    y = np.array([1 if r["judge"] == "fallacy" else 0 for r in all_rows])
    score_llr = np.array([r["score_llr"] for r in all_rows])
    pred_llr = (score_llr > 0).astype(int)
    acc = accuracy_score(y, pred_llr)
    auc = roc_auc_score(y, score_llr)
    print(f"\nStudy B results: {len(all_rows)} sessions | AUC={auc:.3f} | accuracy={acc:.3f}")
    print("Study B AUC is overfitting-to-us gate: higher = better generalization")


async def main():
    # Topic pool — genuine, debated, no injected instruction.
    topics = [
        "UBI", "carbon tax", "vaccines", "nuclear power", "remote work",
        "student debt forgiveness", "gun control", "universal healthcare",
        "cryptocurrency regulation", "AI in hiring", "4-day work week",
        "immigration reform", "4-year college cost", "prison reform",
        "climate geoengineering", "social media for minors", "minimum wage",
        "genetically engineered food", "nuclear weapons abolition",
        "space colonization priorities", "data privacy law", "4-day work week",
        "cryptocurrency", "four-day work week", "cannabis legality",
        "school funding", "broadband access", "AI sentencing", "transparency",
    ]
    per_type = 15  # 15 each, mixed fallacy/valid targets
    all_rows = []
    half = per_type // 2

    # Generate + (parallel) judge fallacy reasoning and valid reasoning.
    print("[1] generating fallacy-style reasoning...")
    fall_sessions = await asyncio.gather(*(gen_one(idx, topics[i % len(topics)]) for i, idx in
                                          enumerate(range(per_type))))
    for steps, topic in fall_sessions:
        all_rows.append({"steps": steps, "topic": topic, "target": "fallacy"})
    print("[2] generating valid reasoning...")
    valid_sessions = await asyncio.gather(*(gen_one(idx, topics[i % len(topics)]) for i, idx in
                                            enumerate(range(per_type, per_type*2))))
    for steps, topic in valid_sessions:
        all_rows.append({"steps": steps, "topic": topic, "target": "valid"})

    json.dump(all_rows, open(os.path.join(RESULTS_DIR, "studyB_sessions.json"), "w"))
    print("[3] judging...", flush=True)
    judged = await judge_batch(all_rows)
    for (jg, r) in judged:
        r["judge"] = jg
        r["raw_judge"] = r["target"]
        r["agreed"] = (jg == r["target"]) if jg in ("fallacy", "valid") else None
    json.dump(all_rows, open(os.path.join(RESULTS_DIR, "studyB_judged.json"), "w"))

    eval_studyB(all_rows)


if __name__ == "__main__":
    asyncio.run(main())
