# Falsification Tests — Results

## Test 1: Random label control — PASSED
- Random labels: 9.8% (nomic), 10.3% (qwen3) — both near 7.7% chance
- No data leakage. Methodology sound.

## Test 2: Binary fallacy-vs-valid — STRONG SIGNAL
- 29 valid arguments (crafted, logically sound) vs 39 fallacious examples (sampled from dataset)
- nomic-embed-text: CV Accuracy 92.4%, ROC-AUC 0.987
- The probe CAN distinguish fallacious from valid arguments, not just fallacy types from each other

## Test 3: Adversarial / trick examples — 87% (13/15)

### Valid-but-looks-fallacious: 7/8 correctly identified as valid

| # | Example | Predicted | Confidence | Status |
|---|---|---|---|---|
| 1 | Researcher falsified data → don't trust findings (looks like ad hominem, is valid) | VALID | p=0.00 | CORRECT |
| 2 | Border separation causes documented harm (looks like appeal to emotion, is evidence-based) | VALID | p=0.16 | CORRECT |
| 3 | Bachelor = unmarried man by definition (looks circular, is definitional truth) | FALLACY | p=0.83 | **WRONG** |
| 4 | Freezing → pipes burst (looks like false causality, has causal mechanism) | VALID | p=0.00 | CORRECT |
| 5 | 95% of climate scientists agree (looks like ad populum, is expert consensus) | VALID | p=0.00 | CORRECT |
| 6 | Interest rates → defaults → investment decline (looks like slippery slope, has mechanism) | VALID | p=0.00 | CORRECT |
| 7 | Operate or patient dies (looks like false dilemma, is genuine dilemma) | VALID | p=0.36 | CORRECT |
| 8 | Earth looks flat → Earth is flat (factual error, not logical fallacy) | VALID | p=0.10 | CORRECT |

### Fallacious-but-looks-valid: 6/7 correctly identified as fallacy

| # | Example | Predicted | Confidence | Status |
|---|---|---|---|---|
| 9 | Bible is true because God said so, God said so because Bible says (circular reasoning) | FALLACY | p=0.99 | CORRECT |
| 10 | Homeopathic remedy → cold cured (post hoc, cold resolves naturally) | VALID | p=0.02 | **WRONG** |
| 11 | Two rude New Yorkers → all New Yorkers rude (hasty generalization) | FALLACY | p=0.96 | CORRECT |
| 12 | Actor recommends crypto → good investment (false authority) | FALLACY | p=0.98 | CORRECT |
| 13 | Support war or hate country (false dilemma) | FALLACY | p=0.84 | CORRECT |
| 14 | "Fine for parking" → okay to park (equivocation) | FALLACY | p=0.99 | CORRECT |
| 15 | Natural things are good, poison ivy is natural (valid form, false premise) | FALLACY | p=0.89 | CORRECT |

### Where it failed

**Failure 1 (false positive):** "A bachelor is an unmarried man by definition" classified as fallacy (p=0.83). The probe detected the tautological structure ("X is Y because X is Y") and flagged it as circular reasoning. This is a known edge case — tautologies share the geometric shape of circular reasoning. The probe can't distinguish definitional truths from circular arguments.

**Failure 2 (false negative):** "I took a homeopathic remedy and my cold went away" classified as valid (p=0.02). The probe failed to detect post hoc ergo propter hoc here. The argument describes a temporal sequence without an explicit causal claim ("I took X and Y happened" rather than "X caused Y"), which may not trigger the false causality geometric signature. The reasoning is implicit.

### Multiclass classifier on trick examples

The multiclass classifier (trained only on fallacy types, no valid class) predictably classified all examples as some fallacy type. Notably, it classified valid examples as the fallacy they most resemble:
- Valid causal reasoning → "false causality" (confidence 0.73-1.00)
- Valid expert consensus → "fallacy of credibility" (0.64)
- Valid either/or → "appeal to emotion" (0.96)
- Definitional truth → "fallacy of logic" (0.99)
- Factual error → "ad hominem" (0.45)

This confirms the multiclass probe detects fallacy-shaped reasoning, not fallacy per se.

## What this means for the claim

### Claim survives:
1. **Fallacy types are separable** — confirmed (50.8% / 65.1%, random control passes)
2. **Fallacy vs valid is separable** — confirmed (AUC 0.987)
3. **Emergent across encoders** — confirmed (consistent ranking)
4. **Adversarial robustness** — mostly confirmed (87%, with clear failure modes)

### Claim is bounded by:
1. **Tautologies look like circular reasoning** — the probe conflates definitional truths with circular arguments. The geometric signature is the same.
2. **Implicit fallacies are harder to detect** — post hoc reasoning without an explicit causal claim may not trigger detection.
3. **The signal is partly stylistic** — the binary classifier trained on a small valid set (29 examples) may be detecting "reads like a textbook example" vs "reads like a real argument." Need more diverse valid arguments to rule this out.

### Falsification risks still open:
1. **Stylistic confound**: need to test with paraphrased fallacies (same logic, different surface form)
2. **Topic confound**: need to verify classification isn't driven by topic overlap
3. **Small valid set**: 29 valid arguments is too few. Need a larger, systematic valid argument set.
4. **Distribution**: the binary classifier was trained on crafted valid arguments that may share stylistic features (formal academic tone). Need more diverse valid examples.