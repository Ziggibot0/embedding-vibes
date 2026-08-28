# Prior Art Audit — Has This Been Done Before?

## What we did
Linear probe (LogisticRegression) on frozen embeddings from two different encoder families (nomic-embed-text 137M, qwen3-embedding 7.6B) to classify 13 logical fallacy types from the Jin et al. (2022) LOGIC dataset. Found 50.8% and 65.1% multiclass accuracy (6.6x and 8.5x chance), with consistent per-fallacy AUC rankings across both encoders.

## Has anyone done exactly this? NO.

### What exists in fallacy detection (closely related work):

1. **Jin et al. (2022) — Logical Fallacy Detection (EMNLP Findings)**
   - Used fine-tuned pretrained LMs (BERT, RoBERTa, Electra, DeBERTa, etc.) on the LOGIC dataset
   - Best fine-tuned model: Electra at 53.31% F1
   - Their structure-aware classifier (NLI-based with logical form matching) achieved 58.77% F1
   - They used Sentence-BERT embeddings for COSINE SIMILARITY MATCHING to identify paraphrase spans for masking — but NOT as a classification probe
   - **They did NOT test frozen embeddings with a linear probe.** Their baselines all involved fine-tuning the encoder.

2. **Sourati et al. (2023) — Case-Based Reasoning with Language Models (IJCAI)**
   - Used LM embeddings for RETRIEVAL (cosine similarity to find similar past cases), then fine-tuned an attention-based adapter
   - The embeddings were used for similarity-based retrieval, not direct classification
   - **Not a linear probe on frozen embeddings.**

3. **Lei & Huang (2024) — Boosting Logical Fallacy Reasoning via Logical Structure Tree (EMNLP)**
   - Built logical structure trees and incorporated them into LLMs via hard/soft prompts
   - Used RoBERTa embeddings as part of the tree construction, but the classification head was trained with backprop
   - **Not a frozen-encoder linear probe.**

4. **Alhindi et al. (2024) — Are LLMs Good Zero-Shot Fallacy Classifiers? (EMNLP)**
   - Tested GPT models with various prompting strategies (zero-shot, CoT, multi-round)
   - No embedding-based classification at all — purely prompt-based
   - **Not embedding-based.**

5. **Beyond Logical Forms (2026, ArgMining)**
   - Used embeddings for dynamic one-shot retrieval (all-MiniLM-L6-v2, cross-encoder/stsb-roberta)
   - Retrieved similar examples, then prompted LLM with them
   - Explicitly states: "no prior work has explored fallacy classification from a structural perspective without any additional fine-tuning"
   - But their approach uses LLM prompting with retrieved examples, not a linear probe on frozen embeddings
   - **Closest in spirit (no fine-tuning), but different method (prompting vs probing).**

6. **LCF — Logic Control Framework (AAAI 2025)**
   - Disentangled LLM hidden representations into content and logic spaces
   - Used contrastive learning to separate valid/invalid logic regions
   - Showed t-SNE visualizations of valid vs invalid logic space separation
   - **This is the closest to "geometric separability of fallacy structure"** — BUT:
     - They use TRAINED projectors (not frozen embeddings + linear probe)
     - They separate valid vs invalid (binary), not fallacy TYPES (13-class)
     - They modify representations (steering), not just probe them
     - They don't test multiple encoders for emergent property

7. **LoFa Benchmark (ACL 2026)**
   - Used LogiLens to visualize internal activation trajectories during fallacy attacks
   - Found a "cognitive void" phase in middle layers when models encounter fallacies
   - **Related to trajectory geometry but very different** — they study how LLMs process fallacious input, not whether fallacy types are separable in embedding space

8. **MALTO at FadeIT (EVALITA 2026)**
   - Fine-tuned BERT (AlBERTo) for multi-label fallacy detection in Italian
   - Used mean pooling + classification head — but with fine-tuning
   - **Not frozen embeddings.**

9. **Multimodal Fallacy Classification (ArgMining 2025)**
   - Used RoBERTa + audio features with logistic regression meta-classifier
   - Fine-tuned text encoder + logistic regression on concatenated features
   - **Fine-tuned, not frozen.**

### What exists in linear probing (the method, applied to other tasks):

- Linear probing on frozen representations is a well-established methodology (Alain & Bengio 2017)
- Extensively used in vision (DINO, CLIP), NLP (BERT hidden states), and multimodal
- Microsoft (ACL 2026) used linear probes on hidden states to predict reasoning correctness (AUC 0.87) — but for math reasoning, not fallacy detection
- "Do transformers notice their own mistakes?" (Baseten) used linear probes to detect hallucination — but for hallucination, not fallacy type classification
- No prior work applies linear probing specifically to fallacy type classification on frozen embeddings

### What exists in embedding geometry for reasoning:

- "Geometry of Reasoning: Flowing Logics" (2025) — showed LLMs internalize logical structure as geometry, but used hidden state trajectories, not frozen embeddings
- CraEG (ICML 2026) — embedding-space crowding correlates with reasoning failure, but studied decoding-time token crowding, not fallacy type separability
- "Reasoning emerges from constrained manifolds" (2026) — studied manifold structure of reasoning, but not fallacy-specific
- No prior work specifically studies the geometric separability of fallacy types in embedding space

## Conclusion: This is novel.

Nobody has:
1. Applied a linear probe to frozen embeddings specifically for fallacy type classification
2. Tested whether fallacy types are geometrically separable across multiple encoder families
3. Produced a per-fallacy-type AUC ranking showing which fallacies are easy/hard to detect geometrically
4. Demonstrated the emergent property (consistent ranking across unrelated encoders)

The closest work is:
- Jin et al. (2022): used fine-tuned models, not frozen probes
- LCF (AAAI 2025): showed valid/invalid separation with trained projectors, not frozen probes, and binary not multiclass
- Beyond Logical Forms (2026): no fine-tuning but used LLM prompting, not embedding probing

Our specific contribution: **showing that fallacy type is linearly decodable from frozen embeddings without any fine-tuning, across two unrelated encoder families, with a consistent per-type difficulty ranking — an emergent geometric property of how fallacies manifest in representation space.**

This is a clean, novel finding that doesn't step on anyone's toes.