"""
Exp6 — train a small BPE tokenizer on the trajectory corpus.

The from-scratch encoder needs its own tokenizer (we're not reusing an
embedding model's). Train a small BPE tokenizer on the collected session text.
"""
import os, json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATA = os.path.join(os.path.dirname(__file__), "data", "sessions.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
VOCAB_SIZE = 8000


def iter_text():
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            for step in s["steps"]:
                yield step


def main():
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
    )
    tok.train_from_iterator(iter_text(), trainer=trainer)
    out = os.path.join(OUT_DIR, "tokenizer.json")
    tok.save(out)
    print(f"Trained BPE tokenizer (vocab={VOCAB_SIZE}) -> {out}")
    # quick sanity
    test = "I need to understand the task and then call the tool."
    ids = tok.encode(test).ids
    print(f"  '{test[:40]}...' -> {len(ids)} tokens")
    print(f"  decode roundtrip: {tok.decode(ids)[:60]!r}")


if __name__ == "__main__":
    main()
