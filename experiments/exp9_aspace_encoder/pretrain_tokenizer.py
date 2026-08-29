"""Pre-train tokenizer on the big 60k corpus (CPU, no GPU needed).

Uses the same MASK token string as train.py's get_tokenizer — the byte
0xF0 0x9F 0xA7 0x90 (the emoji that train.py uses as its mask token).
"""
import json, time, os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

HERE = os.path.dirname(os.path.abspath(__file__))
corpus = os.path.join(HERE, "data", "train_corpus.jsonl")
out = os.path.join(HERE, "results", "tokenizer.json")
if os.path.exists(out):
    os.remove(out)

# Match train.py exactly: same special tokens, same vocab size
MASK = "\U0001F9D0"
specials = ["[PAD]", "[UNK]", "[CLS]", MASK]

print("training BPE on 60k corpus ...")
t0 = time.time()
tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=specials,
                              min_frequency=2)


def iter_text():
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            for step in s["steps"]:
                yield step


tok.train_from_iterator(iter_text(), trainer=trainer)
tok.save(out)
print(f"tokenizer trained in {time.time()-t0:.0f}s -> {out} ({tok.get_vocab_size()} vocab)")

test = "I need to understand the task and then call the tool."
ids = tok.encode(test).ids
print(f"  '{test[:40]}...' -> {len(ids)} tokens")
print(f"  decode roundtrip: {tok.decode(ids)[:60]!r}")