# Module 03 — Tokenization

> Part of [Part 1 — Data](../). Reading time: ~50 minutes. Compute cost: $0 (CPU-only).

## The thesis

A tokenizer is a *learned compression scheme* that turns bytes into integers your model can do math on. The choices you make here — vocabulary size, what gets a dedicated token, how digits and whitespace are handled, what languages are represented — quietly constrain *every* decision after it:

- **Embedding table** dimensions (`vocab_size × d_model`).
- **Output projection** dimensions (often tied to the embedding).
- **Sequence length budget** — fewer tokens per document means longer effective context, or shorter sequences for the same training compute.
- **What domains and languages the model can represent at all** — if your tokenizer was trained on English, Chinese text shreds into individual UTF-8 bytes and the model learns terribly on it.
- **Hardware accounting** — vocab > 65k forces `uint32` token IDs and doubles the size of your pre-tokenized corpus on disk (see [Module 02](../02-the-corpus/#how-much-disk-do-you-actually-need)).

The tokenizer is also the place where a surprising fraction of "model is dumb" failures actually live. Glitch tokens, digit fragmentation, leading-space inconsistencies — they're all here, and they're all observable before you train a single step.

## What you'll be able to do at the end

- Implement BPE from scratch in ~80 lines of Python.
- Train a real byte-level BPE tokenizer on a corpus shard using HuggingFace's `tokenizers` library.
- Measure tokenizer compression rate (characters per token) on English, code, and Chinese text.
- Diagnose digit-tokenization and whitespace-handling failure modes.
- Pick a vocabulary size for a given (model size, language mix, deployment cost) and defend the choice.
- Use [`tokenizer.py`](tokenizer.py) as the canonical tokenizer interface for every later training script.

## Why we need a tokenizer at all

Three alternatives, ruled out:

| Approach | Problem |
|---|---|
| **Character-level** | A 1000-char sentence is a 1000-token sequence. Attention is `O(n²)`. Sequences are ~5× longer than BPE. Pretty much abandoned. |
| **Word-level** | Closed vocabulary. Out-of-vocabulary words → `<UNK>` → information loss. Brittle to typos, neologisms, multilingual text, code. Used in 2014; not in 2026. |
| **No tokenizer at all (byte-level)** | Works (every byte is a token, vocab = 256), but compression is so bad that sequences become 4× longer than BPE for English. Tried by ByT5; the loss in compute efficiency was rarely worth the gain in robustness. |

BPE — Byte-Pair Encoding — splits the difference. It starts at the byte level (so it can represent any string, no UNKs) and *learns* longer tokens for the patterns that show up in your corpus. Common words become single tokens; rare words decompose into subword pieces; truly novel strings still encode as raw bytes. Best of all three worlds.

## BPE in one page

The algorithm:

```
1. Start with the 256 single-byte tokens (the vocabulary).
2. Encode the corpus as a sequence of byte IDs.
3. Find the most frequent adjacent pair (a, b) in the encoded corpus.
4. Add a new token with ID 256 + i, meaning "a followed by b".
5. Replace every occurrence of the pair (a, b) with the new token ID.
6. Repeat from step 3 until you have `vocab_size` tokens total.
```

That's it. The full implementation in [`bpe_from_scratch.py`](bpe_from_scratch.py) is ~80 lines including the encode/decode logic.

A worked example on the string `"low low low lower newer newest"`:

- Initial tokens: each byte. The pair `(' ', 'l')` (space-l) is among the most frequent.
- Merge step 1: ` l` becomes a single token. Now `low low low lower` is `[" l", "o", "w", " l", ...]`.
- Subsequent merges find `("o", "w")` → `ow`, then `(" l", "ow")` → ` low`, and so on.
- After enough merges, `"low"` (or ` low`) is a single token; rare suffixes like `est` are also single tokens.

**Byte-level BPE** (GPT-2, Llama, Qwen, DeepSeek — all of them) means the base alphabet is the 256 byte values, not Unicode code points or characters. This means:

- *Any* string is encodable, with no UNK token. Even arbitrary binary data round-trips.
- Multi-byte UTF-8 characters (Chinese, emoji, accents) decompose into their constituent bytes if the BPE didn't merge them during training. Whether this is good or bad depends on whether your training corpus had enough of that script for BPE to learn longer merges (the multilingual question, below).

### Encoding a new string

To tokenize new text, you greedily apply the same merges in priority order (the order they were learned during training). The training process produces a `merges` list; at inference time you apply them. [`bpe_from_scratch.py`](bpe_from_scratch.py) shows the algorithm; in practice you use a high-performance Rust implementation via the `tokenizers` library.

## The vocab-size tradeoff

This is the single most consequential tokenizer decision, and there is no "correct" answer — only tradeoffs you should be able to articulate.

**Bigger vocabulary →**

- **Better compression** — common words and phrases collapse to single tokens. Fewer tokens per document. Either shorter sequences (cheaper to train) or more effective context length (more text fits in your context window).
- **Larger embedding and output-projection layers** — both are `vocab_size × d_model`. At `d_model=2048`:
  - 50k vocab → 102M params per layer
  - 128k vocab → 262M params per layer
  - 152k vocab → 311M params per layer
  These are often *tied* (the same matrix used for input embedding and output projection), so it's one cost, not two — but it's still material.
- **Harder cross-entropy** — the model predicts a softmax over more classes. Loss numbers are not directly comparable across tokenizers.
- **Disk doubles past 65k vocab** — `uint16` no longer fits; you need `uint32` for stored token IDs. (Module 02 has the storage math.)
- **More room for multilingual / domain coverage** — tokens for Chinese characters, math notation, code constructs all eat into the budget.

**Smaller vocabulary →**

- More tokens per document, longer sequences, more compute per epoch.
- Smaller embedding tables — possibly meaningful at the small-model end. For a 100M-param model, an 80M embedding table is a problem; for a 70B model, it's noise.
- Worse multilingual coverage by default.

**Empirical sweet spots in 2026:**

| Model family | Vocab | Notes |
|---|---|---|
| GPT-2 (2019) | 50,257 | The original byte-level BPE choice. English-centric. |
| Mistral 7B (2023) | 32,000 | Small, fast, English+European. Notable lower end. |
| Llama 3 (2024) | 128,256 | Bigger than Llama 2's 32k. Multilingual + code reflective. |
| DeepSeek-V3 (2024) | 129,280 | Chinese + English, similar reasoning to Llama 3. |
| Qwen 3 (2025) | 151,936 | Largest among major frontier models. Heavy Chinese coverage. |

The trajectory is clear: vocabs have grown over the past few years, primarily to accommodate non-English coverage and code. They're not growing much *past* ~150k because the embedding-table cost is starting to be felt at smaller model sizes and the compression gains diminish.

**For this course's pretraining demo** (a ~125M-param model on FineWeb-Edu), 32k is the right call: keeps the embedding table to ~20M params (a sane fraction of the model), compresses English well enough that 1B tokens is a meaningful amount of text, and trains in minutes on a single GPU.

## The multilingual question

English compresses to about **4 characters per token** with a well-trained 32–50k BPE. Chinese, in raw UTF-8 bytes, is *3 bytes per character* — so naive byte-level BPE on a corpus with little Chinese gives you maybe **1 character per token** for Chinese text. That's a 12× tokenization efficiency gap.

Concretely: a 1000-character Chinese document might use ~1000 tokens with an English-trained tokenizer and ~250 tokens with a Chinese-aware one. For a Chinese user, training on the English-trained tokenizer means 4× longer sequences, 4× the inference cost, and a model that's pre-disadvantaged on their language.

What frontier Chinese-trained models do (Qwen, DeepSeek):

- **Upweight Chinese / CJK content during BPE training.** If your BPE training corpus has 30% Chinese text, you'll get a lot of Chinese-character tokens. If it has 0.1%, you won't.
- **Reserve explicit Chinese-character tokens.** Some recipes pre-allocate single-token entries for all the common CJK characters (a few thousand of them) before BPE starts merging. Guarantees a floor of compression.
- **Use a larger vocab.** Qwen 3's 152k is partly to make room for Chinese-character coverage without sacrificing English compression.

The general principle: **a tokenizer is biased toward whatever was in its training corpus.** If you're training a model that will see multilingual text or code or math or chemistry, your BPE training corpus needs to include those in roughly the proportion you care about them — or you'll pay a compression tax forever after.

The notebook measures this gap empirically.

## Pretokenization: the regex you forgot was there

Before BPE training starts, the corpus is split into chunks by a regex. This is called *pretokenization*, and it determines what BPE is allowed to merge across.

GPT-2's regex (still widely used as a starting point):

```regex
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

It splits on: contractions (`'s`, `'t`, etc.), runs of letters (optionally preceded by a space), runs of digits, runs of punctuation, and whitespace. BPE then learns merges *within* these chunks but never *across* them — so you'll never get a single token spanning a word boundary into the next word.

Why this matters:

- **`" hello"` and `"hello"` are tokenized differently.** The leading-space case is a single token (the regex grabs the leading space with the word); the no-space case starts mid-chunk. Same word, different tokens. This is the source of *many* subtle prompting bugs.
- **Numbers.** `"1234"` is a single run-of-digits chunk, so BPE is allowed to merge `12`, `34`, `1234` into single tokens if they're frequent enough. This is bad for arithmetic — the model has to memorize that "two-hundred-and-fifty-six" is the same concept as the token "256". **Modern tokenizers (Llama 3, DeepSeek) force digit splitting** so every digit is its own token. We do the same in [`train_bpe.py`](train_bpe.py).
- **Code.** Tabs and spaces become their own tokens. Indentation patterns get learned. A tokenizer trained on prose handles code badly; one trained on code+prose handles both.

The pretokenizer is also where a tokenizer can be biased toward your domain. There's a `regex` for code (split on camelCase), a `regex` for math (don't split numbers), a `regex` for chat (preserve `<|im_start|>`-style tokens). Most labs use minor variations on GPT-2's regex with digit splitting added.

## Special tokens

A small set of reserved tokens at the top of the vocabulary, used for control flow rather than content. The common ones:

| Token | Purpose | Where it appears |
|---|---|---|
| `<|endoftext|>` (BOS/EOS) | Start/end of document, sequence separator | Every training sample, every generation |
| `<|pad|>` | Padding for fixed-length batches | Padded shorter sequences in a batch |
| `<|im_start|>`, `<|im_end|>` | Chat turn boundaries (ChatML format) | Post-training / SFT (Module 13) |
| `<|user|>`, `<|assistant|>`, `<|system|>` | Role markers in chat | Post-training |
| Tool / function-calling tokens | Various; format depends on the model | Tool-using models |

Two things to know:

1. **Reserve them at the *top* of the vocab.** The IDs `0, 1, 2, ...` for special tokens; everything else shifts up. Makes them easy to identify and lets you safely add more later by extending the embedding table.
2. **Pretraining usually only needs `<|endoftext|>`.** Chat tokens come in during post-training. Reserving the IDs now (even if unused) saves you embedding-table surgery later.

We reserve a handful in [`train_bpe.py`](train_bpe.py).

## Decisions you're making (the cheat sheet)

| Decision | This course | Frontier reference |
|---|---|---|
| Vocab size | 32k | 128–152k |
| Pretokenizer | GPT-2 regex + digit splitting | Same; minor variations |
| Special tokens | `<|endoftext|>`, `<|pad|>` + 8 reserved for post-training | ~10–30, mix of fixed + reserved |
| Training corpus for BPE | FineWeb-Edu sample (~100k docs) | Sample of the full pretraining mix |
| Implementation | HF `tokenizers` (Rust under the hood) | Same |
| Storage type for token IDs | `uint16` (32k < 65k) | `uint32` (vocab > 65k) |

## Failure modes you should recognize

These are real bugs that have shipped in real models. Knowing what they look like is enough to debug them later.

**Glitch tokens.** Tokens that were in the BPE vocab but barely appeared in pretraining — the model has essentially never seen them in context, so generating one produces nonsense. The famous GPT-2/3 example is `" SolidGoldMagikarp"`, a Reddit username that got into the BPE training corpus but not the LM training corpus. Symptom: the model generates wild output when prompted with the token. Avoidance: training BPE and the LM on the same corpus (which we do).

**Digit fragmentation without splitting.** A tokenizer where `"1234"` is one token, `"1235"` is one token, but `"4321"` is three tokens (`4`, `32`, `1`). Arithmetic accuracy collapses because the model can't see digit positions consistently. Avoidance: force digit splitting (we do).

**Whitespace surprises.** `"hello"` and `" hello"` are different tokens. `tokenize("Hello, world")` gives different IDs depending on whether there's a leading space. Many chat-template bugs are this. Avoidance: be deliberate when constructing input strings; pay attention to BOS handling.

**Multilingual cliff.** Tokenizer trained on 99% English; user sends Chinese; sequences become 4× longer; inference cost balloons; quality drops. Avoidance: train BPE on the corpus mix you'll actually see at inference.

**Domain mismatch.** Tokenizer trained on web prose; you fine-tune on Python code; code documents are 2× longer in tokens than they should be; SFT becomes more expensive. Avoidance: same as above — the BPE corpus should look like the LM corpus.

## The `tokenizer.py` API

```python
from tokenizer import load_tokenizer

tok = load_tokenizer()  # loads results/tokenizer.json
ids = tok.encode("hello world").ids       # -> list[int]
text = tok.decode(ids)                    # -> "hello world"
print(tok.get_vocab_size())               # -> 32000
```

This is the contract every later module uses. `train_bpe.py` writes to `results/tokenizer.json`; `tokenizer.py` reads from there. The notebook walks through both ends.

## Next

[Part 2 — Architecture](../../part-2-architecture/). Tokens are now integers your model can do math on. The next question is *what math*: attention, MLPs, normalization, the transformer block, MoE, and how to assemble them into something that runs.
