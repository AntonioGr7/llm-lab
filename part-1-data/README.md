# Part 1 — Data

Most courses start with the model. The best researchers start with the data.

The single most reliable predictor of how good your final model will be is the quality of your pretraining corpus. Architecture choices give you maybe 10–20% on benchmarks. Data choices give you multiples. A small model on a clean, well-mixed corpus will outperform a much larger model on noisy web scrapes, every time.

This Part is short because the techniques are simple. The reason it comes before architecture is more important than the content itself: **data is a design decision, not preprocessing**.

## Modules

- **[02 — The Corpus](02-the-corpus/)** — What makes a good pretraining corpus. FineWeb-Edu walkthrough. Deduplication, filtering, mixing. The Chinchilla lesson (data quantity vs model size). Streaming dataloaders that don't bottleneck training.
- **[03 — Tokenization](03-tokenization/)** — BPE from scratch. Why vocabulary size is a tradeoff, not a default. What Chinese-language tokenizers do differently and why. Tokenizer choice as a constraint on everything downstream.

## What you'll be able to do at the end of this Part

- Build a streaming data pipeline from a real dataset (FineWeb-Edu) to tokenized training batches.
- Train your own BPE tokenizer and recognize its failure modes.
- Make an informed decision about vocab size for a given model and corpus.

## Time and cost

- Reading + light coding: ~3 hours.
- Compute cost: ~$0. CPU-only. The point of this Part is that data work happens before GPU work.
