# Part 1 — Data

Most courses start with the model. The best researchers start with the data.

The single most reliable predictor of how good your final model will be is the quality of your pretraining corpus. Architecture choices give you maybe 10–20% on benchmarks. Data choices give you multiples. A small model on a clean, well-mixed corpus will outperform a much larger model on noisy web scrapes, every time.

This Part is short because the techniques are simple. The reason it comes before architecture is more important than the content itself: **data is a design decision, not preprocessing**.

## Modules

- **[02 — The Corpus](02-the-corpus/)** — What makes a good pretraining corpus. FineWeb-Edu walkthrough. Deduplication, filtering, mixing. The Chinchilla lesson (data quantity vs model size). Streaming dataloaders that don't bottleneck training.
- **[03 — Tokenization](03-tokenization/)** — BPE from scratch. Why vocabulary size is a tradeoff, not a default. What Chinese-language tokenizers do differently and why. Tokenizer choice as a constraint on everything downstream.
- **[03a — Curating SFT Data](03a-curating-sft-data/)** *(extra, optional)* — A 5-stage filter cascade for cleaning machine-translated SFT data, walked through on `DeepMount00/OpenItalianData` (2.14M rows → ~50–150k high-quality Italian). Covers length / langid / artifact / dedup / diversity / LLM-as-judge, plus a held-out eval set methodology and HF dataset publishing. Skip on first pass; come back when you want to understand how Module 15's input is actually built.

## What you'll be able to do at the end of this Part

- Build a streaming data pipeline from a real dataset (FineWeb-Edu) to tokenized training batches.
- Train your own BPE tokenizer and recognize its failure modes.
- Make an informed decision about vocab size for a given model and corpus.
- *(03a)* Take an aggregated machine-translated SFT dataset and produce a clean, dedup'd, diversity-balanced subset with a full drop-reason audit trail.

## Time and cost

- Reading + light coding: ~3 hours (~4 with 03a).
- Compute cost: ~$0. CPU-only. The point of this Part is that data work happens before GPU work. (Module 03a's optional stage 5 — LLM-as-judge — is the one exception; ~$5–15 with a small API, or $0 with a local GPU.)
