# Module 02 — The Corpus

> Part of [Part 1 — Data](../). Reading time: ~45 minutes. Compute cost: $0 (CPU-only, streaming over the network).

## The thesis

**A small model on a clean, well-mixed corpus will outperform a much larger model on noisy web scrapes — every time.**

Architecture gives you maybe 10–20% on benchmarks at fixed compute. Data gives you multiples. This isn't a research opinion; it is the lesson every frontier lab has had to relearn the hard way. The question "what's in the corpus, and in what proportion?" is the single highest-leverage decision in pretraining.

Data is not preprocessing. It is a *design decision*, made by the same people deciding the model size, with the same care.

## What you'll be able to do at the end

- Articulate, in concrete terms, what makes a pretraining corpus good or bad.
- Stream FineWeb-Edu (or any HuggingFace dataset) into a training loop without it bottlenecking the GPU.
- Filter and shard a dataset across multiple ranks for distributed training, reproducibly.
- Recognize the failure modes of deduplication, language filters, and quality classifiers — and know which knobs to turn.
- Use the [`corpus.py`](corpus.py) module as the data layer for every later training script in this course.

## What makes a corpus good

Five properties, roughly in order of impact.

### 1. Provenance — where did the text actually come from?

A corpus is its sources. Common Crawl is the web, warts and all. Books and academic papers are clean but narrow. Code is structured but full of boilerplate. Conversational data is fluent but full of low-information chit-chat. *Knowing the provenance is non-negotiable*, because every downstream filter exists to compensate for some property of the source.

Frontier labs in 2026 typically blend:
- **Filtered web** (60–80%) — Common Crawl, passed through quality classifiers and language filters. The bulk.
- **Code** (5–15%) — GitHub-style, deduplicated, license-filtered. Heavily upweighted for reasoning gains even in non-code tasks.
- **Books / academic** (5–15%) — long-form, well-edited. Punches above its weight on perplexity benchmarks.
- **Math / synthetic** (1–10%) — fast-growing category; many 2025–2026 papers attribute reasoning gains here.
- **Multilingual** (variable) — depends on the target locale mix.

You won't reproduce this exactly. You'll use one well-curated open corpus (FineWeb-Edu) that already represents the "filtered web" slice done right. That's enough to learn the lesson.

### 2. Scale — how many tokens?

Not "how many GB". *Tokens*. After tokenization. Modern pretraining is measured in **trillions of tokens**: DeepSeek-V3 was trained on 14.8T; Llama 3 405B on ~15T; smaller frontier models routinely see 5–10T. You will not pretrain on trillions of tokens for $50. You will pretrain on ~100M–1B tokens, just enough to see the dynamics, then read the literature to understand how they scale up.

The exact number of tokens you need depends on the model size — see the Chinchilla section below.

### 3. Quality — does the text actually contain something worth learning?

A corpus full of SEO spam, navigation bars, machine-translated junk, and copy-pasted product descriptions teaches the model to *predict SEO spam*. Quality filters exist to remove this. The state of the art has shifted dramatically:

- **Heuristic filters** (line lengths, punctuation ratios, language detection) — necessary but coarse. The Gopher rules from 2021 are still used as a first pass.
- **Classifier filters** (a small model that scores each document for "educational quality" or "high-information content") — the FineWeb-Edu approach. This is the biggest single quality lever discovered in the last few years.
- **Perplexity filters** (using a reference model's surprise to flag outliers) — useful but model-dependent.

Quality filtering throws away 80–95% of raw Common Crawl. That's normal.

### 4. Diversity — does it cover the space?

A corpus that's 90% English news from 2019 will produce a model that's great at English news from 2019 and bad at everything else. Diversity is measured along several axes — domain (web/code/books/math), language, time, register (formal/casual/technical). You can't measure diversity precisely, but you can do *domain ablations*: train two small models on different mixes, evaluate, pick the better mix. This is how labs actually tune their data recipe.

### 5. Deduplication — is the same thing in there 50 times?

Web scrapes have brutal duplication. The same article republished on a hundred mirrors. The same boilerplate footer on every page. Near-duplicates from translation, paraphrase, or templating. Deduplication matters because:

- **Exact duplicates** waste training compute on tokens the model has already seen.
- **Near-duplicates** can cause memorization, which hurts generalization and creates copyright/privacy risk.
- **Boilerplate** (cookie banners, navigation, footer text) is the worst kind of near-dup: it pulls the model toward predicting nav menus.

The standard approach is two-stage: exact dedup (hash every document, drop matches) followed by near-dup detection via **MinHash** + **Locality-Sensitive Hashing**. MinHash gives you a quick approximate Jaccard similarity score; LSH lets you cluster billions of documents in reasonable time. FineWeb-Edu has both already applied — you'll just need to recognize what was done and why.

## Why FineWeb-Edu

Three reasons.

1. **It's the best-documented open web corpus that resembles what frontier labs use.** Released by HuggingFace in 2024, refined since. ~1.3T tokens. Source: Common Crawl, filtered with a Llama-3-based educational-quality classifier.
2. **The work is reproducible.** The filtering pipeline, the classifier weights, the dedup recipe — all published. You can see exactly what was thrown away and why. This is rare.
3. **It's available via `datasets.load_dataset(..., streaming=True)`.** No 5TB download. You stream documents over the network as you train.

What we are *not* doing in this course: building our own corpus from raw Common Crawl. That's a multi-week, multi-CPU project. The lesson — *that* the filters and dedup matter, what they look for, what they break — is fully transferable to FineWeb-Edu as the substrate.

## The Chinchilla lesson

This is the most-misunderstood result in pretraining. Read carefully.

DeepMind's 2022 *Chinchilla* paper asked: given a fixed compute budget, what is the optimal trade-off between model size and number of training tokens? They found a roughly *equal scaling* relationship: doubling the parameters means roughly doubling the tokens. The rule of thumb people remember is "**20 tokens per parameter is optimal**".

This number is for a specific question — "minimize loss at fixed training compute". It is *not* an upper bound. Modern labs routinely train far past Chinchilla-optimal (200+ tokens per parameter) because:

- **Inference cost dominates training cost** at scale. A model trained 10× longer than Chinchilla-optimal is slightly worse per training-FLOP, but the same size at inference. Cheaper to deploy.
- **Loss isn't capability.** Past Chinchilla-optimal, loss continues to drop and downstream capabilities continue to improve, even if FLOP-per-loss-bit efficiency degrades.

The decision Chinchilla informs:

| Situation | Rule of thumb |
|---|---|
| Research / one-off / measuring loss | Train near Chinchilla-optimal (~20 tokens/param). |
| Production model you'll deploy | Train 5–20× past Chinchilla-optimal. Inference is cheaper than retraining. |
| You're learning (this course) | Train enough tokens that the loss curve is clearly past its initial drop and into the slow grind phase. Roughly ~10 tokens per parameter is plenty to see the right shape. |

For our pretraining demo in Module 11, a small model (~125M params) trained on ~1B tokens gives you a textbook loss curve at a textbook price. You will not match DeepSeek's perplexity, and you don't need to — the *shape* of the run is what we're after.

## Streaming, the only sane way to handle this much data

You cannot load 1.3T tokens of FineWeb-Edu into RAM. You cannot even load 1B tokens into RAM. Tokenized at ~5 bytes per token, that's 5GB just for the encoded form; raw text is roughly 10× that.

You also can't pre-download it all to disk for a small course — it's terabytes.

**Streaming** solves this. `datasets.load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)` returns an iterable that fetches documents on demand from HuggingFace's CDN. You apply filters, tokenization, and batching on the fly. The training loop sees a smooth flow of tokens; the network and CPU stay busy in the background, prefetching the next shard.

The two things that go wrong:

1. **Network bottlenecks the GPU.** A loaded A100 wants ~1–2 GB/sec of tokens. If your pod's network is slow or the shuffle buffer is too small, the GPU waits. Solved by larger shuffle buffers, more dataloader workers, or pre-staging shards to local disk for the duration of the run.
2. **Shuffle quality.** Streaming shuffles can only mix within a buffer. A 10k-document buffer over a 100M-document corpus is barely a shuffle. Solved by larger buffers and by interleaving multiple shards.

The [`corpus.py`](corpus.py) module wires these up correctly so you don't have to think about them every module.

## What top labs actually do

This course streams from HuggingFace's CDN because it works on a $50 budget with zero preparation step. **That is not the production path.** Frontier labs almost never stream raw text from a public dataset over the internet during a training run — the bandwidth, the per-document tokenizer CPU cost, and the failure modes all rule it out at scale.

The production pattern, roughly:

- **Pre-tokenize once, offline.** The corpus is tokenized ahead of the training run and stored as a giant flat array of token IDs (`uint16` or `uint32`) on disk. The training loop reads token IDs directly — no tokenizer in the hot path, no Unicode parsing, no Python overhead per document.
- **Custom binary shard formats.** Common ones: **Mosaic Streaming** (`.mds`, designed exactly for this problem), **WebDataset** (`.tar` shards optimized for sequential reads), and **Megatron-LM's `.bin` + `.idx`** pair (a single mmap-able file of tokens plus a document-boundary index — the simplest workable thing). nanoGPT uses raw `uint16` memmaps and trains GPT-2-scale models on a single node.
- **Data lives on the training cluster, not on the public internet.** Either pre-downloaded to each node's local NVMe at job-launch, or fronted by an internal object store (S3 / GCS / proprietary storage like Meta's f4 or Google's Colossus) with an aggressive on-node cache. The training loop's data reads are local-disk speed.
- **Deterministic, resumable iterators.** Checkpoints save model weights *and* the data-iterator position. A run that dies at step 10,000 resumes exactly where it left off — same token, same order. This requires the iterator's state to be `(seed, position)` and nothing else; an in-flight shuffle buffer that gets lost on restart breaks this property.
- **Mixing is a weighted sampler over multiple shard sets**, often with a schedule that ramps proportions across training (e.g. more code in the second half, more math near the end).

What we keep, what we drop, for the course:

| Concept | HF streaming (what we use) | Production (what they use) |
|---|---|---|
| Source format | Parquet on HF CDN | Pre-tokenized binary shards on local NVMe |
| Tokenization timing | At training time (in the dataloader) | Once, offline, before training |
| Sharding by rank | `split_dataset_by_node` | Per-file shard assignment baked into dataset prep |
| Shuffle | Buffered shuffle within RAM | Pre-shuffled at prep time + small in-RAM jitter |
| Resumability | Best-effort (buffer state is lost) | Bit-exact resume from `(step, seed)` |
| Network during training | Over public CDN | Cluster-local, often zero network |

The *concepts* — sharding, filtering, shuffling, mixing — are identical. The substrate is different. Once you understand them on FineWeb-Edu over HuggingFace, swapping in a pre-tokenized Mosaic Streaming dataset on a real cluster is a config change, not a re-think. The training loop calls `next(loader)` the same way either way.

### How much disk do you actually need?

If you commit to the pre-tokenized approach, the storage cost is the first thing to plan for. The math is easy: each token is either a `uint16` (2 bytes, if vocab ≤ 65k) or a `uint32` (4 bytes, otherwise).

| Corpus / scale | Tokens | uint16 on disk | uint32 on disk |
|---|---|---|---|
| Our pretraining demo (Module 11) | ~1B | 2 GB | 4 GB |
| FineWeb-Edu `sample-10BT` | 10B | 20 GB | 40 GB |
| FineWeb-Edu `sample-100BT` | 100B | 200 GB | 400 GB |
| FineWeb-Edu `default` (everything) | ~1.3T | 2.6 TB | 5.2 TB |
| Frontier pretrain (DeepSeek-V3 / Llama 3 / Qwen 3 era) | ~15T | 30 TB | 60 TB |

The uint16/uint32 choice matters at scale. The modern frontier-lab vocabs (Llama 3 = 128k, Qwen 3 = 152k, DeepSeek-V3 = 129k) all exceed 65k and so all require `uint32` — doubling the storage cost vs the older 32k-vocab regime (GPT-2, Llama 2). This is one of the (many) reasons vocab size is not "the bigger the better" (Module 03 has the full tradeoff).

A few things to internalize from this table:

- **Pre-tokenization compresses against raw text, by a lot.** FineWeb-Edu `sample-10BT` is ~50 GB of compressed parquet → ~150 GB raw text → 20 GB as `uint16` tokens. The tokenizer is acting as a domain-specific codebook compressor. Counterintuitive but true.
- **Our course never gets near "big disk" territory.** Our 1B-token demo at uint16 is 2 GB; the entire `sample-10BT` corpus fits on a laptop SSD. Storage is not a constraint for anything in this repo.
- **The frontier is where it gets interesting.** A 15T-token uint32 corpus is ~60 TB. That doesn't fit on a single drive, but it comfortably fits on the local NVMe of a single modern 8×H100 node (typically 15–30 TB). Beyond that, you stream from object storage with on-node caching.

**Disk bandwidth is not the bottleneck**, which is the whole point of pre-tokenizing. A loaded A100 wants ~1–2 GB/sec of tokens; at `uint16` that's 1 GB/sec of disk reads. A single PCIe Gen3 NVMe (~3.5 GB/sec) saturates ~3 GPUs; PCIe Gen4 NVMe (~7 GB/sec) feeds a full 8×A100 node. For 8×H100 in FP8 you can stripe across multiple drives. There is no plausible single-node training config where a properly pre-tokenized corpus on local NVMe is the bottleneck — and that is the design goal.

**Where the bytes actually live in a real setup:**
- The *master copy* is on the lab's internal object store (S3 / GCS / proprietary). Durable, deduplicated, the source of truth across many training jobs.
- For each run, the relevant shards are *pre-staged* to each compute node's local NVMe at job launch. Modern training nodes ship with 3–15 TB of local NVMe specifically for this — it's not coincidence, it's sized for this workload.
- Runs that exceed local disk (the 15T-token regime on smaller nodes) stream from object store with an aggressive on-node cache (hundreds of GB), evicting LRU.

### A closer look: Mosaic Streaming

[Mosaic Streaming](https://github.com/mosaicml/streaming) (the `streaming` library, originally from MosaicML, now part of Databricks) is the closest thing to a drop-in production replacement for HF streaming. It's worth understanding because it shows what a *correctly-designed* training dataloader looks like — and several of its design choices are non-obvious until you've been bitten by their absence.

**The format:** `.mds` shards. Each shard is a self-describing binary file storing a fixed-schema set of samples plus a header. You declare columns at write time — e.g. `{"tokens": "ndarray:int32", "doc_id": "str"}` — and the library handles serialization, sharding, indexing, and integrity checks.

**The design choices that matter:**

- **Deterministic, resumable iteration.** Iteration order is a function of `(seed, sample_id)` only. Two ranks restarting from the same checkpoint with the same seed see the exact same next sample. No in-flight shuffle buffer to lose on restart — which is the property HF streaming most conspicuously *doesn't* have, and which you absolutely need if a long pretraining run is going to survive node failures.
- **Hierarchical shuffle.** Instead of HF's single-RAM-buffer approach, Mosaic Streaming pre-shuffles each shard at write time *and* shuffles the shard assignment across ranks at read time. The effective shuffle window is the whole corpus, not a 10k-document RAM buffer. This actually matters for training quality at scale — a 10k window over a 100M-document corpus is a 0.01% sample, and the model will see correlated document clusters as a result.
- **Elastic determinism.** You can resume a checkpoint on a *different* number of GPUs and get the same training trajectory. The rank-to-sample assignment is computed from world_size at runtime, not baked into the dataset prep. This is what lets you debug a 1024-GPU run on 8 GPUs and trust that you're seeing the same failure.
- **Remote-first storage.** Data lives on an object store (S3 / GCS / Azure / OCI / local filesystem); the library pulls shards into a local on-disk cache, with LRU eviction as the cache fills. You point it at a bucket URL and a local cache directory, and it handles the lifecycle. The training loop never touches the network directly.
- **Same dataloader API as everything else.** It's a `torch.utils.data.IterableDataset`, so the training loop doesn't change when you swap it in.

**Provenance.** Built for MosaicML's own MPT model family and Databricks' DBRX (a 132B MoE), so it's been load-tested at multi-thousand-GPU scale in production. Open source, Apache 2.0.

**The upgrade path from this course is straightforward:** when you outgrow HF streaming, you write a one-time MDS conversion script that tokenizes your corpus into `.mds` shards on S3, swap `stream_fineweb_edu` for `streaming.StreamingDataset(remote=..., local=..., shuffle=True)` in the training loop, and the rest is unchanged. We won't actually do this conversion in the course (it's an exercise in plumbing, not learning), but if you ever need to scale beyond the $50 regime, that's the path.

The other reference worth knowing is **Megatron-LM's data prep scripts** — the canonical `.bin`+`.idx` style. Less ergonomic than Mosaic Streaming, more bare-metal, and what most of the published "we trained a 70B model" recipes from research groups actually use. Worth reading once for the simplicity of the format.

## Distributed sharding

This course's contract: every training script launches with `torchrun`, every model wraps in FSDP, every dataset shards across ranks. (See [Module 00, rule 4](../../part-0-mental-model/00-what-we-are-building/) and the deep dive in Module 10.)

For data, the sharding rule is simple: **each rank sees a disjoint subset of the corpus.** Otherwise you'd train on duplicates and your effective batch size would be a lie. HuggingFace `datasets` supports this directly:

```python
dataset = load_dataset("HuggingFaceFW/fineweb-edu", streaming=True, split="train")
dataset = dataset.shard(num_shards=world_size, index=rank)
```

That single line is what makes the rest of the training stack distributed-correct. It's in `corpus.py`. You will not think about it again.

## The `corpus.py` module

[`corpus.py`](corpus.py) is the canonical data layer for every later training script. Read it once, then use it as a black box:

```python
from corpus import stream_fineweb_edu

stream = stream_fineweb_edu(
    rank=rank,
    world_size=world_size,
    min_score=3.0,        # FineWeb-Edu educational-quality score, 0–5
    min_chars=200,        # drop very short docs (likely boilerplate)
    shuffle_buffer=10_000,
    seed=42,
)
for doc in stream:
    text = doc["text"]
    ...  # tokenize, batch, feed to model
```

The notebook walks through what each knob does, what happens when you turn it the wrong way, and how to measure whether the stream is keeping up with the GPU.

## Decisions you'll keep making

| Decision | How to think about it |
|---|---|
| Which corpus? | Start from a well-documented filtered web corpus (FineWeb-Edu). Add code / math / domain data only if you can ablate and confirm a win. |
| How many tokens? | At least 10 tokens per parameter to see a clean loss curve. 20+ for research; 100+ if you intend to deploy. |
| Filter aggressiveness? | Higher quality threshold → less data, cleaner data. Tune on a validation perplexity loss; don't tune by intuition. |
| Shuffle buffer size? | As large as RAM permits. 10k–50k documents is a reasonable default for streaming. |
| Dedup level? | Use a corpus that has near-dup removal already done. Don't re-implement MinHash for a course; do recognize what it solved. |

## Next

[Module 03 — Tokenization](../03-tokenization/). Documents are not tokens. The tokenizer is the bridge, and the choices you make there constrain *every* downstream module — vocab size sets the embedding table size, sets the output projection size, sets compression rate, sets which languages and domains are over- or under-represented.
