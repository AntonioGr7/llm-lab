# 12 — Production Data Pipelines

> Indexed-shuffle, a pre-tokenized binary corpus, and **O(1) bit-exact resume** — the data pipeline frontier labs actually run.

Module 11 gave you a complete pretraining framework, and its data pipeline — streaming FineWeb-Edu, tokenizing on the fly, shuffling a buffer — is the *right* place to start. It needs no preprocessing, no disk budget, and one `pip install datasets`. For a course demo it is exactly correct.

It also has two limitations that you hit the moment a run gets long, and Module 11's own code says so out loud. From [`11-.../checkpoint.py`](../11-pretraining-in-practice/checkpoint.py):

> **NOT persisted (known limitation): Data loader position.** Streaming datasets restart from sample 0 on resume; the model replays the first few percent of the corpus.

This module fixes that, the way Megatron-LM, TorchTitan, nanotron, and MosaicML do: **tokenize once into an indexed binary corpus, shuffle with an explicit permutation, and check the data position into a single integer** so a crashed run resumes in O(1) — exactly bit-for-bit where it left off, no replay, no re-tokenization.

Like Module 11, this is a **framework directory you lift out**. Drop [`indexed_data.py`](indexed_data.py) into your pretraining repo, point it at a corpus you built with [`prepare_fineweb_edu.py`](prepare_fineweb_edu.py), and your data loader is now checkpointable.

---

## What you'll be able to do at the end of this module

- Explain, by name, the two things Module 11's streaming pipeline can't do and why a real run cares.
- Build a pre-tokenized binary corpus (`.bin` + index) and reason about its disk cost from the tokenizer's vocab size.
- Read any sample in O(1) from a memory-mapped corpus, and understand why that's the prerequisite for fast resume.
- Implement a resumable, distributed, permutation-shuffled sampler and prove its resume is bit-exact with a test.
- Decide, for a given run, whether streaming or indexed is the right pipeline — and defend the choice.

---

## 1. The two limitations of streaming

Module 11's [`FineWebEduDataset`](../11-pretraining-in-practice/data.py) is an `IterableDataset`: a one-way valve. You ask for the next packed sequence; it tokenizes documents off the stream and hands one over. That design forces two costs.

**(a) The data position can't be cheaply checkpointed → resume replays.**
An iterator's "position" is hidden state living inside a streaming HTTP connection, a shuffle buffer, and a half-consumed document. There's no integer to save. So on resume, Module 11 opens a *fresh* iterator at position 0 with the same seed — and the model re-sees the first `start_step × tokens_per_step` tokens before it reaches new data. For the 3B-token demo that's ~1–3% wasted after a single crash: annoying. For a trillion-token run that crashes at 80%, it's catastrophic — you cannot afford to replay 800B tokens, and the run is no longer reproducible.

**(b) The tokenizer runs every epoch.**
Streaming tokenizes *as it reads*. Read the corpus twice (two epochs) and you tokenize it twice. Tokenization is CPU-bound; on a multi-epoch run it competes with your dataloader for the same cores and can starve the GPU. The tokens are deterministic — computing them more than once is pure waste.

Both come from the same root: **the corpus is never materialized, so it has no addresses.** Fix that and both problems dissolve.

---

## 2. The fix: a pre-tokenized, indexed binary corpus

Run the tokenizer **once**, ahead of time, and write the integer token ids to disk in a format you can address randomly. [`prepare_fineweb_edu.py`](prepare_fineweb_edu.py) does this; [`indexed_dataset.py`](indexed_dataset.py) reads it back. Three files share a prefix:

```
results/corpus/fineweb_10bt.bin        ← the token stream: one flat array of uint16/uint32
results/corpus/fineweb_10bt.idx.npy    ← int64 array: how many tokens each document has
results/corpus/fineweb_10bt.meta.json  ← {dtype, total_tokens, n_docs, vocab_size, eos_id, ...}
```

```
                         .bin  (flat token stream, memory-mapped — never fully loaded)
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ doc0 tokens … EOS │ doc1 tokens … EOS │ doc2 tokens … EOS │ doc3 … EOS │ ... │
   └───────────────────────────────────────────────────────────────────────────┘
     ▲                   ▲                   ▲
     └─ offsets[0]=0     └─ offsets[1]       └─ offsets[2]      ← .idx.npy = cumsum(sizes)

   Training doesn't even use doc boundaries: it slices fixed (seq_len+1) windows
   (same packing as Module 11), so sample i = flat[i·bl : (i+1)·bl].  O(1).
```

Megatron-LM packs all of this into a single binary `.idx` with a magic header; we split it into three plain files so you can `cat` the meta and `np.load` the index in a notebook. The mechanics are identical.

### The dtype decision (easy to get wrong, costs you 2× disk)

A token id is just an integer, so the token stream's element type should be the **smallest unsigned int that holds your vocabulary**:

| Tokenizer | Vocab size | Fits in | Bytes/token | 10B tokens on disk |
|---|---:|---|---:|---:|
| GPT-2 | 50,257 | `uint16` (≤ 65,535) | 2 | ~20 GB |
| **Qwen3** | **151,936** | `uint32` (> 65,535) | **4** | **~40 GB** |

This is the whole reason FineWeb-Edu `sample-10BT` is ~40 GB with the Qwen3 tokenizer, not ~20 GB: 151,936 overflows `uint16`, so every token costs 4 bytes. [`best_dtype_for_vocab`](indexed_dataset.py) makes this choice for you (Megatron calls it `best_fitting_dtype`). Get it wrong in the *small* direction and your tokens silently wrap around mod 65,536; in the large direction and you double your disk and read bandwidth for nothing.

---

## 3. O(1) random access via mmap

[`IndexedDataset`](indexed_dataset.py) opens the `.bin` with `np.memmap(..., mode="r")`. `memmap` maps the file into the process's virtual address space **without reading it** — pages are pulled from disk by the OS only when you touch them. So:

```python
corpus = IndexedDataset("results/corpus/fineweb_10bt")
corpus[9_000_000]      # the 9-millionth document: a pointer + a slice. O(1).
corpus.flat[start:end] # any token range: O(1), and only those pages page in.
```

[`PackedIndexedDataset`](indexed_data.py) is a thin map-style `Dataset` on top: sample `i` is the `i`-th non-overlapping `(seq_len + 1)`-token block of `flat`, returned as `{input_ids, labels}` with the same +1 shift convention as Module 11 (`labels[t] == input_ids[t+1]`). `__getitem__(i)` is `flat[i·bl : (i+1)·bl]` — reading sample 9,000,000 costs the same as reading sample 0.

That constant-time-anywhere property is the whole game. **Streaming can only give you the next sample; an indexed corpus gives you *any* sample for free.** Resume is the payoff.

---

## 4. The shuffle is an explicit permutation

Streaming shuffles a sliding `shuffle_buffer`-sized window — good enough locally, but two samples 50 GB apart in the stream never mix. With random access we can do better: shuffle the *whole* corpus by drawing a permutation of `[0, num_samples)` and visiting samples in that order.

```python
perm = np.random.default_rng(seed + epoch).permutation(num_samples)
# epoch 0 visits perm[0], perm[1], …; epoch 1 reseeds with seed+1; etc.
```

Two properties matter:

- **Deterministic.** NumPy's PCG64 stream is stable across machines and versions, so `(seed, epoch)` fully determines the order. That's what makes resume reproducible.
- **Reseeded per epoch.** Each epoch gets a fresh order (`seed + epoch`), so the model doesn't see the same sequence of batches every pass.

The permutation can be **regenerated from the seed** (instant for small/medium corpora) or **materialized once to `.npy` and memory-mapped** (for huge corpora, so resume pages in only the slice it needs — this is Megatron's cached "shuffle index"). `prepare`'s `--write-perm` writes the artifact; the sampler uses it if present and regenerates otherwise. Both paths produce the identical array.

---

## 5. O(1) bit-exact resume — the headline

Here is the entire resume state for the data loader:

```python
sampler.state_dict()   # {'consumed_samples': 4_915_200, 'seed': 0, 'num_samples': ..., 'step_samples': ...}
```

One integer — `consumed_samples`, the number of samples consumed across all ranks — plus the knobs that define the permutation (saved so a config typo on resume is *caught*, not silently honored). On restart, [`ResumableDistributedSampler`](indexed_data.py) recovers everything by arithmetic:

```python
epoch          = consumed_samples // active_per_epoch          # which permutation
chunk_to_skip  = (consumed_samples % active_per_epoch) // step_samples
```

It rebuilds `perm_epoch`, jumps the chunk pointer to `chunk_to_skip`, and continues. **Computing where to start is O(1); reaching the first sample is one mmap slice.** No replay, no re-tokenization. The resumed run sees *exactly* the samples the uninterrupted run would have, in the same order — including across epoch boundaries, where the permutation reseeds.

Contrast the two pipelines after a crash at step N:

| | Module 11 (streaming) | Module 12 (indexed) |
|---|---|---|
| Resume state | none (hidden iterator) | one integer `consumed_samples` |
| Cost to reach step N+1 | replay N·tokens_per_step tokens, re-tokenizing | O(1) arithmetic + one slice |
| Same data as no-crash run? | **no** — replays from sample 0 | **yes** — bit-exact |

[`tests/test_indexed_resume.py`](tests/test_indexed_resume.py) proves the bit-exact claim: it runs a sampler uninterrupted, then runs it with a checkpoint-and-rebuild in the middle (across an epoch boundary), and asserts the two visit the identical sample sequence. Run it whenever you touch the sampler:

```bash
python tests/test_indexed_resume.py
```

---

## 6. Distributed and worker sharding via the index

With `world_size` data-parallel ranks, every rank must see a **disjoint** slice of each step's data, and together they must cover everything — no overlaps (wasted compute, double-counted gradients) and no gaps. Random access makes this trivial: shard the *index*, not the stream.

Each epoch's permutation is cut into chunks of `step_samples = micro_batch × world`. Chunk `c` is one synchronized step across the group; rank `r` takes the contiguous slice `chunk[r·micro_batch : (r+1)·micro_batch]`:

```
chunk c (one global step) = perm[c·step_samples : (c+1)·step_samples]
   rank 0 → [............]   rank 1 → [............]   rank 2 → [............]   rank 3 → [............]
            micro_batch idxs          micro_batch idxs          micro_batch idxs          micro_batch idxs
```

Across ranks this tiles the chunk exactly — disjoint, complete, balanced — and because every rank derives the same permutation from the same `(seed, epoch)`, no rank-to-rank communication is needed to agree on the split. (DataLoader **workers** within a rank subdivide further automatically: each worker handles a subset of the batches the sampler emits.) The trailing `num_samples % step_samples` samples are dropped each epoch (a fraction of one step) so every step is full and balanced — standard `drop_last` behavior. The test verifies disjointness + complete coverage for `world=4`.

---

## 7. Running it

### Step 1 — build a corpus (tokenize once)

```bash
# $0 / offline: a tiny synthetic corpus to play with (used by the notebook & tests)
python prepare_fineweb_edu.py --source synthetic \
    --output results/corpus/tiny --num-docs 4000 --vocab-size 512

# A small real slice — ~20k FineWeb-Edu docs with the Qwen3 tokenizer, 8 procs
python prepare_fineweb_edu.py --source fineweb_edu \
    --output results/corpus/fineweb_small \
    --tokenizer Qwen/Qwen3-0.6B --num-docs 20000 --workers 8

# The full sample-10BT corpus (~40 GB on disk, a few hours on 16 cores — once)
python prepare_fineweb_edu.py --source fineweb_edu \
    --output results/corpus/fineweb_10bt --subset sample-10BT --workers 16 \
    --write-perm --seq-len 2048
```

Tokenization is embarrassingly parallel (`multiprocessing.Pool`, one tokenizer per worker), and `imap` keeps memory flat so you can write a 40 GB corpus without 40 GB of RAM.

### Step 2 — it's already wired into Module 11

Module 11 now ships this integration: set `data.source: indexed` and point `data.index_prefix` at your corpus (there's a ready config at [`11-.../configs/demo_indexed.yaml`](../11-pretraining-in-practice/configs/demo_indexed.yaml)). `indexed_dataset.py` and `indexed_data.py` are copied into Module 11's directory — the same "lift it out, copy what you need" pattern Module 11 uses for every other component.

The three edits below are what that wiring *is*, shown so you can replicate it in your own repo. The indexed loader is a near drop-in for Module 11's `make_dataloader`, with one difference that drives all three: it returns **`(loader, sampler)`**, because the *sampler* owns the resume state.

**a) `config.py` — add the source and the corpus path:**
```python
DataSource = Literal["synthetic", "fineweb_edu", "indexed"]      # + "indexed"

@dataclass
class DataConfig:
    ...
    index_prefix: str = ""        # e.g. "results/corpus/fineweb_10bt"
    perm_path: str = ""           # optional cached permutation (.npy); "" = regenerate
```

**b) `data.py` — route `source == "indexed"` to this module:**
```python
from indexed_data import make_indexed_dataloader   # this module

def make_dataloader(cfg, vocab_size):
    if cfg.source == "indexed":
        import torch.distributed as dist
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank  = dist.get_rank() if dist.is_initialized() else 0
        return make_indexed_dataloader(
            cfg.index_prefix, seq_len=cfg.seq_len,
            micro_batch=cfg.batch_size_per_device,
            num_replicas=world, rank=rank, seed=cfg.seed,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            perm_path=cfg.perm_path or None,
        )                                  # returns (loader, sampler)
    ...                                    # existing synthetic / fineweb_edu branches,
    return loader, None                    # ...now return (loader, None) so the
                                           #    call site unpacks the same shape
```
Making every branch return `(loader, sampler)` — with `sampler=None` for the streaming/synthetic sources, which have no checkpointable position — is what keeps `train.py`'s call site uniform.

**c) `train.py` — keep the sampler, and checkpoint its one integer.**
Capture it and persist the position to a tiny `data_state.json` sidecar next to the model checkpoint, restored on resume:
```python
loader, sampler = make_dataloader(cfg.data, vocab_size=cfg.model.vocab_size)
...
# on save (rank 0):   _write_data_state(sampler, ckpt_path, steps_done, cfg, world)
# on resume:          _restore_data_state(sampler, resume, start_step, cfg, world, is_main)
```

**The trap (this is the one thing the naïve version gets wrong).** It is tempting to just `json.dump(sampler.state_dict(), ...)` on save. But with `num_workers>0` the DataLoader **prefetches** `num_workers × prefetch_factor` micro-batches, so `sampler.consumed_samples` has already raced *ahead* of what the training loop actually trained on. Save that live counter and resume **skips** those prefetched batches — silently breaking the bit-exact guarantee that is the entire point of this module. The bug never appears in the isolated sampler test (no workers, no prefetch); it only bites once it's behind a real DataLoader.

The fix: don't trust the live counter — **derive the position from the optimizer-step boundary**, where it's exact and prefetch-independent. One optimizer step consumes `grad_accum` micro-batches of `batch_size_per_device` samples per rank across `world` ranks, so:
```python
consumed_samples = steps_done * grad_accum * batch_size_per_device * world
```
`_write_data_state` writes *that* value (reusing `state_dict()`'s `seed`/`num_samples` only for the resume-time mismatch guard); `_restore_data_state` reads it back, `load_state_dict` validates the corpus/seed are unchanged, and the sampler seeks there in O(1). Verified bit-exact across a kill-and-resume with `num_workers=2`: loss, grad_norm, and LR are identical to the uninterrupted run, step for step.

(Tidier still: thread `data_state` through `checkpoint.save`/`load` so it lives inside the DCP checkpoint instead of a sidecar — left as an exercise. Note the sidecar lives *inside* the `step_XXXX/` dir, so checkpoint rotation prunes it along with the weights — no orphaned files.)

### Step 3 — re-run and compare (the capstone)

You already trained a model in Module 11 with the **streaming** source. Now train the *same model* on the **indexed** corpus and compare — this is the payoff of the module. The discipline that makes the comparison honest: change **only** `data.source`. Same model shape, same `seed`, same `total_steps`, same tokens/step. [`configs/demo_indexed.yaml`](../11-pretraining-in-practice/configs/demo_indexed.yaml) is `demo_a100.yaml` with that single structural change already made.

```bash
# A — streaming (what you already ran in Module 11). No prep: it tokenizes as it reads.
torchrun --standalone --nproc_per_node=1 train.py --config=configs/demo_a100.yaml

# B — indexed. This path reads integers off disk, so the corpus MUST exist first.
#   B0 (once, the prerequisite): tokenize ~1B tokens to disk → ~4 GB, ~15-40 min.
#       --num-docs ~1.1M ≈ 1B tokens at FineWeb-Edu's ~900 tok/doc; watch the
#       running token counter and stop when it crosses 1B.
python ../12-production-data-pipelines/prepare_fineweb_edu.py \
    --source fineweb_edu --output results/corpus/fineweb_1b \
    --tokenizer Qwen/Qwen3-0.6B --subset sample-10BT --num-docs 1100000 --workers 16
#   B1: train on it (demo_indexed.yaml's index_prefix already points at fineweb_1b).
torchrun --standalone --nproc_per_node=1 train.py --config=configs/demo_indexed.yaml
```

Run B *before* B0 and it fails fast — `data.source='indexed'` with a missing corpus can't open `<prefix>.meta.json`. The prepare step is the line that separates "tokenize every epoch at train time" (A) from "tokenize once, ever" (B) — which is exactly the next question.

**Set expectations first: the two loss curves will not be bit-identical, and that's the lesson, not a bug.** Streaming shuffles a sliding buffer window and packs across that shuffled stream; indexed applies a *global* permutation over fixed-position samples. Different sample orderings → different curves. What you actually compare is four things:

| Signal | What to look for | Why it differs |
|---|---|---|
| **tok/s + SM%** | Indexed should be steadier and higher, especially on epoch 2+ | Streaming re-tokenizes every epoch on the CPU and can starve the GPU; indexed reads pre-computed integers off an mmap |
| **Shuffle quality** | Indexed mixes the whole corpus from step 0; streaming only mixes a `shuffle_buffer` window | Global permutation vs local buffer |
| **Resume** | Kill both at step N and restart: streaming replays from sample 0; indexed continues exactly (watch the `data position restored` log) | Checkpointable position vs hidden iterator state |
| **Reproducibility** | Re-run indexed with the same seed → same order every time; streaming's order depends on buffer timing | Explicit permutation vs stream-dependent shuffle |

The resume contrast is the one to actually demonstrate: `Ctrl-C` each run a few hundred steps in, relaunch the same command, and watch streaming's first-batch log replay early data while indexed prints `data position restored: consumed_samples=…` and picks up where it left off.

---

## 8. Streaming vs indexed — which to use

Neither is strictly better; they trade setup cost for run-time guarantees.

| | Streaming (Module 11) | Indexed (Module 12) |
|---|---|---|
| Preprocessing | none | tokenize once → `.bin` (minutes–hours) |
| Disk | ~0 (reads what it consumes) | full corpus on disk (~40 GB / 10BT, Qwen3) |
| Tokenizer cost | every epoch | once, ever |
| Shuffle quality | local (buffer window) | global (full permutation) |
| Random access | no | O(1) |
| Resume | replays from sample 0 | O(1), bit-exact |
| Best for | a first run, exploration, a corpus bigger than your disk, single-epoch | multi-epoch, long runs that *will* crash, reproducibility, anything you'd call "production" |

**Rule of thumb:** start streaming to get a run going. Switch to indexed the moment (a) you'll do more than one epoch, (b) the run is long enough that a crash is likely, or (c) you need a result you can reproduce exactly. At frontier scale all three are always true, which is why every major framework ships an indexed pipeline.

---

## 9. Disk and preprocessing cost (plan before you run)

- **Disk** = `total_tokens × bytes_per_token`. With Qwen3 (`uint32`, 4 B): `sample-10BT` ≈ 40 GB; the full FineWeb-Edu (~1.3T tokens) ≈ 5.2 TB. With a `uint16` tokenizer, halve both. Point `--output` at a fast local SSD, not network storage — training reads it at random.
- **Time** scales with `--workers` (CPU-bound). The full `sample-10BT` is a few hours on a 16-core box; a 20k-doc slice is seconds. You pay it **once** — every subsequent epoch and every resume reads pre-computed integers.
- **The `.idx.npy` and `.meta.json` are tiny** (one int64 per document + a few fields). The permutation cache, if written, is `num_samples × 8 B` — for 10BT at seq_len 2048 that's ~20M samples ≈ 160 MB, trivially mmappable.

---

## 10. The $0 hands-on

Everything in this module runs on a laptop CPU with no network:

- [`notebook.ipynb`](notebook.ipynb) — builds a tiny corpus, shows O(1) access and the per-epoch permutation, then **simulates a crash and proves bit-exact resume**, and finishes by simulating 4-way distributed sharding. This is the "play with a custom implementation on a small portion" lab.
- [`tests/test_indexed_resume.py`](tests/test_indexed_resume.py) — the three correctness properties as a pass/fail script.

```bash
python prepare_fineweb_edu.py --source synthetic --output results/corpus/tiny --num-docs 4000 --vocab-size 512
python tests/test_indexed_resume.py
jupyter notebook notebook.ipynb
```

---

## 11. Stretch goals

- **Document-respecting sample index (full Megatron).** We pack fixed-stride windows that can cross document boundaries (so does Module 11). Megatron builds a *sample index* that maps each sample to `(doc, offset)` and never splits the wrong way, plus a separate document-shuffle. It's more faithful but costs an O(num_tokens) build at startup (cached to disk). Implement it and compare loss curves — for pretraining the difference is usually negligible, which is *why* fixed-stride packing is so common.
- **Blended / weighted corpora.** Real runs mix sources (web + code + math + books) at chosen ratios. Megatron's `BlendedDataset` samples a source per index from a weight vector. Add a `weights=[0.6, 0.2, 0.2]` over several `.bin` files and make the sampler draw the source first, then the within-source index — still O(1), still resumable.
- **Stream the indexed corpus from cloud storage (MosaicML MDS / Streaming).** When the corpus doesn't fit on local disk, MDS shards the binary into cloud objects and prefetches shards with a resumable, deterministic order — indexed semantics without a full local copy. Read their `StreamingDataset` and note how it keeps O(1)-ish resume across nodes.
- **Online dedup / quality filtering at prep time.** `prepare_*.py` is the natural place to run MinHash dedup or a quality classifier before tokens hit disk, so the expensive filtering also happens once.

---

## 12. Reading list

Ordered by how directly it maps to this module's code:

- **Megatron-LM** — [`megatron/core/datasets/indexed_dataset.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py) (the `.bin`/`.idx` format and `MMapIndexedDataset` we simplified) and `gpt_dataset.py` (the sample-index + shuffle-index build). This is the reference implementation everyone copies.
- **nanotron** (HuggingFace) — `nanoset.py`: a leaner take on the same flat-token-file + index idea, close to what we built here.
- **litgpt** (Lightning AI) — its `prepare_*` scripts and the `litdata`/streaming dataset: a clean, readable preprocessing → packed-binary → training path.
- **TorchTitan** — its data loader and checkpoint integration show how a modern PyTorch-native stack threads data-iterator state through `torch.distributed.checkpoint` (the tidy version of §7c).
- **MosaicML Streaming (MDS)** — [docs.mosaicml.com/projects/streaming](https://docs.mosaicml.com/projects/streaming/): deterministic, resumable, shard-based streaming from cloud storage when the corpus won't fit on local disk.
- **FineWeb / FineWeb-Edu technical report** (Penedo et al., 2024) — what's actually in the corpus you're indexing, and why the filtering matters more than the pipeline.

---

*Previous: [11 — Pretraining in Practice](../11-pretraining-in-practice/). This module closes the data-loader-resume gap that module's checkpointing flagged. Next: [Part 4 — Post-Training](../../part-4-post-training/).*
