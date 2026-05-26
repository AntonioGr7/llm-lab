# Module 11 — Pretraining in Practice

> Part of [Part 3 — Pretraining](../). Reading time: ~90 minutes. Compute cost: $0 (read + CPU notebook) to ~$15–25 (single-A100 demo run on FineWeb-Edu).

## The thesis

Modules 08–10 built the components. This module is the **actual run**: a real dataset (FineWeb-Edu), a real model (~150M Qwen3-shaped), a real `torchrun train.py` invocation that you could scale from your single A100 to a 64-GPU cluster by changing two flags. The artifact is **this directory** — lift it out, drop it into a new repo, and you have a working pretraining stack.

The new code here is the *integration* and the *data*. Everything else is Modules 08/09/10's components, copied in so the directory is self-contained. The shape of the bet:

- **FineWeb-Edu** (HuggingFace) as the corpus. Streamed, tokenized on the fly, packed into 2k-token sequences.
- **150M-param Qwen3** (12 layers × 768 width × GQA(12, 2) × FFN 2048). Chinchilla-optimal at ~3B tokens — small enough to demo, big enough to be interesting.
- **FSDP2 with activation checkpointing**, BF16 mixed precision, cosine LR schedule with warmup, AdamW with the standard decay/no-decay split, gradient clipping at 1.0.
- **Effective batch size ~1M tokens/step** — on 1 GPU: `8 per-device × 64 grad-accum × 2048 seq ≈ 1.05M`. On 8 GPUs: `8 × 8 × 8 × 2048 ≈ 1.05M` (override `--training.grad_accum=8`). The 8× drop in grad-accum is the only flag that moves when you scale.
- **~3000 steps, ~3B tokens** — close to Chinchilla-optimal (≈15:1 against the 206M-param real count; the embedding adds 117M on top of the ~89M "hidden" parameters).

Everything below tells you how to actually run this, what to monitor while it's running, and what to do when something goes wrong.

## What you'll be able to do at the end

- Run a real pretraining session — single GPU, multi-GPU, or multi-node — with the same code.
- Stream and tokenize a 10B-token corpus without ever materializing the dataset on disk.
- Build a monitoring dashboard with the *right* metrics: loss, grad-norm, tokens/sec, MFU, free HBM, learning rate.
- Diagnose a loss spike from its grad-norm signature and decide whether to resume from an earlier checkpoint or change a hyperparameter.
- Evaluate a base model (perplexity + generation sanity-check + a pointer at lm-evaluation-harness) and know what each signal can and can't tell you.
- Scale the demo to a 1B / 7B / 70B model by editing the config — without touching the training loop.

## 1. The directory layout

```
11-pretraining-in-practice/
  README.md          # this file
  config.py          # the composed TrainConfig
  model.py           # build_model — Qwen3 default, swap-friendly
  optim.py           # build_optimizer — AdamW with decay/no-decay groups
  schedule.py        # build_scheduler — warmup-cosine (default) and WSD
  fsdp_setup.py      # init_distributed + apply_fsdp
  efficiency.py      # apply_activation_checkpointing
  checkpoint.py      # DCP save/load
  loop.py            # forward_loss + train_step
  data.py            # FineWeb-Edu streaming pipeline + Synthetic fallback
  train.py           # the entrypoint you launch with torchrun
  eval.py            # perplexity + a small generation sanity check
  configs/
    demo_a100.yaml   # the 150M / 3B-token demo config — A100-80GB baseline
    demo_h100.yaml   # same run, tuned for H100-80GB (bigger batch, no AC)
  notebook.ipynb     # narrative walkthrough on CPU
  results/           # pre-run loss curve + checkpoint (committed)
```

**This is the directory.** Copy `11-pretraining-in-practice/` into any new repo and you have a working pretraining stack. No external course dependencies.

The integrators are `train.py`, `eval.py`, `data.py`. Everything else is consolidated from Modules 08–10 — same code, just gathered in one place so the directory is self-contained.

## 2. The data — FineWeb-Edu, streamed and packed

The corpus for 2026 small-to-mid open pretraining is **FineWeb-Edu** (HuggingFace, 2024): a ~1.3T-token, quality-filtered subset of FineWeb specifically scored for educational content. It outperforms raw FineWeb at iso-token-count on every reasoning benchmark — exactly the regime a small model lives in. The public **`sample-10BT`** subset (10B tokens) is more than enough for any model up to ~500M.

We **stream** the dataset, never materialize it on disk:

```python
from datasets import load_dataset

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)
```

This returns an `IterableDataset`. Each example is a dict with a `"text"` field — one document. The streaming reader pulls a few hundred MB at a time, no `data/` directory needed.

### Tokenize and pack on the fly

The pipeline is short. Per rank, we:

1. **Shard** the streaming dataset by rank (`ds.shard(num_shards=world_size, index=rank)`). Each rank reads a disjoint slice.
2. **Shuffle** a small in-memory buffer (~10k examples) so consecutive batches aren't from the same source URL.
3. **Tokenize** each document with the Qwen3 tokenizer, append the EOS token.
4. **Pack** tokens into a rolling buffer; whenever the buffer has ≥ `seq_len + 1` tokens, emit one sample and slide.
5. **Batch** `B` packed samples into a single `(B, seq_len + 1)` tensor; the loop splits this into `input_ids` (the first `seq_len`) and `labels` (the last `seq_len`).

The "+1" matters. For next-token prediction with cross-entropy, you need `input_ids[t]` and `labels[t] = input_ids[t+1]` aligned. Packing one extra token per sample makes the shift clean. This is the canonical preprocessing every frontier-model team uses.

**Where the loss is computed.** The shift happens **in the dataset, once**. The training loop calls `model(input_ids=...)` (no `labels=` kwarg), takes `.logits`, and applies cross-entropy against `batch["labels"]` itself — see [`loop.py:forward_loss`](loop.py). The model contract is therefore minimal: *return logits of shape `[B, S, V]`*. This avoids the per-architecture quirk where some HF causal LMs shift internally when you pass `labels=` (Qwen, Llama, Mistral, Gemma) and others don't (older GPT-2, plus any custom `nn.Module`). The framework stays swap-friendly across HF families and your own implementations.

**Why packing.** Without packing, each document becomes one (short) sample, padded to `seq_len`. Most batches are mostly padding. Packing concatenates documents (separated by EOS) and chops the stream into fixed-length blocks — **no padding waste, ~100% of GPU FLOPs do real work**. The model learns that EOS-then-new-document is a discontinuity, which is fine; document boundaries are not a meaningful structure to preserve at pretraining.

**Why streaming over downloading.** FineWeb-Edu sample-10BT is ~40 GB on disk after tokenization. Streaming costs ~50 MB/s of network on a typical Lambda Labs node, well below the GPU's appetite. You skip the 30-minute download and the disk space.

See [`data.py`](data.py) for the implementation.

### Why use Qwen3's tokenizer (not the one we trained in Module 03)?

[Module 03](../../part-1-data/03-tokenization/) taught how to train a BPE tokenizer from scratch and produced [`tokenizer.py`](../../part-1-data/03-tokenization/tokenizer.py) — explicitly meant as "the canonical tokenizer interface for every later training script." Then Module 11 reaches for `Qwen/Qwen3-0.6B`'s tokenizer instead. What gives?

The honest tradeoff at this scale:

| Path | Pros | Cons |
|---|---|---|
| **Use a frontier tokenizer (Qwen3, here)** | Seen *trillions* of tokens during training; near-optimal compression on diverse text; vocab covers code, math, multilingual; lets you continue into Part 4 post-training on Qwen3-base directly (same vocab). | You skipped your own training step. The 151k vocab is overkill for English-only Edu content; the LM head gets bigger. |
| **Use the Module 03 BPE we trained ourselves** | Sized to your corpus (32k vocab, no waste); embedding is half the size; the pedagogical loop closes — "we built our own tokenizer and used it." | Trained on the small Module 03 corpus, *not* 10B tokens of FineWeb-Edu; compression on real Edu data will be 5–15% worse than Qwen3's; can't load Qwen3-base weights for any later continued-pretraining ablation. |

At ~150M params on ~3B tokens, **either path produces a working model**; the model-quality gap is in the noise compared to "did the training loop converge at all." We default to the frontier tokenizer because (a) it's what every production small-model team in 2026 does, and (b) it lets Part 4's SFT/DPO/GRPO inherit Qwen3's chat template without surgery.

If you want to close the pedagogical loop — train your own tokenizer on a real FineWeb-Edu slice and use it here — `data.py` supports both transparently. Set:

```yaml
data:
  tokenizer_name: ../../part-1-data/03-tokenization/results/tokenizer.json
model:
  vocab_size: 32000      # whatever your trained tokenizer reports
```

`load_tokenizer(name)` in `data.py` detects the format from the string:
- ends in `.json` or contains a path separator → loads via `tokenizers.Tokenizer.from_file` (the Module 03 path);
- otherwise → loads via `transformers.AutoTokenizer.from_pretrained` (HF Hub IDs like `"Qwen/Qwen3-0.6B"`).

The EOS handling differs by convention — Module 03's BPE uses `<|endoftext|>` as its EOS/EOT; the loader picks it up automatically. Everything downstream (`forward_loss`, the loop, the scheduler) is tokenizer-agnostic.

A note on going further: for a *production* small model on ~10B+ tokens of a specific corpus, training your own BPE is the right call — the corpus-fit compression gain pays for itself. The course's `train_bpe.py` is sized for a teaching demo; for a real run you'd retrain on a larger slice of FineWeb-Edu and bump the vocab to ~50k.

### The fallback: `SyntheticDataset`

For the $0 tier (CPU walkthrough, smoke tests, CI), `data.py` also provides `SyntheticDataset`: random token IDs from a small vocab, no internet, instant. The demo config has a `source: "synthetic"` toggle that switches to it.

## 3. The model — 150M Qwen3, Chinchilla-optimal at 3B tokens

The demo config is sized to **fit on a single A100-80GB** with comfortable headroom and to be **Chinchilla-optimal** at ~3B tokens:

| Knob | Value | Why |
|---|---|---|
| `vocab_size` | 151,936 | Qwen3 tokenizer vocabulary |
| `hidden_size (d_model)` | 768 | Standard width for ~150M |
| `n_layers` | 12 | |
| `n_heads` | 12 | Q heads |
| `n_kv_heads` | 2 | GQA(6×) — same as Qwen3-0.6B |
| `intermediate_size (d_ffn)` | 2,048 | ~2.7× d_model |
| `max_seq` | 2,048 | The pretraining sequence length |
| `rope_theta` | 10,000 | Standard |
| `tie_word_embeddings` | True | Embedding ≡ LM head (saves params) |

Parameter count: **~150M** (with the 152k embedding dominating).

**Total training tokens**: ~3B. Sized so the headline "150M-class" model is close to Chinchilla. With the embedding included the count is 206M; against the ~89M of hidden weights the ratio is even higher. Either way, the budget is the same.

```
steps × tokens_per_step = 3000 × 1,048,576 ≈ 3.1B tokens
tokens_per_step = batch_per_device × grad_accum × world_size × seq_len
               = 8 × 64 × 1 × 2048 = 1,048,576   (single-A100, demo default)
               = 8 ×  8 × 8 × 2048 = 1,048,576   (8-A100 node, --training.grad_accum=8)
```

Both configurations train the same model with the same effective batch — gradient accumulation and data parallelism are equivalent ways to spend tokens. The 8× drop in `grad_accum` is exactly compensated by the 8× rise in `world_size`. The demo YAML picks the single-GPU shape; the multi-GPU command override drops grad_accum.

## 4. The training configuration

[`configs/demo_a100.yaml`](configs/demo_a100.yaml) is the source of truth for the A100 baseline; [`configs/demo_h100.yaml`](configs/demo_h100.yaml) is the H100-tuned twin (same model and token budget, just sized for H100's headroom — see §5 for the launch and §11 for the diff). Highlights of the A100 baseline:

```yaml
model:
  vocab_size: 151936
  d_model: 768
  n_layers: 12
  n_heads: 12
  n_kv_heads: 2
  d_ffn: 2048
  max_seq: 2048
  tie_weights: true

optimizer:
  type: adamw
  lr: 3.0e-4              # peak LR
  betas: [0.9, 0.95]      # the LLM-canonical betas
  weight_decay: 0.1

schedule:
  type: cosine
  warmup_steps: 200       # ~7% of total — generous because the run is short
  min_lr_ratio: 0.1       # cosine floor at 10% of peak

data:
  source: fineweb_edu     # set "synthetic" for offline smoke tests
  seq_len: 2048
  batch_size_per_device: 8
  num_workers: 2

training:
  total_steps: 3000
  grad_accum: 4           # → 8 × 4 × 1 × 2048 = 65k tokens/step on 1 GPU
  grad_clip: 1.0
  dtype: bf16
  activation_checkpointing: true
  log_every: 10
  save_every: 500
  checkpoint_dir: ./results/checkpoints
```

The YAML maps 1:1 onto [`config.py`](config.py)'s `TrainConfig` dataclass. `train.py` loads it with `load_yaml(...)`. CLI overrides use dotted paths: `--training.grad_accum=8`.

## 5. Running it

### Single A100 (the canonical demo)

```bash
cd part-3-pretraining/11-pretraining-in-practice/
torchrun --standalone --nproc_per_node=1 train.py --config=configs/demo_a100.yaml
```

Expected wallclock: **~6–10 hours** on A100-80GB at ~25% MFU. Cost on RunPod/Lambda: ~$15–25.

### Single H100 (same run, faster)

If your provider offers H100-80GB, use the H100-tuned config — same model, same token budget, just with activation checkpointing turned off to recover the ~33% FLOP tax:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/demo_h100.yaml --gpu=H100
```

Expected wallclock: **~1.5–3 hours** on H100-80GB at ~35–50% MFU. Cost on RunPod (~$2–3/hr for H100): ~$5–10. The `--gpu=H100` flag is only used for the MFU printout — it tells `train.py` to compare against H100's 990 BF16 TFLOPs instead of A100's 312. The training itself reads the dtype and shape from the YAML.

The diff against the A100 demo is a single field (`activation_checkpointing: false`); the rationale is in the header of [`configs/demo_h100.yaml`](configs/demo_h100.yaml). You might expect a bigger micro-batch on H100 too, but at vocab=151,936 the `B·S·V` logits tensor and the FP32 cross-entropy intermediates dominate memory, not parameters or activations — so the H100's extra HBM is already spoken for by the CE step at bs=8. Pushing bs higher needs chunked CE or a Liger fused kernel (stretch goal). FP8 on H100 is another lever — ~40% on top of BF16 — but needs Transformer Engine wiring in `model.py` that isn't included here; also a stretch goal in §11.

### 8×A100 node — same config, ~8× faster

The same config with `--nproc_per_node=8` and `grad_accum` cut 8×:

```bash
torchrun --standalone --nproc_per_node=8 \
    train.py --config=configs/demo_a100.yaml \
    --training.grad_accum=8
```

Tokens-per-step are preserved (1M either way), but each step's micro-batches run in parallel across ranks instead of sequentially. Finishes in ~45–90 minutes. Same effective batch size, same loss curve to within noise. (On 8×H100, use `demo_h100.yaml` with `--training.grad_accum=8` for the same reason.)

### Multi-node — 4 × 8 GPUs

```bash
# On every node:
torchrun --nnodes=4 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=node-0:29500 \
    train.py --config=configs/demo_a100.yaml \
    --training.grad_accum=1     # 32 GPUs × bs 8 × 1 × 2048 ≈ 524k tokens/step;
                                # halve total_steps if you want to keep tokens fixed
```

No code changes. FSDP2 + DCP handles the sharding; the only flags that move are torchrun's rendezvous setup.

### Smoke test — CPU, synthetic data, 10 steps

For development:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
    --config=configs/demo_a100.yaml \
    --data.source=synthetic \
    --training.total_steps=10 \
    --training.activation_checkpointing=false \
    --model.d_model=128 --model.n_layers=2 --model.d_ffn=256 \
    --data.batch_size_per_device=2 --data.seq_len=128
```

Runs in ~30 seconds on a laptop. Useful for verifying changes before launching the real thing.

## 6. Monitoring while it runs

The training loop logs every `log_every` steps to stdout (always) and to **Weights & Biases** (if configured). The W&B integration is wired into `train.py` — you turn it on via config, not code.

### Logging to W&B

Three steps:

**1. Install the SDK and authenticate (once per machine):**

```bash
pip install wandb
wandb login           # interactive — paste your API key from https://wandb.ai/authorize
# or, for headless / CI:
export WANDB_API_KEY=...
```

**2. Set the project in your config.** Edit your demo YAML (`configs/demo_a100.yaml` or `configs/demo_h100.yaml`) or pass on the CLI:

```yaml
training:
  wandb_project: llm-lab-pretrain      # your project; empty = W&B disabled
  wandb_entity:  your-username         # or team slug; leave "" to use your default
  wandb_run_name: 150M-fineweb-demo    # "" = let W&B auto-name
  wandb_tags: ["demo", "fineweb-edu", "150M"]
```

Or override at launch:

```bash
torchrun ... train.py --config=configs/demo_a100.yaml \
    --training.wandb_project=llm-lab-pretrain \
    --training.wandb_run_name=150M-fineweb-demo
```

**3. Watch.** `train.py` prints the run URL on startup; click it. The metrics logged every `log_every` steps: `loss`, `grad_norm`, `lr`, `tokens_per_second`, `mfu`, `step`. The full `TrainConfig` is dumped as the run's hyperparameters.

**Resuming a W&B run after a crash / restart.** Set `wandb_run_id` to the previous run's id (visible in its URL and in `wandb_run.id` on the run page). `train.py` passes `resume="allow"` so the new process appends to the same run instead of starting a fresh one:

```bash
torchrun ... train.py --config=configs/demo_a100.yaml \
    --training.resume_from=./results/checkpoints/step_00002000 \
    --training.wandb_run_id=abc12def         # the existing run's id
```

**Multi-rank etiquette**: only rank 0 calls `wandb.init` and `wandb.log`. Other ranks are silent — they have no metrics to add beyond what rank 0 reduces. The training loop already enforces this.

**Disabling**: set `wandb_project=""` (the default). No W&B import is attempted; stdout logging is unchanged.

**Two W&B-specific gotchas this module fixes for you.** Both have the same symptom — the wandb run URL prints, then the loop never starts — and both are easy to chase for hours if you don't know about them:

1. **`wandb.init(..., settings=wandb.Settings(console="off"))`.** By default wandb redirects stdout/stderr at the file-descriptor level so it can mirror your terminal into the run page. Under `torchrun` (and in Lightning AI / Docker / other non-TTY launches) that redirect can deadlock the writer pipe right after `wandb.init` returns. `console="off"` keeps prints on the local terminal and skips the mirror — the right trade for a GPU-rented run where silence is the worst outcome.
2. **`DataLoader(..., multiprocessing_context="forkserver")` whenever `num_workers > 0`.** The standard PyTorch DataLoader forks worker subprocesses on the first batch. `wandb.init` starts background threads (heartbeat, uploader, file watcher) *before* that fork happens, and a POSIX `fork()` in a process that already has threads is undefined behavior — it deadlocks reliably when any of those threads hold a lock at the fork instant. `"forkserver"` routes the fork through a tiny helper process with no pre-existing threads, so workers spawn safely. (Without wandb the main process has no extra threads, fork is fine, and you'd never see this. Add any thread-spawning library and the latent bug surfaces.)

### The metrics that matter

In order of "how often does watching this save a run":

| Metric | What "healthy" looks like | What "bad" looks like |
|---|---|---|
| **loss** | Smooth descent; ends ~2.8–3.3 for the 150M demo on FineWeb-Edu | Spikes, plateaus, NaN, oscillation |
| **grad_norm** | Stable 0.3–1.5 after warmup | Sustained > 5, sudden spikes to 100+ |
| **lr** | Follows your scheduler exactly | Mismatch = scheduler not wired correctly |
| **tokens/sec** | Stable around the model's roofline | Drops = data loader stall or shared GPU |
| **MFU (model FLOPs utilization)** | 20–50% on FSDP2 for this size | < 10% = comm-bound or framework overhead |
| **HBM used (peak)** | Stable each step; some room | Slowly growing → activation/optimizer leak; spikes to OOM |

**The math for tokens/sec → MFU**:

```
training_FLOPs/step = 6 × N × tokens_per_step      (the 6ND rule from Module 10)
GPU_peak_FLOPs/s    = 312 TFLOPs (A100 BF16)
MFU                 = (training_FLOPs/step / step_time) / GPU_peak_FLOPs/s
```

`train.py` computes this for you and prints it every `log_every`. If MFU drops below 20% for the 150M-Qwen3 shape, something is wrong (data loader, network, or you forgot FSDP2).

**Free memory headroom** is your buffer for spikes. On A100-80GB with this config you should have ~25–30 GB free. If you're under 10 GB, a long sequence batch will OOM — increase grad_accum or enable activation checkpointing if it isn't already.

## 7. Reading the loss curve — and what to do when it spikes

The expected loss trajectory for the 150M / 3B-token demo:

| Step | Loss (approx) | What's happening |
|---|---|---|
| 0 | 11.9 | Random-init — log(vocab_size) ≈ ln(151936) = 11.93 |
| 100 | ~5.5 | End of warmup; the model has learned that frequent tokens are frequent |
| 500 | ~3.8 | The shape of grammar starting to emerge |
| 1500 | ~3.2 | Mid-run plateau on subtler structure |
| 3000 | ~2.8–3.0 | Final — what `results/loss_curve.png` shows |

A run that ends much higher than 3.2 is probably under-converged (LR too low, or you ran fewer tokens than 20:1). A run that *spikes* mid-training looks like this:

```
loss
 │ ╲___________
 │            ╲___╱╲╱╲___          ← a spike, then recovery
 │                      ╲___
 │                          ╲___
 │                              ╲___╱╲___
                              ↑
                          a SECOND spike — bad, the run is unstable
       steps →
```

**Single-spike-with-recovery**: usually a bad batch (a contaminated document, an extremely long URL, a non-text glyph the tokenizer balked on). Grad norm jumps to 30–100 in one step. Gradient clipping caught it, the optimizer dampened the update, training continued. **Action: log the batch, but don't restart.**

**Repeated spikes**: LR is too high for the current regime, or the data is consistently nasty. Grad norm stays high (>5) between spikes. **Action: rollback to the last clean checkpoint, reduce LR by ~30%, resume.** The framework supports this:

```bash
torchrun ... train.py --config=configs/demo_a100.yaml \
    --training.resume_from=./results/checkpoints/step_00002000 \
    --optimizer.lr=2.0e-4
```

**Loss diverges to NaN**: BF16 overflow somewhere. On A100 / H100, BF16 is wide enough that this is almost always a data bug (degenerate sample). On older hardware running FP16 without GradScaler, it's the precision. **Action: switch to FP32 master + BF16 compute (already the case for the demo), inspect the recent N batches.**

**Loss plateaus too early**: LR-min-ratio aggressive, or simply enough tokens. If the eval loss is also plateaued, you're done — extend training only if you have more high-quality data.

## 8. Checkpoint resume

Checkpoints are sharded via `torch.distributed.checkpoint` (DCP) — every rank saves its own shard into one directory:

```
results/checkpoints/
    step_00000500/    # 6 files for rank 0..5 on an 8-GPU node
    step_00001000/
    step_00001500/
    ...
```

### What's inside a single `step_XXXXXXXX/` directory

If you peek inside one of these folders, you won't find a single `model.bin` — you'll find two kinds of files, both required:

```
step_00000500/
    .metadata           # the index: which tensor lives in which shard, and where
    __0_0.distcp        # rank 0's shard of model + optimizer + RNG + step
    __1_0.distcp        # rank 1's shard         ← only present if you trained on >1 GPU
    __2_0.distcp        # ...
    ...
```

- **`.metadata`** is the index. It lists every persisted tensor — `model.layers.0.attn.q_proj.weight`, `optimizer.state.0.exp_avg`, the integer `step`, the RNG blobs — together with their global shape, dtype, and a pointer to which `.distcp` file holds each slice.
- **`__<rank>_<n>.distcp`** is a binary shard, one file per rank (the `_<n>` suffix is an internal chunk counter, almost always `_0` for our model sizes). The `0` in `__0_0.distcp` is the rank that wrote it.

So a checkpoint saved on **1 GPU** is exactly `.metadata` + `__0_0.distcp` — that's the complete artifact, nothing is missing. On 8 GPUs it's `.metadata` + `__0_0.distcp` … `__7_0.distcp`. Step 500 of your demo run looking like just two files is the expected, healthy state.

**Why this format instead of one big file.** Each rank writes its local shard straight to disk in parallel, no rank-0 bottleneck collecting weights first. For a 70B model the difference is a ~30-second save vs a ~30-minute one, and the latter blocks training. DCP also re-shards on load — the same checkpoint can rehydrate into a model with a *different* number of GPUs, which is what enables the resume-on-different-cluster-shape trick below.

### Resuming on a different cluster shape

```bash
# Trained on 8 GPUs, resume on 4:
torchrun --standalone --nproc_per_node=4 train.py \
    --config=configs/demo_a100.yaml \
    --training.resume_from=./results/checkpoints/step_00001500
```

The optimizer state, scheduler step count, RNG state, and global step counter all restore. The next step picks up exactly where the previous run left off — no LR jump, no duplicated updates.

### Resume correctness — what's guaranteed, what isn't

Pretraining is expensive. If a 10-hour run crashes at hour 8, you want the resume to produce the *exact same model* you'd have gotten without the crash. The framework guarantees this for everything **except the data loader's position**, with one caveat.

**What's persisted in every checkpoint:**

| State | How |
|---|---|
| Model weights | `model.state_dict()` via DCP (or `torch.save`) |
| Optimizer state | `optimizer.state_dict()` — Adam moments, step counter, etc. |
| Scheduler state | `scheduler.state_dict()` — `last_epoch`, ensuring LR aligns exactly |
| Step count | The number of optimizer updates already completed |
| PyTorch CPU RNG | `torch.get_rng_state()` |
| PyTorch CUDA RNGs | `torch.cuda.get_rng_state(i)` for every device |

**What's NOT persisted (and why this can matter):**

- **Data-loader position in the stream.** `FineWebEduDataset` is a streaming `IterableDataset`. On resume, a fresh iterator opens the dataset at position 0 with the same shuffle seed — so the next batch will be the *first batch the original run saw*, not the batch that would have come next. The model **replays** the first `start_step × tokens_per_step` tokens before reaching new content.

What this means in practice:

- **For the 3B-token demo**: replaying ~1–3% of tokens after a single mid-run crash is annoying but doesn't meaningfully hurt the final model. Loss curves will look fine.
- **For frontier-scale runs (T-trillion tokens)**: this *is* a real loss of compute and reproducibility. Frontier teams use data-iterator checkpoints — Megatron-LM and TorchTitan both serialize the position into the global shuffled corpus index. Adding that to `data.py` is ~50 lines but uses internal-API HF `datasets` patterns that aren't stable across versions; we leave it as a stretch goal.

**Verifying the rest**: [`tests/test_resume.py`](tests/test_resume.py) builds a tiny model, runs N steps uninterrupted, then runs N steps with a save-and-reload in the middle, and asserts every loss + LR + parameter + optimizer-state value matches *to zero* between the two runs. Run it any time you change `train.py` or `checkpoint.py`:

```bash
python tests/test_resume.py
```

If this test fails, **do not deploy the framework to a long run** — something in the resume path has regressed.

### Managing disk — checkpoint rotation

A frontier-sized model can produce a 100+ GB checkpoint *per save*. Saving every 500 steps over a long run will exhaust any disk you point it at. The framework prunes automatically — two complementary policies you configure together:

```yaml
training:
  keep_last_n_checkpoints: 3       # rolling window of the 3 most-recent checkpoints
  milestone_every: 1000            # plus: keep step_1000, step_2000, ... forever
```

After every save, `train.py` calls `checkpoint.cleanup_old(...)`:

1. **Rolling window** — keep the `keep_last_n` most recent checkpoints (here: the last 3).
2. **Milestones** — *additionally* keep any checkpoint whose step is a multiple of `milestone_every` (here: every 1000). These never get pruned.

The union of those two sets stays on disk; everything else is deleted (recursive, idempotent — safe across crashes).

**For the demo** (3000 steps, save every 500) the directory after a successful run contains:

```
results/checkpoints/
    step_00001000/   ← milestone
    step_00002000/   ← milestone + within rolling window
    step_00002500/   ← within rolling window
    step_00003000/   ← final + within rolling window + milestone
```

Four directories, not six. Steps 500 and 1500 were rolled out.

**Disabling pruning entirely**: set `keep_last_n_checkpoints` to a huge number (e.g. 1000000). Setting it to 0 raises — we require at least 1 (the latest, so resume works).

**Disabling milestones**: set `milestone_every: 0`. Then only the rolling window survives.

**Sizing rule of thumb**: a checkpoint of an N-param model in BF16 with AdamW state ≈ `N × 16 bytes`. For 150M that's ~2.4 GB per checkpoint; for 70B it's ~1.1 TB. With `keep_last_n=3` and one milestone per ~5% of training, you cap disk at ~3.3× and ~3.3 TB respectively.

### Pulling a checkpoint off the training box for local inference

A common workflow: training runs on a rented GPU box, but you want to vibes-check the model — generate completions, poke around in a notebook, hand it to a teammate — from your laptop. The DCP format is fine for this; you just need to know two things.

**1. Transfer the directory as a unit.** A DCP checkpoint *is* a directory — copy the whole `step_XXXXXXXX/` folder, not individual files:

```bash
# from your local machine
rsync -avh --progress \
  user@host:/.../11-pretraining-in-practice/results/checkpoints/step_00003000 \
  ./results/checkpoints/
```

`scp -r ...` works too; `aws s3 sync` / `gsutil -m rsync -r` are noticeably faster when there are many small shard files (large multi-GPU runs).

**2. Load it locally — two ways, and you need to pick one deliberately.** Look at [`checkpoint.py`](checkpoint.py)'s `load()` function: there's a `dist.is_initialized()` branch that reads the DCP shards via `dcp.load`, and a non-distributed branch that reads a single `state.pt` via `torch.load`. These are two **different on-disk formats** — running `python eval.py --checkpoint=step_00003000/` on a downloaded DCP folder falls into the non-distributed branch and errors on the missing `state.pt`. Pick one:

**Option A — keep DCP, run eval under torchrun (1 process).** No conversion. The distributed branch handles single-rank fine; you're just borrowing `torchrun` to set up the process group.

```bash
torchrun --standalone --nproc_per_node=1 eval.py \
    --checkpoint=./results/checkpoints/step_00003000 \
    --config=configs/demo_a100.yaml \
    --device=cpu --prompt="The capital of France is"
```

This is the path that requires zero extra steps if you've kept the full `11-pretraining-in-practice/` directory locally. Use `--device=cpu` if your laptop has no GPU; the 150M demo runs slowly but correctly on CPU.

**Option B — convert DCP → `state.pt` once, then load with plain Python.** Better when you want to load the checkpoint in a notebook, hand a single file to someone else, or run on a machine where setting up `torchrun` is annoying. PyTorch ships the converter as a module:

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch_save \
    ./results/checkpoints/step_00003000 \
    ./results/checkpoints/step_00003000/state.pt
```

After that, `state.pt` lives alongside the `.distcp` shards (no conflict — `load()` only looks at `state.pt` in the non-distributed branch and only at the shards in the distributed branch). Then:

```bash
python eval.py \
    --checkpoint=./results/checkpoints/step_00003000 \
    --config=configs/demo_a100.yaml \
    --device=cpu --prompt="The capital of France is"
```

just works — no `torchrun` needed.

Both options produce **bit-identical** loaded weights; only the on-disk encoding differs. Option A is canonical (it's what the framework uses internally); Option B is the ergonomic choice for laptop play.

**Sanity expectations for an early checkpoint.** If you're loading something from very early in training (e.g. step 500 of the 3000-step demo), the model has seen only ~3% of its token budget. Don't be alarmed if perplexity is in the hundreds and completions read as word-shaped but semantically incoherent. The honest vibes-check at that stage is "does it produce English tokens at all," not "does it know facts." Re-pull a later checkpoint (step 2000+) before judging the run.

## 9. Evaluating a base model

A base model (no SFT, no RLHF) is hard to evaluate well. We provide three signals:

**1. Validation perplexity** (`eval.py`). Run on a held-out 50M-token slice of FineWeb-Edu the training never saw. Perplexity = $\exp(\text{loss})$. The demo run should land around **15–22**.

```bash
python eval.py --checkpoint=./results/checkpoints/step_00003000 --slice=valid
```

**2. Generation sanity check** (`eval.py --generate`). Sample 5 short completions from canonical prompts. The model should produce coherent English-shaped text. It will not answer questions or follow instructions (that's post-training, Part 4).

```
Prompt: "The capital of France is"
Output: " Paris, which is one of the most-visited cities in the world..."
```

If the output is gibberish, something is wrong even if perplexity looks fine.

**3. Downstream benchmarks** — point at lm-evaluation-harness. The demo model is too small to score meaningfully on most benchmarks, but `arc_easy`, `piqa`, `hellaswag` will be at 25–35% (above random, below an instruction-tuned model). The `eval.py --harness` flag prints the lm-eval-harness command to run.

**Perplexity is necessary, not sufficient.** A model can have great perplexity on FineWeb-Edu and still fail on out-of-distribution prompts. Always pair it with generation samples on at least 5 prompts. After post-training you'll lean on benchmarks; for a base model, "does this read like English?" is the honest test.

## 10. Scaling up — 150M → 1B → 7B → 70B

Everything in this directory is sized for 150M, but **nothing in the code assumes it**. To train a 1B model, copy `configs/demo_a100.yaml` (or `demo_h100.yaml`) and edit:

```yaml
model:
  d_model: 2048
  n_layers: 24
  n_heads: 16
  n_kv_heads: 4
  d_ffn: 5440           # ~2.66× d_model
training:
  total_steps: 20000    # Chinchilla: ~20 × 1B = 20B tokens at 1M/step
optimizer:
  lr: 1.5e-4            # Halved roughly per width doubling (muP-shaped)
```

The same `torchrun train.py` works. What changes is hardware: 1B on a single A100 is tight (turn on `activation_checkpointing=true`); 1B on 8×A100 is comfortable; 7B needs 8×A100 minimum; 70B needs multi-node FSDP and likely tensor parallelism (see Module 10 § 5).

**The rules of thumb when scaling**:

- **Peak LR** drops with model width (per Module 09): rough rule LR ∝ 1/√d_model. Use muP if you want to skip the sweep.
- **Warmup** stays roughly fixed in absolute steps — 200–2000 — not proportional to total steps.
- **Total tokens** scale with N at 20:1 Chinchilla unless you're deliberately over-training (Llama 3 at 200:1) for inference cost.
- **Activation checkpointing** is on by default once $L \times d \times N_{\text{layers}}$ gets big. The wallclock tax is ~33%; the memory headroom is non-negotiable.

## 11. Stretch goals

The framework supports the following with config-only changes (or one extra import):

- **FP8 training on H100/H200**: `--training.dtype=fp8`. Requires `transformer-engine` installed; `model.py` swaps in `te.Linear` for the hidden weights. ~40% wallclock savings vs BF16. Module 10 § 4.
- **WSD schedule** instead of cosine: `--schedule.type=wsd`. Use this if total_steps is uncertain or you want to extend mid-run. Module 09 § 4.
- **Muon optimizer** for the hidden 2D weights, AdamW for everything else: `--optimizer.type=muon`. Module 08 § 4.
- **Multi-token prediction (MTP)**: not wired by default. To add it, attach 2 small auxiliary heads to the final hidden state and add their cross-entropy with weight 0.3 each. ~40 lines on top of `loop.py`. Module 10 § 7.
- **Longer context (8k, 32k)**: bump `max_seq`. With activation checkpointing on, 8k is feasible on A100-80GB. For 32k+ you want FlashAttention (already implicit in HF Qwen3) + selective checkpointing (Module 10 § 3 selective checkpointing note).

## 12. Reading list

- **FineWeb-Edu**: Penedo et al., "The FineWeb Datasets" (HuggingFace, 2024). The corpus we trained on. Read at least Section 4 (the quality filter).
- **The Llama 3 training recipe**: Meta AI, "The Llama 3 Herd of Models" (2024), Section 3. Cross-reference your monitoring against theirs.
- **DeepSeek-V3 Technical Report** (2024). Section 5 (training) is the clearest open description of a real frontier run, including incident-recovery decisions.
- **The Chinchilla compute-optimality paper**: Hoffmann et al. (2022), and the Llama 3 anti-Chinchilla section (Meta AI 2024, § 3.4) for the trade-off.
- **W&B's "Reproducible AI" playbook**: practical advice on logging, checkpoint hygiene, and run organization. Worth a skim before launching your first long run.

## 13. What's next

Part 3 ends here. The next module is [Part 4 — Post-Training](../../part-4-post-training/) and Module 12, which begins with the post-training landscape: SFT, preference optimization (DPO/IPO/KTO), reasoning (GRPO), and distillation (offline, on-policy, SDFT). The base model you produce in this module is the starting point for Module 13 (SFT) — though for cost reasons Part 4 actually uses **Qwen 3.6 1.7B** as the post-training base, so the techniques are demonstrated on a more capable model than this 150M demo.

Your 150M demo model is yours, though. Save the checkpoint somewhere — it's a real artifact, trained on real data with the real recipe. The next time someone says "I've never trained an LLM from scratch," you'll be able to disagree.
