# Module 08 — The Training Loop

> Part of [Part 3 — Pretraining](../). Reading time: ~90 minutes. Compute cost: ~$0–2 (CPU is enough for the walkthrough; one A100 hour if you want to run a real distributed step).

## The thesis

Pretraining isn't a different kind of programming. It's one loop, written carefully:

```
for step in range(total_steps):
    batch = next(loader)
    loss = forward(model, batch)
    loss.backward()
    clip_gradients(model)
    optimizer.step()
    optimizer.zero_grad()
    log_metrics()
    maybe_checkpoint()
```

Twelve lines. Frontier labs spend months on the dressings — distributed sharding, mixed-precision, accumulation, async data loading, fault recovery, monitoring — but the inner shape never changes. Every one of those concerns is *additive* to those twelve lines. This module builds the loop one concern at a time, in the order that makes it production-shaped from the first commit.

By the end of this module you have:

- A working `torchrun train.py` that launches on 1 GPU or 64 GPUs from the same code.
- A model builder that defaults to Qwen3 but takes ~5 lines to swap.
- AdamW with the param groups frontier labs actually use.
- BF16 mixed precision, gradient accumulation, gradient clipping, FSDP2 sharding, and FSDP-aware checkpointing — all wired in.
- A complete framework you could `cp -r` into another repo and pretrain a model with.

The next modules (09, 10, 11) add the learning-rate schedule, the scaling story, and the real dataset. The loop itself is finished here.

## What you'll be able to do at the end

- Read any pretraining script and identify which of the 12 concerns it's addressing.
- Build a training loop from scratch that scales from 1 to many GPUs without rewriting.
- Choose BF16 / FP16 / FP8 with awareness of what each costs and what each saves.
- Pick a reasonable AdamW configuration without guessing.
- Diagnose the most common training-loop failure modes (NaN at step 0, OOM, slow data loader, dead grads).

## 1. The framework layout

This module ships the framework directory the rest of Part 3 builds on:

```
08-training-loop/
  config.py        # TrainConfig dataclass — every knob
  model.py         # build_model(cfg) — defaults to Qwen3
  optim.py         # build_optimizer(...) — AdamW with param groups
  loop.py          # train_step() and forward_loss() — the inner loop helpers
  data.py          # SyntheticDataset + make_dataloader — replace in Module 11
  checkpoint.py    # save/load with FSDP2-aware sharded state dicts
  fsdp_setup.py    # init_distributed() and apply_fsdp() — torchrun + FSDP2 wrapping
  train.py         # the entrypoint you launch with torchrun
  notebook.ipynb   # CPU-friendly walkthrough of each component
```

Modules 09–11 add files (LR schedule, FSDP advanced patterns, real data loaders) but don't rewrite anything here. The contract is stable.

## 2. Anatomy of a training step

Look at what one optimizer step actually does, broken into the smallest meaningful pieces:

```python
optimizer.zero_grad(set_to_none=True)        # 1. Clear old gradients

for _ in range(grad_accum_steps):            # 2. Accumulate gradients over
    batch = next(loader)                     #    micro-batches before stepping
    with autocast(dtype=bf16):
        loss = forward_loss(model, batch) / grad_accum_steps  # 3. Mixed precision
    loss.backward()                          # 4. Backward — accumulates into .grad

grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)  # 5. Clip gradients

optimizer.step()                             # 6. AdamW update

# 7. Logging and checkpointing happen here, not in the step itself
```

Seven concerns, six lines of substance. Each gets its own section below.

## 3. The model — defaulting to Qwen3, swap-friendly

For pretraining you need a model. Part 2 spent four modules teaching how to build one; for the practical framework here, we use `transformers.Qwen3ForCausalLM` with the config initialized to whatever architecture you want.

```python
# 08-training-loop/model.py
from transformers import Qwen3Config, Qwen3ForCausalLM

def build_model(cfg: ModelConfig) -> nn.Module:
    config = Qwen3Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        intermediate_size=cfg.d_ffn,
        max_position_embeddings=cfg.max_seq,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_weights,
        rms_norm_eps=cfg.norm_eps,
    )
    return Qwen3ForCausalLM(config)
```

That's the whole file. `Qwen3ForCausalLM(config)` creates a randomly-initialized model — same as if you'd called your own `TransformerLM(cfg)`.

**Why Qwen3, why not our Part-2 model.** Two reasons:

1. *Pedagogical separation.* Part 2 taught you how every component works. Part 3 teaches how to train. Using a model you didn't write yourself proves the loop is the loop — it doesn't care which transformer is inside it.
2. *Production reality.* In 2026 you don't ship your own transformer for ordinary pretraining. You take a frontier architecture from HF or a research-lab repo and train it. The Qwen3 implementation is correct, FlashAttention-2-enabled, FSDP2-compatible, and well-tested across thousands of fine-tunes.

**To swap in a different architecture**, edit `model.py`:

```python
# Llama: same shape, different config class
from transformers import LlamaConfig, LlamaForCausalLM

def build_model(cfg):
    return LlamaForCausalLM(LlamaConfig(...))

# Your Part-2 TransformerLM: import and instantiate
import sys; sys.path.append("../../part-2-architecture/07-full-model")
from model import TransformerLM, ModelConfig as Part2ModelConfig

def build_model(cfg):
    return _Part2Adapter(TransformerLM(Part2ModelConfig(...)))
```

The adapter is one wrapper class that translates `(input_ids=..., labels=...)` to your model's interface and returns an object with `.loss` and `.logits`. ~20 lines. We don't ship it; if you need it, you write it where you need it.

## 4. The optimizer — AdamW (with a Muon alternative)

The production-canonical optimizer for LM pretraining is **AdamW**. Every frontier 2026 model — Llama, DeepSeek, Qwen, Mistral — ships AdamW. The hyperparameters are stable enough that you read them once and rarely tune from scratch:

| HP | Value | Why |
|---|---|---|
| $\beta_1$ | 0.9 | Standard. Higher = more momentum; rarely changed. |
| $\beta_2$ | 0.95 | Llama / DeepSeek / Qwen use 0.95. Default PyTorch is 0.999 — use 0.95 for LLMs. |
| eps | 1e-8 | Default. Some labs use 1e-10 for FP8 stability. |
| weight_decay | 0.1 | Standard. Applied to weights, NOT to norms/biases (see below). |

**The param groups split.** AdamW applies weight decay to every parameter unless you tell it not to. Two groups of parameters where weight decay actively hurts:

1. **Norm `gamma` parameters** (RMSNorm, LayerNorm). These are scale factors that the model uses to calibrate the residual stream's magnitude. Decaying them toward zero is decaying *the model's calibration*, which is what you don't want.
2. **Bias terms** (rare in modern transformers, but if any). Decaying biases doesn't help and isn't standard.

Some labs also exclude **embeddings** from decay; this is more contested. Llama 3 decays embeddings; DeepSeek-V3 does not. The course defaults to *decay* embeddings (Llama convention) because tied weights mean the embedding *is* the LM head, and decaying the head matters.

```python
# 08-training-loop/optim.py
def build_optimizer(model, lr, betas, weight_decay, eps):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No weight decay for any 1D parameter (norms and biases are 1D).
        if p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=eps,
    )
```

The `p.ndim < 2` trick catches RMSNorm `gamma` (shape `(d_model,)`) and any biases automatically. Cleaner than name-matching, robust to layer renames.

When Module 09 lands the LR schedule and Module 07's muP groups, this signature accommodates both — we'll add a `param_groups` argument that lets the caller pass custom group splits.

### Muon — the 2025 alternative worth knowing about

AdamW is universal at frontier production scale, but it's not the only thing being tried. **Muon** (Keller Jordan, late 2024) replaces AdamW for the 2D weight matrices in a network with a Newton-Schulz orthogonalized momentum update.

The mechanism in one paragraph: compute momentum as in SGD, then apply ~5 Newton-Schulz iterations that approximate the matrix sign function. The result is a momentum tensor with approximately-unit singular values, which means each singular direction of the weight gets a comparably-sized update — instead of the most-active directions dominating. Embeddings, norms, and biases still use AdamW; a real "Muon training" is a hybrid: Muon on hidden 2D matrices, AdamW on everything else.

**Where it stands in 2026** (honest read):

- *Empirically*: Jordan and replicators report ~30–50% wall-clock speedup vs AdamW at matched loss on small/mid scale. Lower memory too — no second-moment $\hat v_t$.
- *Scale validation*: works at least to ~16B parameters (Moonlight 2025).
- *Frontier production*: **Llama 3, DeepSeek-V3, Qwen 3 are all still AdamW**. Muon hasn't displaced AdamW in any shipped frontier model as of early 2026, but the trajectory looks favorable.

**The framework ships both.** Set `cfg.optimizer.type = "muon"` (or pass `--optimizer=muon` on the CLI) and `build_optimizer` returns a `MuonAdamW` hybrid with the canonical split. Implementation lives in [`muon.py`](muon.py).

```python
# Switch to Muon: one config field.
optimizer = build_optimizer(model, OptimizerConfig(type="muon", lr=3e-4, muon_lr=2e-2))
```

The hybrid optimizer is still a `torch.optim.Optimizer` subclass — so it composes with any `LRScheduler`, gets the same gradient clipping treatment, checkpoints through DCP just like AdamW. The training-loop changes are zero.

**When to pick which** (rough guide):

| You're... | Use |
|---|---|
| Pretraining for production at any scale | AdamW. Battle-tested, scales known. |
| Running an experiment where wall-clock matters more than reliability | Muon — accept the lower-coverage validation in exchange for the speedup |
| Doing a research ablation comparing optimizers | Both, with everything else fixed |
| Following the next frontier-model release | Watch this space — Muon adoption depends on a top-3 lab shipping it as the production default |

The course's pretraining demo (Module 11) uses **AdamW**, but the muon path is wired and tested. Try it.

## 5. Mixed precision — BF16, FP16, FP8

A modern training run is **never pure FP32**. Memory and compute savings are enormous, and the numerical care is well-understood.

**BF16 (bfloat16)** is the default. 16 bits, but with a wider exponent range than FP16 — the same range as FP32. This is the *point* of BF16: you get FP16's memory/compute footprint without FP16's overflow problems.

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = forward(model, batch)
loss.backward()
```

Weights stay in FP32 (the *master weights*). Activations and gradients flow through BF16 ops. AdamW state stays in FP32. No loss scaling needed — BF16's exponent range handles ordinary LM gradients without overflow.

**FP16 (half-precision)** is the *old* recipe — pre-2022. Narrower exponent range than BF16, so gradients overflow easily; needs `GradScaler` to dynamically scale losses up before backward. Works, but is fiddly, and the only reason to use it over BF16 is hardware that doesn't support BF16 (older Volta GPUs). **A100 and newer support BF16 natively** — use BF16. We don't ship FP16 support in the course framework.

**FP8** is the frontier (2024+). 8-bit floats. Two formats: E4M3 for forward, E5M2 for backward (different exponent allocations for different dynamic ranges). Needs:

- H100 / H200 / B100 GPUs (Hopper or newer); Ampere does NOT support FP8 matmul.
- `transformer-engine` library (NVIDIA-provided) or PyTorch 2.6+'s native `torch.float8_*` types.
- More careful scaling: per-tensor scale factors, sometimes per-block.

FP8 cuts memory and compute roughly in half vs BF16. DeepSeek-V3 reports a ~40% speedup at training time over BF16 on H800s. We cover FP8 in Module 10; the framework here is BF16-default with FP8 as an optional path documented in `config.py` (`dtype="fp8"`).

**The rule of thumb for 2026**:

- A100 or older: BF16.
- H100 or newer: BF16 for development, FP8 for production large runs.
- V100 or T4: FP16 (and consider switching to a different cloud).

## 6. Gradient accumulation

The effective batch size is `B_eff = B_per_device × world_size × grad_accum_steps`. You want `B_eff` set by the problem (typically 1M–4M tokens per step for LLM pretraining), not by what fits in GPU memory. Accumulation lets you decouple them.

```python
optimizer.zero_grad()
for _ in range(grad_accum_steps):
    micro_batch = next(loader)
    loss = forward(model, micro_batch) / grad_accum_steps  # scale to mean
    loss.backward()                                         # ACCUMULATES into .grad
optimizer.step()
```

Three details:

1. **Divide the loss** by `grad_accum_steps`. PyTorch's `backward()` *adds* gradients to the existing `.grad` attribute. After K micro-batches you have K times the gradient of one micro-batch — too big by a factor of K. Pre-dividing the loss fixes this.
2. **`zero_grad(set_to_none=True)`** is preferred over `zero_grad()`. Setting to `None` skips a memset and lets the next backward allocate fresh tensors at the right dtype.
3. **No `optimizer.step()` inside the inner loop.** Step only after all micro-batches have contributed.

For FSDP, gradient accumulation has one extra subtlety: by default FSDP all-reduces gradients across ranks at every backward, so you'd be doing the all-reduce K times per step. Use `model.no_sync()` for the first K-1 micro-batches:

```python
for i in range(grad_accum_steps):
    micro_batch = next(loader)
    # No grad sync except on the last micro-batch (then optimizer.step)
    sync = (i == grad_accum_steps - 1)
    ctx = nullcontext() if sync else model.no_sync()
    with ctx:
        loss = forward(model, micro_batch) / grad_accum_steps
        loss.backward()
```

This saves K-1 all-reduces per step. At small models the win is modest; at 70B+ across many nodes it's the difference between 50% and 90% MFU.

## 7. Gradient clipping

A single bad batch can produce a huge gradient that destabilizes training. Gradient clipping bounds the global norm of all gradients to a maximum value, scaling them down proportionally if they exceed it.

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**The value: 1.0** for almost every LLM. This is the Llama/DeepSeek/Qwen default and nobody has found a serious reason to change it.

**What to log**: the *unclipped* gradient norm (returned by `clip_grad_norm_`). A healthy training run has a stable grad norm trajectory; spikes are diagnostic. If grad norm spikes above ~10 during training, you have either a bad batch, a learning rate that's too high, or a model bug. The clipping protects training in the moment; the logged norm tells you what was happening.

**For FSDP**: `clip_grad_norm_` on an FSDP-wrapped model handles the cross-rank communication automatically. Don't do anything special.

## 8. Distributed: `torchrun` + FSDP2 from day one

Every script in this Part is launched with `torchrun`:

```bash
# Single GPU (still distributed, just with 1 rank)
torchrun --standalone --nproc_per_node=1 train.py --config=cfg.yaml

# Single node, 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py --config=cfg.yaml

# Multi-node, 4 nodes × 8 GPUs each
torchrun --nnodes=4 --nproc_per_node=8 \
         --rdzv_backend=c10d --rdzv_endpoint=node-0:29500 \
         train.py --config=cfg.yaml
```

The training script doesn't change between these. `torchrun` populates `RANK`, `LOCAL_RANK`, `WORLD_SIZE` env vars; the script reads them and sets up the process group.

### FSDP2 wrapping

PyTorch 2.4+ ships **FSDP2** as `torch.distributed.fsdp.fully_shard`. Compared to FSDP1, it's based on DTensor, has cleaner mixed-precision semantics, and integrates better with pipeline / tensor parallelism. **We use FSDP2 throughout the course.**

The minimum setup:

```python
# 08-training-loop/fsdp_setup.py
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def apply_fsdp(model, dtype=torch.bfloat16):
    """Shard every transformer layer, then the root module."""
    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    # Shard each transformer layer separately for fine-grained gradient sync.
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    # And the root.
    fully_shard(model, mp_policy=mp_policy)
    return model
```

Two things to understand:

1. **What gets sharded.** Each `fully_shard(module)` call shards *that module's* parameters across the data-parallel group. The layer-level wrap means each layer's params are fetched (all-gathered) when its forward runs, then freed; this is what makes FSDP memory-efficient.
2. **Mixed precision.** `MixedPrecisionPolicy` tells FSDP to keep params in `param_dtype` (BF16) during computation, but reduce gradients in `reduce_dtype` (FP32). This is the standard recipe — BF16 forward/backward, FP32 grad reductions for stability.

**Why we shard by layer.** In `Qwen3ForCausalLM`, `model.model.layers` is a `ModuleList` of `Qwen3DecoderLayer`s. Each layer is the natural unit of "fetch, compute, free." Sharding the root only would force *all* params to all-gather at once, defeating the memory savings.

Module 10 covers the FSDP2 internals (the ZeRO stages, the shard groups, when to mix in tensor parallelism). For this module, the setup above is enough.

## 9. Checkpointing — FSDP2-aware

A real run checkpoints every N steps and resumes from the last checkpoint on restart. Under FSDP2 each rank only holds a shard of the parameters, so checkpointing requires either:

1. **Full state dict** — gather all params to rank 0 (or all ranks) and save as a single file. Simple, but the gather is expensive at scale (rank 0 needs RAM for the whole model).
2. **Sharded state dict** — each rank saves its own shard. Smaller files per rank, but more bookkeeping on load.

For the course framework we use **DCP (`torch.distributed.checkpoint`)**, PyTorch's native distributed checkpointing API. DCP handles sharded saves transparently and lets you load a checkpoint into a *differently sharded* model (e.g., resume on a different number of GPUs):

```python
# 08-training-loop/checkpoint.py
import torch.distributed.checkpoint as dcp

def save(model, optimizer, step, out_dir):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    dcp.save(state, checkpoint_id=f"{out_dir}/step_{step:08d}")

def load(model, optimizer, ckpt_dir):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 0}
    dcp.load(state, checkpoint_id=ckpt_dir)
    return state["step"]
```

`dcp.save` writes per-rank shards as a directory; `dcp.load` reads them back into whatever sharding the current model has. This is the right pattern for production training.

## 10. The complete loop

Putting it together (this is `train.py` in essence — see [`train.py`](train.py) for the full version with argument parsing and logging):

```python
def train(cfg):
    init_distributed()                                                 # § 8
    model = build_model(cfg.model)                                     # § 3
    model = apply_fsdp(model, dtype=cfg.dtype)                         # § 8
    optimizer = build_optimizer(model, **cfg.optimizer)                # § 4
    loader = make_dataloader(cfg.data)                                 # § 1
    start_step = maybe_load_checkpoint(model, optimizer, cfg.resume)   # § 9

    model.train()
    for step in range(start_step, cfg.total_steps):
        # § 6 — gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for accum_step in range(cfg.grad_accum):
            batch = next(loader)
            with torch.autocast("cuda", dtype=cfg.dtype):              # § 5
                loss = forward_loss(model, batch) / cfg.grad_accum
            # § 6 — no_sync on all but the last micro-step
            sync = (accum_step == cfg.grad_accum - 1)
            (nullcontext() if sync else model.no_sync()).__enter__()
            loss.backward()
            total_loss += loss.item()

        grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip) # § 7
        optimizer.step()

        if step % cfg.log_every == 0 and is_main_rank():
            log({"step": step, "loss": total_loss, "grad_norm": grad_norm})

        if step % cfg.save_every == 0:
            save(model, optimizer, step, cfg.checkpoint_dir)           # § 9
```

That's the full inner skeleton. ~25 lines. The framework files in this directory implement each helper.

## 11. Running it

The directory is built so this works on day one:

```bash
cd part-3-pretraining/08-training-loop/
torchrun --standalone --nproc_per_node=1 train.py \
    --total_steps=10 --log_every=1 --vocab_size=2048 --d_model=128
```

This launches a tiny 3M-param run on a synthetic dataset, logs ~10 steps, and exits. It's not a *useful* model, but it proves the loop is wired correctly. Module 11 replaces `SyntheticDataset` with the FineWeb-Edu pipeline and runs at scale.

You can also work through [`notebook.ipynb`](notebook.ipynb) on CPU to see each piece — model building, optimizer construction, one forward/backward pass — without launching torchrun.

## 12. Common failure modes (and what they look like)

| Symptom | Likely cause |
|---|---|
| Loss is NaN at step 0 | Init blew up; check the model's `count_params()` and forward shape. See Module 07 Section 8 (loss-at-init test). |
| Loss is constant | LR is 0, no_sync misuse, or labels are misaligned (off-by-one shift). |
| Loss decreases very slowly | LR too low, or you didn't apply muP scaling for a wide model. |
| Loss spikes mid-training | Bad data sample (Module 11 covers data cleaning), or LR too high for the current grad-norm regime. |
| Grad norm is huge (~100s) | LR too high. Reduce, or check that clipping is enabled. |
| OOM on first step | Reduce per-device batch size and increase `grad_accum`. Or enable gradient checkpointing (Module 10). |
| Data loader is the bottleneck | Increase `num_workers`; check that `pin_memory=True` and the dataset isn't doing CPU work inside `__getitem__`. |
| Throughput drops with more GPUs | Communication overhead — check that you have NVLink (single-node) or InfiniBand (multi-node). Profile with `torch.profiler`. |

This is the punch list. If it's not on it, suspect the data first.

## 13. What's next

- **[Module 09 — Learning Rate](../09-learning-rate/)** adds `schedule.py` with warmup + cosine decay and the muP transfer recipe. This module's `train.py` updates to call the scheduler after each step.
- **[Module 10 — Scaling and Efficiency](../10-scaling-and-efficiency/)** goes deep on FSDP2 — the sharding choices, mixed-precision policies, gradient checkpointing, tensor parallelism, MTP, Chinchilla. Throughput tuning.
- **[Module 11 — Pretraining in Practice](../11-pretraining-in-practice/)** replaces `SyntheticDataset` with the FineWeb-Edu pipeline, ships the actual demo config, and walks through a real run end-to-end with monitoring.

## 14. Reading list

- **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017). The paper that established the AdamW we use.
- **FSDP2 + DTensor**: PyTorch blog, "Introducing FSDP2" (2024). The motivation for the new API.
- **The Llama 3 training recipe**: Meta AI, "The Llama 3 Herd of Models" (2024), Section 3. The clearest open description of a real frontier pretraining loop. Read after this module.
- **DeepSeek-V3 training**: DeepSeek-AI, "DeepSeek-V3 Technical Report" (Dec 2024), Section 5. The FP8 details and the MTP head are particularly clean.
- **Bf16 stability**: Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training" (2019). Why BF16 won over FP16.
