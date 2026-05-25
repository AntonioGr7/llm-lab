# Module 10 — Scaling and Efficiency

> Part of [Part 3 — Pretraining](../). Reading time: ~70 minutes. Compute cost: ~$0 (CPU walkthrough; the techniques start to *pay* on multi-GPU).

## The thesis

Your training script runs. Now you want it to run **on more GPUs, with bigger models, at higher throughput, without wasting compute on the wrong (params, tokens) ratio**. That's this module. Four levers:

1. **Sharding** — how to split params/grads/optimizer states across ranks so one model doesn't have to fit on one GPU.
2. **Activation memory** — how to cut the *other* big memory consumer (activations), trading FLOPs for HBM.
3. **Parallelism shapes** — beyond data parallelism: tensor and pipeline parallelism, and when you actually need them.
4. **Budgeting** — Chinchilla's law of "how many tokens for how many params at this FLOP budget?"

Plus two pieces of 2026 frontier practice: **DeepSeek-V3's multi-token prediction (MTP)** as a free training-signal upgrade, and a **DeepSpeed sidebar** so you recognize it when you meet it.

The framework piece from this module is one file: [`efficiency.py`](efficiency.py) — activation checkpointing for Qwen3-shaped models, plus a Chinchilla budgeting function and a memory-breakdown calculator. Module 11 wires it into the real demo run.

## What you'll be able to do at the end

- Read a frontier-model training config and predict its per-GPU memory budget within ~10%.
- Pick the right FSDP2 sharding stage (`reshard_after_forward` on/off) for your memory headroom.
- Add activation checkpointing in one line and know when it's worth the ~33% FLOP overhead.
- Decide when data parallelism alone is insufficient (model bigger than one GPU's HBM even sharded) and what to reach for next.
- Allocate a fixed compute budget across (params, tokens) using Chinchilla — and know when to deliberately violate it (Llama 3).
- Decide whether MTP is worth adding to your run.

## 1. Where the memory goes

A training step has four big memory consumers per GPU. For an $N$-parameter model in BF16 mixed precision with AdamW:

| Bucket | Bytes per param | What it is |
|---|---|---|
| **Master weights (FP32)** | 4 | The "true" weights AdamW updates. Kept in FP32 for numerical stability. |
| **BF16 weights** | 2 | The cast-down copy used for forward/backward. |
| **Gradients (BF16 or FP32)** | 2–4 | What `loss.backward()` produced. Reduce-scatter can keep these in BF16. |
| **AdamW state (FP32 m, v)** | 8 | Two FP32 vectors per param — first and second moment. |
| **Activations** | — | Depends on batch × seq × depth. Often dominant at large seq. |

Total **fixed per-param overhead**: ~16 bytes for AdamW + BF16 mixed precision. A 1B-param model needs **~16 GB just for params + grads + optimizer state**, before you load a single activation. On a 40GB A100 you have ~24 GB left for activations, the model's own forward buffers, and any framework overhead.

**Activations** are the other half of the story. A rough estimate for a transformer:

$$\text{activation memory} \approx B \cdot L \cdot d_{\text{model}} \cdot N_{\text{layers}} \cdot c$$

where $B$ is batch size, $L$ is sequence length, and $c$ is a small constant (10–30 bytes per "slot," depending on what gets saved — attention probs, projections, MLP intermediates). At $B=2$, $L=8192$, $d=4096$, $N_{\text{layers}}=32$, $c=20$, that's ~40 GB of activations alone. This is the term you cut with checkpointing.

## 2. ZeRO and FSDP2 — the sharding stages

The Zero Redundancy Optimizer paper (Rajbhandari et al., 2019) is the conceptual root of every modern data-parallel training stack. The idea: in vanilla data parallelism, *every rank holds a full copy* of params, grads, and optimizer state. That's wasteful. ZeRO partitions them across ranks.

Three stages, each cumulative:

| ZeRO stage | What's sharded | Memory per rank (with $R$ ranks) |
|---|---|---|
| **Stage 1** | Optimizer states | params + grads + opt_state/$R$ |
| **Stage 2** | + Gradients | params + (grads + opt_state)/$R$ |
| **Stage 3** | + Parameters | (params + grads + opt_state)/$R$ |

At Stage 3, with 8 ranks, your **per-rank memory footprint is 1/8 of the single-rank footprint** for params + grads + opt_state. This is what lets a 70B model train on 8×A100-80GB instead of needing a single GPU with 1.1 TB of HBM.

**FSDP2 is the PyTorch-native implementation of ZeRO-3.** When you wrap with `fully_shard`, parameters are sharded across the data-parallel group; they're all-gathered just-in-time when a layer forwards, and freed right after. Gradients are reduce-scattered across ranks during backward. Optimizer states live sharded forever.

```python
# Module 08's apply_fsdp() — already ZeRO-3.
for layer in model.model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

### The reshard knob

FSDP2 exposes one important memory/throughput knob: **`reshard_after_forward`**.

- `True` (default): after a layer's forward, the all-gathered params are freed. Next time we need them (in backward), we all-gather again. **Min memory, max comm.**
- `False`: keep the all-gathered params live until backward consumes them. **Saves the backward all-gather, doubles peak param memory.**

In practice you set `reshard_after_forward=True` on the layers (where memory dominates) and let the root module keep its small embeddings unsharded. Module 11's demo does exactly this.

**The mapping to ZeRO stages, in FSDP2 vocabulary:**

| You want | Set |
|---|---|
| ZeRO-3 (full sharding, default) | `fully_shard(layer, reshard_after_forward=True)` |
| ZeRO-2-ish (don't reshard params after forward) | `fully_shard(layer, reshard_after_forward=False)` |
| ZeRO-1 (only optimizer state sharded) | Use FSDP1's `ShardingStrategy.SHARD_GRAD_OP` — FSDP2 doesn't expose this granularity directly |

The 2026 default is **ZeRO-3 with `reshard_after_forward=True`**. The communication overhead is real (the backward all-gather doubles param-fetching comm) but at modern NVLink/InfiniBand bandwidths it's been amortized below 10% of step time for typical models. Memory savings dominate the tradeoff.

### When sharding alone isn't enough

If your model + a small activation footprint still doesn't fit in one GPU's HBM *even with ZeRO-3*, data parallelism has run out. That happens around:

- **70B params** on A100-80GB with reasonable batch and sequence length.
- **>200B params** on H100-80GB.

At that point you need tensor parallelism (§ 5) or pipeline parallelism (§ 5) or both — and they compose with FSDP. The course's demo run stays well below this threshold.

## 3. Activation checkpointing (gradient checkpointing)

The other half of the memory budget is **activations**. Every forward op produces a tensor that backward needs. By default PyTorch saves them all — that's the term that scales as $B \cdot L \cdot d \cdot N_{\text{layers}}$.

**Activation checkpointing**: instead of saving every layer's activations, you save only the *inputs* to each layer (or group of layers), and *recompute* the rest during backward. The deal:

- **Memory**: activation memory drops from $O(N_{\text{layers}})$ to $O(\sqrt{N_{\text{layers}}})$ if you checkpoint every $\sqrt{N}$ layers, or to $O(1)$ if you checkpoint every layer.
- **Compute**: each checkpointed segment is computed twice — once in forward (not saved), once during backward (saved this time). **~33% extra FLOPs in the typical "checkpoint every transformer layer" recipe.**

The 33% comes from: backward already does roughly 2× the FLOPs of forward (backward is *compute the gradient w.r.t. each input*, which is ~2 matmuls per forward matmul). Adding a forward recomputation makes it backward = 1 (recompute) + 2 (gradient) = 3, vs the original 2 — a 50% increase on the backward, ~33% on the total forward+backward.

In PyTorch, activation checkpointing comes via `torch.utils.checkpoint.checkpoint_wrapper` or the higher-level `apply_activation_checkpointing` utility:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

def is_decoder_layer(m):
    # Qwen3, Llama, Mistral all have this class name shape
    return type(m).__name__.endswith("DecoderLayer")

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
    check_fn=is_decoder_layer,
)
```

This is what [`efficiency.py`](efficiency.py)'s `apply_activation_checkpointing` does.

**`CheckpointImpl.NO_REENTRANT`** is what you want for FSDP and Torch ≥ 2.1. The reentrant version has known issues with certain backward graph shapes and is being deprecated.

### When to turn it on

| Situation | Activation checkpointing? |
|---|---|
| OOM at any reasonable batch size | **Yes** — first thing to try |
| Comfortable HBM headroom, want max throughput | No — pay the 33% only if you must |
| Long sequence length (8k+, 32k+) | Almost always yes — activation memory dominates |
| Very small model (< 100M) | No — params dominate, checkpointing buys little |

The course's pretraining demo (Module 11) uses checkpointing at $L=2048$ because it lets us run the demo on a single A100 instead of needing multi-GPU.

### Selective checkpointing (the 2025 refinement)

A modern refinement: don't checkpoint every layer — checkpoint *only the FFN sub-block* (which is FLOP-cheap to recompute), and keep attention activations live (attention recomputation is expensive due to Flash Attention's own memory contract). This recovers most of the throughput while keeping most of the memory savings. PyTorch 2.5+ supports this via `selective_checkpoint_context_fn`; we don't ship it but the docstring in `efficiency.py` points at it.

## 4. The FSDP2 mixed-precision policy, properly

Module 08 wired `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` without explaining the full surface. Three dtypes matter:

```python
MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,   # cast params to this for compute
    reduce_dtype=torch.float32,   # reduce-scatter grads in this
    output_dtype=None,            # leave outputs in compute dtype (default)
)
```

- **`param_dtype`**: what params look like during forward/backward. BF16 is standard.
- **`reduce_dtype`**: what gradients are reduce-scattered in. **FP32 is the safe choice** — it avoids accumulating BF16 rounding error across thousands of gradient reductions over a long run. The cost is ~2× the gradient-comm bandwidth vs BF16 reductions. At 100B-token scale the FP32 reductions are worth it.
- **`output_dtype`**: if you want activations cast back to a different dtype after each FSDP block (e.g., for FP8 ops in adjacent layers). Usually leave at `None`.

**The asymmetry matters**: params live in BF16 (cheap to all-gather), gradients reduce in FP32 (safe to accumulate). The two extra bytes per param are not on every rank — they're transient during the reduce-scatter.

For **FP8 training** (Hopper+ only): FSDP2's policy still uses BF16 at the param level; FP8 happens *inside* the matmuls via `torch.float8_*` or NVIDIA's `transformer-engine`. The framework structure doesn't change — only the linear-layer implementation does. Module 11 has a note on enabling FP8 if you're on H100.

## 5. Beyond data parallelism — TP and PP, conceptually

Data parallelism (FSDP) splits *data* across GPUs; each GPU sees a different micro-batch. There are two other axes:

### Tensor parallelism (TP)

Split a single matrix multiplication across multiple GPUs. The Megatron-LM paper (Shoeybi et al., 2019) introduced the canonical pattern:

- **Column parallelism**: split the output dimension. `Linear(d_in, d_out)` becomes a `Linear(d_in, d_out/TP)` on each TP rank, then all-gather the outputs.
- **Row parallelism**: split the input dimension. `Linear(d_in, d_out)` becomes `Linear(d_in/TP, d_out)`, then all-reduce the outputs.

In a transformer, the standard recipe is:
- QKV projections: column-parallel
- Attention output projection: row-parallel
- FFN up-projection: column-parallel
- FFN down-projection: row-parallel

This puts one all-reduce per attention block and one per FFN block — **two per layer per forward pass**.

**When you need TP**: when even FSDP-3 can't fit your model on a single GPU's HBM. That's roughly 70B+ on A100 or 200B+ on H100. Below that threshold, TP usually slows you down — the all-reduces aren't free.

**The PyTorch interface**: `torch.distributed.tensor.parallel` provides `parallelize_module` and `ColwiseParallel` / `RowwiseParallel` styles. The model definition stays exactly the same — you decorate which layers get which style.

```python
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

parallelize_module(
    model,
    tp_mesh,
    {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj":   ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    },
)
```

We don't ship a TP example because the course's demo doesn't need it. But the lines above are roughly all there is to it once you have a `DeviceMesh`.

### Pipeline parallelism (PP)

Split the *layers* across GPUs. GPU 0 has layers 0–7, GPU 1 has layers 8–15, etc. Activations flow forward through the pipeline, gradients flow backward. To avoid GPUs sitting idle most of the time, you split each batch into many "micro-batches" and pipeline them — the GPipe / 1F1B schedules.

**When you need PP**: at very large scale (100B+ params) where TP within a node is saturated and you've run out of within-node communication bandwidth. PP lets you scale across nodes with much less inter-node comm than FSDP, because only one set of activations crosses the node boundary at the pipeline stage transitions.

**The downside**: bubble time (idle GPU at the start/end of each pipeline) and the complexity of the schedule. Most teams reach for PP only when forced to.

### The 3D parallelism picture

At frontier scale, all three axes compose: **TP within a node × PP across nodes × FSDP across replicas**. A 405B Llama 3 training run is something like TP=8 × PP=16 × FSDP=R where R is however many replicas your cluster has. The DeviceMesh API in PyTorch 2.4+ makes this declarative:

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", mesh_shape=(num_replicas, pp_size, tp_size),
                       mesh_dim_names=("dp", "pp", "tp"))

# Then fully_shard uses mesh["dp"], parallelize_module uses mesh["tp"], ...
```

You're unlikely to write this code in your first six months. But the abstractions to look for now are: **mesh, dim names, dimensions of parallelism**. Coming back to this section after Module 11 will land differently.

## 6. DeepSpeed — the alternative ecosystem

You will read papers and codebases that use **DeepSpeed** (Microsoft Research) instead of FSDP. They overlap heavily:

| Concept | FSDP2 (PyTorch native) | DeepSpeed |
|---|---|---|
| ZeRO-1 | (FSDP1 only — `SHARD_GRAD_OP`) | `stage: 1` |
| ZeRO-2 | `reshard_after_forward=False` | `stage: 2` |
| ZeRO-3 | `reshard_after_forward=True` (default) | `stage: 3` |
| Mixed precision | `MixedPrecisionPolicy` | `bf16: {enabled: true}` |
| Grad clipping | `clip_grad_norm_` | `gradient_clipping: 1.0` |
| Activation checkpointing | `apply_activation_checkpointing` | `activation_checkpointing.partition_activations` |
| Optimizer offload to CPU | (not in FSDP2 directly) | `offload_optimizer: {device: cpu}` |
| **NVMe offload** | (no) | **`offload_optimizer: {device: nvme}`** (ZeRO-Infinity) |

DeepSpeed has two things FSDP doesn't:

1. **CPU and NVMe offload** — when even ZeRO-3 doesn't fit, you can push optimizer state to host RAM or SSD. This is how the original GPT-NeoX-20B ran on relatively small clusters. The throughput cost is enormous, but the model fits.
2. **A monolithic config file** — DeepSpeed's `ds_config.json` covers training, evaluation, generation, profiling. PyTorch composes these from separate APIs.

**Why we use FSDP2 in this course**:
- PyTorch-native. No extra dependency. Same team as the compiler / DTensor / TP stack.
- Composes naturally with TP and PP via DeviceMesh.
- The mixed-precision and checkpointing APIs are cleaner.
- The future of distributed PyTorch is centered here.

**Why you should still know DeepSpeed exists**:
- The HuggingFace ecosystem defaults to DeepSpeed for many integrations (`accelerate`, `transformers.Trainer`).
- Older research codebases (anything pre-2024) are usually DeepSpeed.
- ZeRO-Infinity / NVMe offload has no FSDP2 equivalent.

Treat DeepSpeed like the other dialect of the same language. Translating a DeepSpeed config to an FSDP2 setup is mostly a name-mapping exercise.

## 7. Multi-token prediction (MTP)

A small architectural change with an outsized effect. DeepSeek-V3 introduced **multi-token prediction** as part of their pretraining objective.

**The standard objective**: at every position $t$ in a sequence, predict token $t+1$ given tokens $\leq t$. Cross-entropy on one prediction per position.

**MTP**: at every position $t$, predict tokens $t+1$, $t+2$, $t+3$ — via a shared trunk and small per-offset heads.

The architectural shape:

```
hidden_t (output of last decoder layer)
   │
   ├── head_1(hidden_t) → predict token at position t+1   (main loss)
   ├── head_2(hidden_t) → predict token at position t+2   (auxiliary loss, weight λ₂)
   └── head_3(hidden_t) → predict token at position t+3   (auxiliary loss, weight λ₃)
```

Each auxiliary head is a small transformer block (one layer in V3) plus an unembedding projection. Total parameter overhead: ~1–2% of model size.

**Two payoffs**:

1. **Denser gradient signal during pretraining**. Every position now contributes 3 prediction losses instead of 1. The model learns longer-range dependencies because head_3 explicitly demands predicting 3 tokens ahead from the same hidden state. DeepSeek reports ~0.3% perplexity improvement and noticeably better downstream benchmark scores at iso-compute.
2. **Speculative decoding for free at inference**. The MTP heads can serve as a built-in draft model: predict next-2 and next-3, verify with the main head. **~1.8× inference speedup** in V3's setup, with no separate draft model needed.

**The tradeoffs**:

- Extra forward/backward FLOPs through the auxiliary heads — ~5–10% per training step.
- More implementation complexity in the loss computation.
- λ tuning (V3 uses λ₂ = λ₃ ≈ 0.3 — the auxiliary losses are downweighted, not equal-weighted).

The 2026 read: MTP is a strong default for new pretraining runs above ~1B parameters where the inference-speedup payoff is real. Below that, the implementation cost outweighs the gain. The course's demo (~150M) doesn't ship MTP but Module 11's README points at how to add it as a stretch goal.

If you want to see the math and code, see DeepSeek-V3's technical report, Section 4.3.

## 8. Chinchilla — the budgeting tool

You have $C$ FLOPs of compute. How do you split it between (model size $N$, training tokens $D$)?

The Chinchilla paper (Hoffmann et al., 2022) ran a careful sweep and derived a scaling law:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

where $L$ is loss, and $A, B, \alpha, \beta, E$ are fitted constants. The optimal allocation at a fixed compute budget $C$:

$$N^* \cdot D^* \propto C^{0.5}, \quad \frac{D^*}{N^*} \approx 20$$

**The Chinchilla rule in one line**: at compute-optimal, you train $\approx 20$ tokens per parameter.

| Compute budget (FLOPs) | Optimal params $N^*$ | Optimal tokens $D^*$ |
|---|---|---|
| $10^{19}$ (small academic) | ~400M | ~8B |
| $10^{21}$ (single-node A100, days) | ~3B | ~60B |
| $10^{23}$ (Chinchilla original) | ~70B | ~1.4T |
| $10^{25}$ (Llama-3-405B class) | ~400B | ~8T |

The constants for the 20:1 ratio depend on architecture / data quality — modern frontier work usually quotes 20:1 ± a factor of 2.

### The 6ND approximation

The training-FLOP cost of one forward + backward pass through a transformer is approximately:

$$\text{FLOPs per token} \approx 6N$$

(2N for forward, 4N for backward; the 6 absorbs both into one number, ignoring attention's $O(L^2)$ term which is ~10% at typical seq lengths.)

So **total training FLOPs ≈ $6 \cdot N \cdot D$**. This is what `efficiency.py`'s `chinchilla_optimal(flops)` inverts: given $C$ FLOPs and the 20:1 target, return the $(N, D)$ that uses them up at compute-optimal.

```python
from efficiency import chinchilla_optimal, training_flops

n_params, n_tokens = chinchilla_optimal(compute_flops=1e22)
# (~9B params, ~180B tokens)

flops = training_flops(n_params=9e9, n_tokens=180e9)
# ~9.7e21
```

### Where Chinchilla doesn't apply

The 20:1 ratio is **compute-optimal for the training run alone**. It says nothing about inference cost. If you're going to deploy a model that serves billions of tokens per day, a smaller model trained longer is *much cheaper to serve*, even if the training was suboptimal.

**Llama 3 trained at ~200:1** (15T tokens on an 8B model). This is 10× Chinchilla. The training was wasteful, the model is excellent — because Meta cares about *inference cost over the model's lifetime*, not training compute.

**The rule for 2026**:

- Research run, throwaway artifact, optimize training compute: **stick to ~20:1**.
- Production model, will serve a lot: **train more tokens than Chinchilla optimal**. The model will be smaller and cheaper at inference; you "wasted" training FLOPs but saved on every inference forever.
- Data-constrained (you can't get more high-quality tokens): just train as many epochs as quality allows; don't pretend Chinchilla solves you.

**For the course's pretraining demo (Module 11)**: ~150M params, ~3B tokens of FineWeb-Edu. That's 20:1. Compute-optimal. The point is to show the discipline, not to maximize downstream eval — you'd train 5–10× longer for a production model.

## 9. What we ship

[`efficiency.py`](efficiency.py) provides three utilities. None of them rewrite anything in Module 08; they're additive helpers Module 11 will compose.

| Function | What it does |
|---|---|
| `apply_activation_checkpointing(model)` | Wraps every Qwen3 decoder layer with `checkpoint_wrapper(NO_REENTRANT)`. One call, model in place. |
| `memory_breakdown(n_params, dtype, optimizer, checkpointing, ...)` | Returns a dict-of-bytes breakdown for a given configuration. The dataset behind every memory plot in this module. |
| `chinchilla_optimal(compute_flops, tokens_per_param=20)` | Given $C$ FLOPs and a tokens-per-param ratio, returns the optimal $(N, D)$. Inverts the 6ND approximation. |
| `training_flops(n_params, n_tokens)` | The forward `6ND` direction — given $(N, D)$, what's the FLOP cost? |

Plus a small `__main__` smoke test.

## 10. Wiring it into the training loop

Two changes to Module 08's `train.py` to use this module's pieces:

```python
from efficiency import apply_activation_checkpointing   # new import

# After build_model(...), before apply_fsdp(...)
if cfg.training.activation_checkpointing:
    apply_activation_checkpointing(model)              # new line — see efficiency.py
```

`apply_activation_checkpointing` must be called **before FSDP wrapping**. Checkpointing wraps the decoder layer with a `checkpoint_wrapper`, and FSDP shards the wrapped module. The reverse order silently breaks FSDP's per-layer sharding logic.

`TrainingConfig` gets one new field: `activation_checkpointing: bool = False`. Module 11's demo config flips it on.

## 11. Reading list

- **ZeRO** (the conceptual root of FSDP): Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2019). Section 4 (the math of the three stages) is the part that pays off most.
- **FSDP2 deep dive**: PyTorch blog "Introducing FSDP2" (2024) and the `torch.distributed.fsdp` API docs. Read after this module if you want the internals.
- **Megatron-LM (tensor parallelism)**: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019). The column/row parallelism diagrams are the canonical reference.
- **Gradient checkpointing**: Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016). The original $O(\sqrt{N})$ memory result.
- **Chinchilla**: Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022). Read at least Section 3 — the empirical fit.
- **Multi-token prediction**: DeepSeek-V3 Technical Report (2024), Section 4.3. The clearest description of MTP in a frontier model.
- **The Llama 3 anti-Chinchilla decision**: Meta AI, "The Llama 3 Herd of Models" (2024), Section 3.4 — explicit discussion of training past Chinchilla optimal for inference efficiency.

## Next

- **[Module 11 — Pretraining in Practice](../11-pretraining-in-practice/)** composes everything: Module 08's loop + Module 09's schedule + this module's checkpointing + the FineWeb-Edu pipeline. The actual demo run, end-to-end.
