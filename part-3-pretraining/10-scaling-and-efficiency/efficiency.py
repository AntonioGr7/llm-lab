"""Scaling and efficiency utilities for Part 3's pretraining framework.

Three additive helpers — none of them rewrite anything from Module 08:

- `apply_activation_checkpointing(model)`: wraps each Qwen3 decoder layer
  with a NO_REENTRANT checkpoint wrapper. Call BEFORE FSDP wrapping.
- `memory_breakdown(...)`: returns a dict of bytes-per-component for a
  given (model size, dtype, optimizer, sharding, checkpointing) config.
  The dataset behind every memory plot in Module 10's notebook.
- `chinchilla_optimal(flops)` and `training_flops(n, d)`: the 6ND
  approximation and its inversion. Compute-optimal (N, D) given a FLOP
  budget; FLOP cost given (N, D).

All three are pure-Python / pure-math except `apply_activation_checkpointing`,
which calls into `torch.distributed.algorithms._checkpoint`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


# ----------------------------------------------------------------------
# 1. Activation checkpointing
# ----------------------------------------------------------------------

def apply_activation_checkpointing(model, layer_class_suffix: str = "DecoderLayer"):
    """Wrap every transformer decoder layer with activation checkpointing.

    Trades ~33% extra FLOPs for ~all of the activation memory. Use when
    activations dominate your HBM budget (long sequence, deep model) or
    when you OOM at any reasonable batch size.

    Implementation detail: uses `CheckpointImpl.NO_REENTRANT`, which is
    the recipe that composes with FSDP for PyTorch ≥ 2.1. The older
    reentrant impl has known issues with certain backward graph shapes
    and is being deprecated.

    A 2025 refinement worth knowing: instead of checkpointing the whole
    decoder layer, checkpoint only the FFN sub-block (cheap to recompute)
    and let attention activations live (Flash Attention has its own
    memory contract that recomputation breaks). PyTorch 2.5+ supports
    this via `selective_checkpoint_context_fn`. We don't ship it — full-
    layer checkpointing is the safer default — but it's documented here
    so you know what to reach for if you need to recover the throughput.

    Args:
        model: an HF-style causal LM (Qwen3, Llama, Mistral — anything
            whose decoder layers are named `*DecoderLayer`).
        layer_class_suffix: pattern to match decoder layer class names.
            Default works for all three families above.

    Returns:
        The same model object, with each matching layer now wrapped in
        place. **Call this BEFORE `apply_fsdp(...)` from Module 08** —
        FSDP must shard the checkpoint-wrapped layer, not the other way.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing as _apply,
        checkpoint_wrapper,
        CheckpointImpl,
    )

    def check_fn(m):
        return type(m).__name__.endswith(layer_class_suffix)

    def wrap_fn(m):
        return checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    _apply(model, checkpoint_wrapper_fn=wrap_fn, check_fn=check_fn)
    return model


# ----------------------------------------------------------------------
# 2. Memory breakdown
# ----------------------------------------------------------------------

Dtype = Literal["fp32", "bf16", "fp16", "fp8"]


_BYTES_PER_DTYPE = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}


@dataclass
class MemoryBreakdown:
    """Per-rank bytes for every consumer in a training step.

    Use `.total` for the headline number; the individual fields are what
    the notebook plots as a stacked bar.
    """
    master_weights: float       # FP32 master copy that the optimizer updates
    compute_weights: float      # BF16/FP8 copy used for forward/backward
    gradients: float            # post-reduce gradients (possibly sharded)
    optimizer_state: float      # m, v for AdamW; small for Muon
    activations: float          # forward activations cached for backward

    @property
    def total(self) -> float:
        return (self.master_weights + self.compute_weights + self.gradients
                + self.optimizer_state + self.activations)

    def as_dict(self) -> dict[str, float]:
        return {
            "master_weights": self.master_weights,
            "compute_weights": self.compute_weights,
            "gradients": self.gradients,
            "optimizer_state": self.optimizer_state,
            "activations": self.activations,
            "total": self.total,
        }


def memory_breakdown(
    n_params: float,
    dtype: Dtype = "bf16",
    optimizer: Literal["adamw", "muon", "sgd"] = "adamw",
    zero_stage: Literal[0, 1, 2, 3] = 3,
    world_size: int = 1,
    activation_checkpointing: bool = False,
    n_layers: int = 32,
    batch_size: int = 1,
    seq_len: int = 2048,
    d_model: int = 4096,
    activation_bytes_per_slot: int = 20,
) -> MemoryBreakdown:
    """Estimate per-rank memory for a training step.

    The model is `n_params` parameters; the geometry args (`n_layers`,
    `batch_size`, `seq_len`, `d_model`) are only used for the activation
    estimate. If you only care about the param/grad/opt-state plot, you
    can pass any sensible defaults for the geometry.

    Returns bytes per *single rank* after ZeRO sharding. Pass `world_size=1`
    for the unsharded (single-GPU) baseline.

    The numbers are approximate — within ~10% of what nvidia-smi will
    show in practice, which is what you need for capacity planning.

    Mixed-precision recipe assumed:
      - Master weights live in FP32 (`adamw` only — `sgd` skips them).
      - Compute weights live in `dtype` (BF16 default).
      - Gradients reduce in FP32, but per-rank shards are `dtype`-sized
        (because they're cast back after reduce-scatter).
      - AdamW state: 2× FP32 vectors (m, v).
      - Muon state: 1× compute-dtype momentum buffer per 2D weight
        (~ half the weights), no v_hat. Modeled as ~1× param_bytes for
        the hidden subset, 2× FP32 for the AdamW-tail.

    For activation memory, the standard estimate:

        activation_bytes ≈ B · L · d_model · n_layers · c

    where `c` (in bytes per "slot") absorbs the dozen-or-so tensors per
    layer that backward needs (Q, K, V projections, attention probs,
    FFN intermediates, residuals). c=20 is a reasonable typical value
    for a Qwen3-shape; FlashAttention drops it ~30%.
    """
    param_bytes = _BYTES_PER_DTYPE[dtype]

    # Per-rank divisor: ZeRO stage k shards (some of) the buckets across world_size.
    div_opt = world_size if zero_stage >= 1 else 1
    div_grad = world_size if zero_stage >= 2 else 1
    div_params = world_size if zero_stage >= 3 else 1

    # Master weights — only for FP32-master mixed-precision optimizers.
    if optimizer == "sgd":
        master = 0.0
    else:
        master = 4.0 * n_params / div_params   # FP32 master copy

    compute = param_bytes * n_params / div_params

    # Gradients are dtype-sized after reduce-scatter (FSDP2 default reduces in
    # FP32 but the per-rank shard lives in dtype-sized memory).
    grads = param_bytes * n_params / div_grad

    # Optimizer state.
    if optimizer == "adamw":
        opt_state = 8.0 * n_params / div_opt   # 2× FP32 vectors (m, v)
    elif optimizer == "muon":
        # Approximation: half the params are 2D hidden weights → Muon momentum
        # in compute dtype (1×). Other half (embeddings, norms, biases) → AdamW
        # (8×). This is rough; real Muon configs vary.
        opt_state = (param_bytes * 0.5 + 8.0 * 0.5) * n_params / div_opt
    else:  # sgd
        opt_state = param_bytes * n_params / div_opt  # one momentum buffer

    # Activations. Two regimes: full save (default) vs activation checkpointing.
    # With per-layer checkpointing, you save only the per-layer *inputs*, not
    # the per-layer intermediates — memory drops from ~ N_layers · c to ~ c
    # for one layer's worth of recompute. Net: divide by n_layers.
    activations_full = (
        batch_size * seq_len * d_model * n_layers * activation_bytes_per_slot
    )
    if activation_checkpointing:
        # Memory drops to roughly the activations of a single layer's recompute.
        activations = activations_full / n_layers
    else:
        activations = activations_full

    return MemoryBreakdown(
        master_weights=master,
        compute_weights=compute,
        gradients=grads,
        optimizer_state=opt_state,
        activations=activations,
    )


# ----------------------------------------------------------------------
# 3. Chinchilla budgeting
# ----------------------------------------------------------------------

def training_flops(n_params: float, n_tokens: float) -> float:
    """The 6ND approximation: total training FLOPs ≈ 6 · N · D.

    2N for the forward pass, 4N for backward (gradient w.r.t. weights and
    activations). Ignores attention's O(L^2) term, which is ~10% of the
    total at typical sequence lengths and shrinks as you go wider.
    """
    return 6.0 * n_params * n_tokens


def chinchilla_optimal(
    compute_flops: float, tokens_per_param: float = 20.0,
) -> tuple[float, float]:
    """Given a FLOP budget, return Chinchilla-optimal (n_params, n_tokens).

    Inverts the 6ND approximation with the constraint D = `tokens_per_param` · N:

        6 · N · D = C
        D = R · N         (R = tokens_per_param)
        → N = sqrt(C / (6·R)),  D = R · N

    The 20:1 ratio is Chinchilla's empirical compute-optimal. Production
    models commonly use higher (Llama 3 ~200:1) when inference cost matters
    more than training compute.

    Returns:
        (n_params, n_tokens), both as floats.
    """
    r = tokens_per_param
    n = math.sqrt(compute_flops / (6.0 * r))
    d = r * n
    return n, d


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    GB = 1024**3

    print("=" * 60)
    print("memory_breakdown — 1B model, BF16 + AdamW")
    print("=" * 60)
    for ws, label in [(1, "single GPU, no sharding"),
                      (8, "8 GPUs, ZeRO-3"),
                      (64, "64 GPUs, ZeRO-3")]:
        zs = 0 if ws == 1 else 3
        b = memory_breakdown(
            n_params=1e9, dtype="bf16", optimizer="adamw",
            zero_stage=zs, world_size=ws,
            activation_checkpointing=False,
            n_layers=24, batch_size=1, seq_len=2048, d_model=2048,
        )
        print(f"\n  {label}:")
        for k, v in b.as_dict().items():
            print(f"    {k:18s} {v/GB:7.2f} GB")

    print("\n" + "=" * 60)
    print("activation checkpointing on/off — 1B model, batch=4, seq=4096")
    print("=" * 60)
    for ac, label in [(False, "no checkpointing"), (True, "with checkpointing")]:
        b = memory_breakdown(
            n_params=1e9, dtype="bf16", optimizer="adamw",
            zero_stage=3, world_size=8,
            activation_checkpointing=ac,
            n_layers=24, batch_size=4, seq_len=4096, d_model=2048,
        )
        print(f"\n  {label}:")
        print(f"    activations         {b.activations/GB:7.2f} GB")
        print(f"    total               {b.total/GB:7.2f} GB")

    print("\n" + "=" * 60)
    print("chinchilla_optimal — sweep compute budgets")
    print("=" * 60)
    for c in [1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25]:
        n, d = chinchilla_optimal(c)
        verify = training_flops(n, d)
        print(f"  C={c:.0e}  →  N≈{n/1e9:7.2f}B params, D≈{d/1e9:7.1f}B tokens "
              f"(verify 6ND={verify:.2e})")

    print("\n  Llama-3-8B trained at ~200:1 (anti-Chinchilla, inference-optimized):")
    n, d = chinchilla_optimal(1e23, tokens_per_param=200.0)
    print(f"    With ratio=200: at C=1e23, N≈{n/1e9:.2f}B, D≈{d/1e12:.2f}T")
