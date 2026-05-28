"""Model builder for Module 13 — Continual Pretraining.

Copied from Module 15. The key difference from Module 11's `model.py`:
pretraining *constructs* a randomly-initialized model from geometry knobs;
continual pretraining *loads* a finished base checkpoint and keeps training it.
So `build_model` here is a thin wrapper over `AutoModelForCausalLM.from_pretrained`
— the base whose knowledge you're about to extend (default `Qwen3-0.6B-Base`).

**The framework's contract is the same as Module 11:**

    build_model(cfg) -> nn.Module
    whose forward accepts `input_ids=..., attention_mask=...` and returns an
    object with `.logits` of shape `[B, S, V]`. Loss is computed in
    `loop.py` (assistant-masked cross-entropy), not here.

Two SFT-specific details baked in:

1. **Loaded in FP32.** We force `dtype=torch.float32` so the parameters are
   the FP32 *master weights* the optimizer updates. FSDP's
   `MixedPrecisionPolicy` (see `fsdp_setup.apply_fsdp`) then casts to BF16
   for the forward/backward and reduces grads in FP32. This is exactly the
   mixed-precision recipe from Module 10, and it's what makes the README §4
   memory table (6.8 GB FP32 master + 3.4 GB BF16 compute) accurate. If we
   instead loaded in BF16, there would be no FP32 master and the run would
   be less numerically stable — wrong for a full-FT job.

2. **`use_cache=False`.** The KV cache is an inference optimization; during
   training it wastes memory and, more importantly, HF emits a warning and
   silently disables activation checkpointing if `use_cache=True`. `eval.py`
   flips it back on for generation.

`cfg.name` is either a HuggingFace Hub ID (`"Qwen/Qwen3-1.7B-Base"`) or a
local directory in HF format — e.g. the output of Module 11's `export_hf.py`,
which is how you SFT your own pretrained 150M model (see `configs/sft_demo.yaml`).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import ModelConfig


def build_model(cfg: ModelConfig, fused_ce: bool = False) -> nn.Module:
    """Load a pretrained HF causal LM for full fine-tuning.

    Args:
        cfg: which model to load (`name`) and the max sequence length we'll
            train at (`max_seq`, used only to set `max_position_embeddings`
            sanity, not to resize anything).
        fused_ce: if True, patches the underlying HF causal-LM class with
            Liger Kernel's fused linear + cross-entropy *before* loading
            weights. The patched model exposes the same forward(input_ids,
            attention_mask, labels=) API, but when `labels` are passed the
            LM head matmul is fused with CE in a chunked Triton kernel — the
            full `[B, S, V]` logits tensor is never materialized. Mirrors
            Module 11's `use_fused_ce`. See `loop.py` for how the loop
            drives the fused path.

    Returns:
        An `AutoModelForCausalLM` in FP32 with `use_cache=False`, every
        parameter `requires_grad=True` (full fine-tuning). FSDP-compatible:
        its decoder layers live in `model.model.layers` (Qwen3 / Llama /
        Mistral / Gemma all satisfy this).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if fused_ce:
        # Patch must happen BEFORE `from_pretrained(...)` instantiates the
        # model — Liger patches the class, not the instance. Unlike Module 11
        # we don't know the arch a priori (`cfg.name` can be Qwen3/Llama/etc.)
        # so we peek at the HF config to dispatch on `model_type`. We only flip
        # `fused_linear_cross_entropy` and leave the other Liger patches
        # (RoPE, RMSNorm, SwiGLU) off to minimize the blast radius — those
        # change numerics subtly and aren't what's saving you the memory.
        try:
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel
        except ImportError as e:
            raise ImportError(
                "training.use_fused_ce=true requires `pip install liger-kernel`."
            ) from e
        model_type = AutoConfig.from_pretrained(cfg.name).model_type
        _apply_liger_kernel(
            model_type=model_type,
            rope=False,
            rms_norm=False,
            swiglu=False,
            cross_entropy=False,                # we want the *fused* variant, not the standalone CE patch
            fused_linear_cross_entropy=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        dtype=torch.float32,      # FP32 master weights; FSDP casts to BF16 for compute
    )
    model.config.use_cache = False

    # Full fine-tuning: make sure nothing arrived frozen (some hub checkpoints
    # ship with requires_grad already toggled on parts of the graph).
    for p in model.parameters():
        p.requires_grad_(True)

    return model


def count_params(model: nn.Module) -> dict[str, int]:
    """Param-count breakdown by group. Useful for the startup banner.

    Carried over from Module 11 — the name heuristics match any HF causal LM
    with the standard `embed_tokens` / `self_attn` / `mlp` / `norm` / `lm_head`
    submodule names. Weight-tied params are counted once.
    """
    counts = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "head": 0, "total": 0}
    seen_ids: set[int] = set()
    for name, p in model.named_parameters():
        if id(p) in seen_ids:           # weight tying — count once
            continue
        seen_ids.add(id(p))
        n = p.numel()
        counts["total"] += n
        if "embed_tokens" in name:
            counts["embedding"] += n
        elif "lm_head" in name:
            counts["head"] += n
        elif "self_attn" in name:
            counts["attention"] += n
        elif "mlp" in name:
            counts["mlp"] += n
        elif "norm" in name:
            counts["norm"] += n
    return counts


if __name__ == "__main__":
    # Smoke test without a network round-trip: build a tiny Qwen3 from scratch
    # (this is what `from_pretrained` returns the *shape* of, minus the weights)
    # and exercise count_params. The real path downloads `cfg.name`.
    from transformers import Qwen3Config, Qwen3ForCausalLM

    m = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=2048, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
        max_position_embeddings=128, tie_word_embeddings=True, use_cache=False,
    ))
    counts = count_params(m)
    total = counts["total"]
    print("count_params on a tiny Qwen3:")
    for k, v in counts.items():
        print(f"  {k:10s} {v/1e6:>7.3f}M  ({v/total*100:>5.1f}%)")
    print("\nTo load a real base model:")
    print("  build_model(ModelConfig(name='Qwen/Qwen3-1.7B-Base'))")
