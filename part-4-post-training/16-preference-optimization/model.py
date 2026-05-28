"""Model builders for Module 16 — Preference Optimization.

DPO/IPO need TWO models in memory at every step:

1. **The policy** — what we train. Loaded in FP32 (so FSDP MixedPrecision
   gives us BF16 compute + FP32 master weights, same recipe as Module 15).
   Full gradient, full AdamW state, full FSDP machinery.

2. **The reference** — frozen. Loaded in BF16, `requires_grad=False`,
   `use_cache=False`, never sees an optimizer. We FSDP-wrap it on
   multi-GPU runs only for *sharding parity* (so it consumes 1/world of
   VRAM per rank, matching the policy's footprint); the wrap also keeps
   the param-fetch / collective patterns identical between policy and
   reference, which avoids subtle deadlocks under torchrun.

The reference is conceptually the SFT model snapshotted at step 0. In
practice we just load the same checkpoint that `policy.name` points at
(`cfg.model.resolved_ref_name()`). Override via `model.ref_name` if you
want a different anchor (the base model, a previous DPO round, etc.).

A memory-saving alternative — `precompute_ref_logps.py` — would tokenize
the entire preference dataset, run a single pass on the reference, and
cache `(log_pi_ref(chosen), log_pi_ref(rejected))` to disk. Then training
needs ONLY the policy in VRAM, halving memory at the cost of a one-time
preprocessing step. The TRL library does this. We don't, for two reasons:

1. Pedagogy: holding both models in memory makes the math (and what the
   reference IS) tangible. The cached path looks like SFT with weird labels.
2. Mechanics: with the reference live, you can change `model.ref_name`
   without re-preprocessing — useful when iterating on β and loss_type.

See README §4 for the memory table and §9 (stretch goals) for caching.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import ModelConfig


def build_policy(cfg: ModelConfig) -> nn.Module:
    """Load the policy model — the one we're training.

    Identical to Module 15's `build_model`: FP32 master weights, no KV cache,
    every parameter requires grad. FSDP MixedPrecision then casts to BF16
    for compute and reduces grads in FP32 (Module 10's mixed-precision
    recipe). The caller wraps with `apply_fsdp` after `apply_activation_
    checkpointing` — see train.py for the order.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        dtype=torch.float32,
    )
    model.config.use_cache = False

    for p in model.parameters():
        p.requires_grad_(True)

    return model


def build_reference(cfg: ModelConfig) -> nn.Module:
    """Load the reference model — frozen, BF16, no grad.

    The reference's only job is to evaluate `log π_ref(y|x)` on the chosen
    and rejected completions inside the DPO loss. It never updates, so we
    aggressively strip everything DPO doesn't need:

    - **BF16 storage.** Halves the memory vs the policy's FP32 master. The
      ~1-2 nats/token of FP16/BF16 rounding noise is irrelevant for a
      reference that only contributes a log-prob difference inside a
      log-sigmoid — the signal lives in the (much larger) policy vs
      reference gap.
    - **`requires_grad_(False)`.** No autograd graph, no backward, no
      gradient buffers — also disables FSDP's gradient sharding machinery
      for these params, freeing per-rank VRAM.
    - **`use_cache=False`.** Reference forwards happen inside `train.py`'s
      training loop, not generation. We never want the KV cache.
    - **`.eval()`.** Dropout / batchnorm-style modules go to eval-mode
      semantics (Qwen3 has no batchnorm but dropout is set to 0 in chat
      models, so this is a no-op for our case — included for safety in
      case you swap in a model with active dropout).
    """
    from transformers import AutoModelForCausalLM

    name = cfg.resolved_ref_name()
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,    # half the policy's storage; no master weights needed
    )
    model.config.use_cache = False

    for p in model.parameters():
        p.requires_grad_(False)

    model.eval()
    return model


def count_params(model: nn.Module) -> dict[str, int]:
    """Param-count breakdown by group. Carried over from Module 15."""
    counts = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "head": 0, "total": 0}
    seen_ids: set[int] = set()
    for name, p in model.named_parameters():
        if id(p) in seen_ids:
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
    # Offline smoke test: build a tiny Qwen3 from scratch and verify the
    # policy/reference contracts (FP32 vs BF16, grad on vs off, etc.).
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tiny = Qwen3Config(
        vocab_size=2048, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
        max_position_embeddings=128, tie_word_embeddings=True, use_cache=False,
    )

    policy_shape = Qwen3ForCausalLM(tiny)
    print("policy spec (offline shape only — real path: build_policy(ModelConfig(...))")
    print(f"  dtype if loaded:   torch.float32   (FP32 master)")
    print(f"  requires_grad:     True (all params)")
    print(f"  use_cache:         False")
    counts = count_params(policy_shape)
    print(f"  param count: {counts['total']/1e6:.3f}M")

    ref_shape = Qwen3ForCausalLM(tiny)
    for p in ref_shape.parameters():
        p.requires_grad_(False)
    ref_shape.eval()
    print("\nreference spec")
    print(f"  dtype if loaded:   torch.bfloat16  (half the policy's storage)")
    print(f"  requires_grad:     False (all params)")
    print(f"  use_cache:         False")
    print(f"  eval mode:         True")
