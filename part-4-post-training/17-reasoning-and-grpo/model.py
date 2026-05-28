"""Model builders for Module 17 — Reasoning and GRPO.

Identical pattern to Module 16: build_policy + build_reference. Two models,
one trainable (FP32 master, BF16 compute via FSDP MixedPrecision) and one
frozen (BF16, no grad, eval mode).

What's new vs Module 16 is the *use*: the policy is asked to GENERATE in
the rollout phase. That puts a couple of constraints on the loader that
DPO didn't care about:

- During training (forward + backward for the loss), `config.use_cache=False`
  is required for FSDP2 — the KV cache breaks the per-layer all-gather
  pattern.
- During generation (the rollout), `config.use_cache=True` is mandatory
  for any reasonable throughput — without it, every generated token is an
  O(N²) re-forward over the whole prefix.

The two modes are flipped *around* `model.generate()` in rollout.py via the
`generation_mode` context manager defined here. The policy alternates many
times per RL step (generate ↔ train), so we need this to be cheap and
reversible.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from config import ModelConfig


def build_policy(cfg: ModelConfig) -> nn.Module:
    """Load the policy model — the one we're training.

    Identical to Module 16's `build_policy`: FP32 master weights, no KV cache,
    every parameter requires grad. FSDP MixedPrecision then casts to BF16
    for compute and reduces grads in FP32. The caller wraps with `apply_fsdp`
    after `apply_activation_checkpointing` — see train.py for the order.
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

    Same recipe as Module 16. The reference contributes the per-token KL
    penalty inside the GRPO loss; it's never updated and is held in BF16
    to halve its memory vs the policy's FP32 master.
    """
    from transformers import AutoModelForCausalLM

    name = cfg.resolved_ref_name()
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    for p in model.parameters():
        p.requires_grad_(False)

    model.eval()
    return model


@contextlib.contextmanager
def generation_mode(model: nn.Module):
    """Flip the policy into generation-friendly mode for the duration of a
    `with generation_mode(policy): ...` block.

    Enables KV cache and switches the model to `.eval()` (so dropout stays
    off — Qwen3 has none but the contract is safe for other families).
    Restores the original `use_cache` flag and training mode on exit.

    Why this matters: FSDP2 + KV cache during training would break the per-
    layer all-gather pattern, but generation can't proceed without the cache
    (one O(N²) pass per token is dead on arrival). We need to toggle, and
    we need to toggle correctly — leaving `use_cache=True` after generation
    will silently slow down the gradient forward to a crawl on long seqs.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        # Bare nn.Module fallback for unit tests.
        yield model
        return

    was_training = model.training
    prev_use_cache = getattr(cfg, "use_cache", False)
    cfg.use_cache = True
    model.eval()
    try:
        yield model
    finally:
        cfg.use_cache = prev_use_cache
        if was_training:
            model.train()
        else:
            model.eval()


def count_params(model: nn.Module) -> dict[str, int]:
    """Param-count breakdown by group. Carried over from Module 16."""
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
    print(f"  use_cache:         False (training)")
    counts = count_params(policy_shape)
    print(f"  param count: {counts['total']/1e6:.3f}M")

    # Demonstrate the generation_mode toggle.
    policy_shape.train()
    with generation_mode(policy_shape):
        print("\ninside generation_mode:")
        print(f"  training = {policy_shape.training}")
        print(f"  use_cache = {policy_shape.config.use_cache}")
    print("\noutside generation_mode:")
    print(f"  training = {policy_shape.training}")
    print(f"  use_cache = {policy_shape.config.use_cache}")

    ref_shape = Qwen3ForCausalLM(tiny)
    for p in ref_shape.parameters():
        p.requires_grad_(False)
    ref_shape.eval()
    print("\nreference spec")
    print(f"  dtype if loaded:   torch.bfloat16  (half the policy's storage)")
    print(f"  requires_grad:     False (all params)")
    print(f"  use_cache:         False")
    print(f"  eval mode:         True")
