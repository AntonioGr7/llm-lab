"""Model builders for Module 19 — Distillation.

Three distillation modes (see config.py docstring) have different
model-loading requirements:

  - **offline**:   one model (the student). The "teacher" is implicit
                   in the pre-generated dataset; we never load it. (Not
                   shipped as a runnable here — README §2 routes to
                   Module 15's SFT code.)
  - **on_policy**: TWO models. Student is FP32-master, trainable, FSDP.
                   Teacher is a *different* (usually larger) BF16-frozen
                   model. Same dual-model pattern as Modules 17/17.
  - **sdft**:      ONE model. The student is FP32-master, trainable. The
                   "teacher" is the SAME OBJECT, queried in `torch.no_grad`
                   with demonstrations prepended in-context. No second
                   model in memory — that's the whole memory advantage
                   of SDFT.

`build_student` is shared across all three modes. `build_teacher` is
called only for `on_policy`; for `sdft`, `train.py` reuses the student.

The `generation_mode` context manager (also in Module 18) flips KV
cache + eval mode for the duration of a `.generate()` call. SDFT calls
this on the STUDENT during teacher rollouts — the same model goes into
"teacher mode" by virtue of the demonstrations in its input, not by
swapping weights.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from config import ModelConfig


def build_student(cfg: ModelConfig) -> nn.Module:
    """Load the student model — the one we're training.

    Identical recipe to Modules 17/17 build_policy: FP32 master, no KV
    cache, every parameter requires grad. FSDP MixedPrecision then casts
    to BF16 for compute and reduces grads in FP32.
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


def build_teacher(cfg: ModelConfig) -> nn.Module:
    """Load the teacher — frozen, BF16, no grad. (on_policy mode only.)

    Same recipe as Module 17's build_reference / Module 18's
    build_reference. For sdft mode, do NOT call this — reuse the student
    via the `no_grad` + demonstration-conditioned-input pattern.
    """
    from transformers import AutoModelForCausalLM

    name = cfg.resolved_teacher_name()
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
    """Flip `model` into generation-friendly mode (KV cache + eval) for the
    duration of the block. Restores training mode + use_cache on exit.

    Same helper as Module 18's. Critically reused here for SDFT: the
    teacher rollout calls `student.generate(...)` after putting it into
    generation_mode; the student then comes BACK into training mode for
    the gradient forward. Both happen in one training step.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
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
    """Param-count breakdown by group. Carried over from Module 18."""
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
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tiny = Qwen3Config(
        vocab_size=2048, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
        max_position_embeddings=128, tie_word_embeddings=True, use_cache=False,
    )

    student_shape = Qwen3ForCausalLM(tiny)
    print("student spec (offline shape only):")
    print(f"  dtype if loaded:   torch.float32   (FP32 master)")
    print(f"  requires_grad:     True (all params)")
    print(f"  use_cache:         False (training)")
    counts = count_params(student_shape)
    print(f"  param count: {counts['total']/1e6:.3f}M")

    student_shape.train()
    with generation_mode(student_shape):
        print("\ninside generation_mode:")
        print(f"  training = {student_shape.training}")
        print(f"  use_cache = {student_shape.config.use_cache}")
    print("\noutside generation_mode:")
    print(f"  training = {student_shape.training}")
    print(f"  use_cache = {student_shape.config.use_cache}")
