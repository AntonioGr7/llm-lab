"""The inner training loop helpers.

Two responsibilities, factored out of `train.py` so they're testable:

- `forward_loss(model, batch, dtype)`: one autocast forward pass returning
  a scalar loss tensor.
- `train_step(model, optimizer, batch_iter, ...)`: one optimizer step,
  including gradient accumulation + FSDP-aware `no_sync` + clipping.

**Model contract.** `forward_loss` calls `model(input_ids=...)` and expects
back an object with `.logits` (shape `[B, S, V]`). It does NOT pass `labels=`
to the model and does NOT rely on any built-in loss computation. The loss is
cross-entropy against `batch["labels"]`, computed here. This keeps the loop
architecture-agnostic — any module with `forward(input_ids) -> (.logits)`
works, whether it's HF Qwen3, HF Llama, a custom nn.Module, or your own
Part-2 TransformerLM with a thin adapter.

Why not let HF compute the loss for us? Because the `labels=` contract is
not stable across architectures or transformers versions — some HF models
shift internally, some don't, some accept `shift_labels=`. We shift once in
the dataset (see data.py) and own the loss here.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


# Dtype string -> torch dtype mapping used by the autocast context.
_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}


def forward_loss(
    model: nn.Module,
    batch: dict,
    dtype: str = "bf16",
    device: str = "cuda",
    fused_ce: bool = False,
) -> torch.Tensor:
    """One forward pass under autocast; returns the scalar loss tensor.

    Default path (`fused_ce=False`): computes cross-entropy against the
    (already-shifted) labels from `data.py`. The model is called without
    `labels=` — we only require its forward to return an object with
    `.logits` of shape `[B, S, V]`. This is the architecture-agnostic path.

    Fused path (`fused_ce=True`): the model has been patched by Liger
    Kernel (see `model.build_model`) so that passing `labels=` triggers a
    fused linear+CE Triton kernel inside the LM head. The `[B, S, V]`
    logits tensor never exists. We just consume `out.loss`.

    Args:
        model: the (FSDP-wrapped) model.
        batch: dict with `input_ids` and `labels` (both LongTensor).
            By the dataset contract `labels[t] == input_ids[t+1]`, so no
            further shifting is done here.
        dtype: "bf16" or "fp32". FP8 is handled in `train_step` directly,
            not here.
        device: where to move the batch.
        fused_ce: use the Liger fused-linear-CE path. Requires the model
            to have been built with `build_model(cfg, fused_ce=True)`.

    Returns:
        Scalar loss tensor on `device`.
    """
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    ctx = (torch.autocast(device_type=device, dtype=_DTYPE[dtype])
           if dtype != "fp32" else nullcontext())
    with ctx:
        if fused_ce:
            # Liger-patched Qwen3: loss is computed inside the model using
            # a fused linear+CE Triton kernel that never materializes the
            # `[B, S, V]` logits tensor. Memory peak at the loss step drops
            # ~5x at vocab=151,936 — the binding constraint for this model
            # size on an H100. See README §11 for the rationale.
            #
            # `return_dict=True` is passed to silence a deprecation warning:
            # Liger's `lce_forward` still falls back to `self.config.use_return_dict`
            # when `return_dict` is None, and that property is deprecated in
            # transformers 5.x (use `config.return_dict`). Passing it explicitly
            # skips the fallback.
            out = model(input_ids=input_ids, labels=labels, return_dict=True)
            return out.loss

        out = model(input_ids=input_ids)
        logits = out.logits if hasattr(out, "logits") else out

    # ignore_index=-100 is the standard HF convention for masked positions
    # (e.g. padding tokens, or prompt tokens in SFT). The pretraining packer
    # doesn't emit -100s, but honoring it keeps the loop reusable downstream.
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_iter: Iterator[dict],
    grad_accum: int,
    grad_clip: float,
    dtype: str = "bf16",
    device: str = "cuda",
    fused_ce: bool = False,
) -> tuple[float, float]:
    """One full optimizer step = `grad_accum` micro-batches.

    Returns:
        (mean_loss, grad_norm) — both floats. The loss is the mean over the
        accumulated micro-batches; the grad norm is the unclipped value
        (so spikes are visible in logs even when clipping kicks in).
    """
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for accum_idx in range(grad_accum):
        batch = next(batch_iter)
        loss = forward_loss(model, batch, dtype=dtype, device=device,
                            fused_ce=fused_ce) / grad_accum

        # FSDP: skip cross-rank gradient sync on all but the last micro-batch.
        is_last = (accum_idx == grad_accum - 1)
        if not is_last and hasattr(model, "no_sync"):
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

        total_loss += loss.detach().float().item()

    # Clip the global grad norm (returns the *unclipped* norm for logging).
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    return total_loss, float(grad_norm)
