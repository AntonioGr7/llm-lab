"""The inner training loop helpers for continual pretraining (Module 13).

Copied from Module 11/15's `loop.py` — and it works for CPT unchanged, which is
the point: continual pretraining *is* pretraining, just resumed from a finished
checkpoint instead of random init.

Two things to note about how the shared code behaves for CPT:

1. `forward_loss` calls `batch.get("attention_mask")`. CPT samples come from the
   packed indexed corpus (Module 12) — fixed-length blocks with NO padding — so
   there is no attention_mask and it is `None`. The HF causal LM then does
   ordinary full-context attention, exactly as in pretraining.

2. `ignore_index=-100` is a no-op here: CPT labels are the next-token shift of
   the whole sequence (no masking), so cross-entropy averages over every token.
   (Contrast Module 15's SFT, where -100 masks the non-assistant positions.)
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}


def forward_loss(
    model: nn.Module,
    batch: dict,
    dtype: str = "bf16",
    device: str = "cuda",
    fused_ce: bool = False,
) -> torch.Tensor:
    """One forward pass under autocast; returns the scalar loss tensor.

    Default path (`fused_ce=False`): the model is called WITHOUT `labels=`,
    we read `out.logits`, and compute `F.cross_entropy(..., ignore_index=-100)`
    out here. CPT packs fixed-length blocks so `ignore_index=-100` is a no-op
    (no masking), but honoring it keeps the loop reusable downstream.

    Fused path (`fused_ce=True`): the model has been patched by Liger Kernel
    (see `model.build_model`) so that passing `labels=` triggers a fused
    linear+CE Triton kernel inside the LM head. The `[B, S, V]` logits tensor
    never exists. We just consume `out.loss`.

    The model contract is identical to Module 11: any module with
    `forward(input_ids, attention_mask=...) -> (.logits)` works. HF causal
    LMs satisfy this directly; custom modules need to accept (and ideally
    honor) the kwarg.
    """
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)

    ctx = (torch.autocast(device_type=device, dtype=_DTYPE[dtype])
           if dtype != "fp32" else nullcontext())
    with ctx:
        if fused_ce:
            # Liger-patched model: loss is computed inside the model using a
            # fused linear+CE Triton kernel that never materializes the
            # `[B, S, V]` logits tensor. `return_dict=True` silences a
            # transformers 5.x deprecation in Liger's `lce_forward` fallback.
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            return out.loss

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out

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
    """One full optimizer step = `grad_accum` micro-batches. Mirrors Module 11.

    Returns:
        (mean_loss, grad_norm) — both floats. The loss is the mean over the
        accumulated micro-batches; the grad norm is the *unclipped* value
        (so spikes are visible in logs even when clipping kicks in).

    SFT note: each micro-batch's loss is `F.cross_entropy(...).mean()`
    averaged over the assistant tokens *in that micro-batch*. Dividing by
    `grad_accum` and summing implicitly equal-weights each micro-batch —
    if assistant-token counts vary wildly across micro-batches, the
    effective per-token loss is slightly biased. For typical chat data
    this asymmetry is small (<5%); for pathological datasets you'd want
    a token-weighted reduction, but the simpler form is standard practice.
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

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    return total_loss, float(grad_norm)
