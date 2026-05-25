"""The inner training loop helpers.

Two responsibilities, factored out of `train.py` so they're testable:

- `forward_loss(model, batch, dtype)`: one autocast forward pass returning
  a scalar loss tensor.
- `train_step(model, optimizer, batch_iter, ...)`: one optimizer step,
  including gradient accumulation + FSDP-aware `no_sync` + clipping.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterator

import torch
import torch.nn as nn


# Dtype string -> torch dtype mapping used by the autocast context.
_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}


def forward_loss(
    model: nn.Module,
    batch: dict,
    dtype: str = "bf16",
    device: str = "cuda",
) -> torch.Tensor:
    """One forward pass under autocast; returns the scalar loss tensor.

    Args:
        model: the (FSDP-wrapped) model.
        batch: dict with `input_ids` and `labels` (both LongTensor).
        dtype: "bf16" or "fp32". FP8 is handled in `train_step` directly,
            not here.
        device: where to move the batch.

    Returns:
        Scalar loss tensor on `device`.
    """
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    if dtype == "fp32":
        out = model(input_ids=input_ids, labels=labels)
    else:
        with torch.autocast(device_type=device, dtype=_DTYPE[dtype]):
            out = model(input_ids=input_ids, labels=labels)
    return out.loss


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_iter: Iterator[dict],
    grad_accum: int,
    grad_clip: float,
    dtype: str = "bf16",
    device: str = "cuda",
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
        loss = forward_loss(model, batch, dtype=dtype, device=device) / grad_accum

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
