"""FSDP2 setup and torchrun distributed initialization.

Two helpers:

- `init_distributed()`: read RANK/LOCAL_RANK/WORLD_SIZE from torchrun env
  vars, call `dist.init_process_group`, set the CUDA device, return ranks.
- `apply_fsdp(model, dtype)`: wrap each transformer decoder layer with
  `fully_shard`, then the root. Mixed-precision policy reduces grads in
  FP32 for stability.

Module 10 covers the FSDP2 internals deeply; this file is the minimum that
makes `torchrun train.py` work correctly on 1, 8, or 64 GPUs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class RankInfo:
    rank: int          # global rank, 0..world_size-1
    local_rank: int    # rank within the local node, 0..local_world-1
    world_size: int    # total processes across all nodes
    is_main: bool      # convenience: rank == 0


def init_distributed() -> RankInfo:
    """Initialize torch.distributed from torchrun env vars.

    Sets the CUDA device to LOCAL_RANK and creates an NCCL process group.
    Safe to call from a non-distributed launch (sets up a single-process
    "world" with rank=0).
    """
    if "RANK" not in os.environ:
        # Not launched via torchrun — single-process fallback. This is for
        # the smoke tests, NOT for actual training. Real runs go through
        # torchrun even at world_size=1 (see Module 08 README §8).
        return RankInfo(rank=0, local_rank=0, world_size=1, is_main=True)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        # CPU-only smoke testing — uses gloo. Don't expect to scale.
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return RankInfo(rank=rank, local_rank=local_rank, world_size=world_size, is_main=(rank == 0))


def apply_fsdp(model: nn.Module, dtype: str = "bf16") -> nn.Module:
    """Wrap a Qwen3-style model with FSDP2 (`fully_shard`).

    Sharding strategy:
    - Each `Qwen3DecoderLayer` is wrapped individually so its params can be
      all-gathered and freed independently. This is the layer-grain sharding
      that gives FSDP its memory efficiency.
    - The root model is then wrapped so embeddings + final norm + LM head
      are also sharded.

    Mixed precision:
    - Params live in `dtype` (BF16) during computation.
    - Gradient reductions happen in FP32 (numerical stability for long runs).
    - Buffers (running stats, RoPE freqs, etc.) stay in FP32.

    Args:
        model: the model returned by `build_model(...)`. Must have
            `model.model.layers` (Qwen3-style) for the per-layer wrap to work.
        dtype: "bf16" or "fp32". FP8 is handled at the autocast level, not
            via FSDP's mixed-precision policy.

    Returns:
        The same model object, now FSDP2-wrapped in place.
    """
    # If not running distributed, FSDP would still work but adds overhead.
    # Skip in single-process so the smoke tests and notebook run unchanged.
    if not dist.is_initialized():
        return model

    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    if dtype == "fp32":
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    else:
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # Per-layer shard. `model.model.layers` is the ModuleList of decoder blocks
    # for Qwen3 (and Llama, and Mistral — every HF causal LM with `model.layers`).
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy)

    # Root shard — gets the embedding, final norm, LM head.
    fully_shard(model, mp_policy=mp_policy)

    return model


def cleanup_distributed():
    """Tear down the process group. Call at the end of `train.py`."""
    if dist.is_initialized():
        dist.destroy_process_group()
